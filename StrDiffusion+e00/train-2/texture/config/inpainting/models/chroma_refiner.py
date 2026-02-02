# -*- coding: utf-8 -*-
"""
chroma_refiner.py - 轻量级色度精炼网络

功能说明：
---------
基于 MDTA-lite + GDFN-lite 的轻量 Transformer 结构，
在 Lab 的 Δab（色度差分）空间做有界残差修正。

核心设计：
----------
1. MDTA-lite: Multi-Dconv Transposed Attention (轻量版)
   - 跨通道注意力 + depthwise conv 引入局部感知
2. GDFN-lite: Gated-Dconv Feed-forward Network (轻量版)
   - 门控 FFN，expand ratio = 2
3. 输出限幅: tanh × learnable_scale，防止 ab 爆炸
4. 门控机制: 只在低置信区域生效

输入通道组成 (Cin=6):
--------------------
| 通道 | 内容         | 计算方式                    |
|------|--------------|----------------------------|
| 0-1  | delta_ab_norm| (ab_prior - ab_img) / 128  |
| 2    | L_norm       | L_img / 100                |
| 3    | conf_final   | Stage1 置信度              |
| 4    | mask_hole    | 1=hole                     |
| 5    | conf_lut     | LUT 置信度 (可选)          |

Author: ChromaRefiner for Mural Inpainting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization (用于图像特征)
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MDTALite(nn.Module):
    """
    Multi-Dconv Transposed Attention (Lite)
    
    轻量版 MDTA:
    - 使用 depthwise conv 引入局部空间感知
    - 跨通道注意力机制
    - 相比原版 Restormer 减少头数和参数量
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 4, 
        bias: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Q, K, V 投影
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        
        # Depthwise conv 用于局部空间感知
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, 
            kernel_size=3, stride=1, padding=1, 
            groups=dim * 3, bias=bias
        )
        
        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 跨通道注意力 (Transposed Attention)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        
        return out


class GDFNLite(nn.Module):
    """
    Gated-Dconv Feed-forward Network (Lite)
    
    轻量版 GDFN:
    - 门控机制控制信息流
    - Depthwise conv 用于局部空间感知
    - expand_ratio = 2 (相比原版减少)
    """
    def __init__(
        self, 
        dim: int, 
        expand_ratio: int = 2, 
        bias: bool = True
    ):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        
        self.dwconv = nn.Conv2d(
            hidden_dim * 2, hidden_dim * 2, 
            kernel_size=3, stride=1, padding=1, 
            groups=hidden_dim * 2, bias=bias
        )
        
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # 门控机制
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class TransformerBlockLite(nn.Module):
    """
    轻量 Transformer Block
    
    结构:
    - LayerNorm -> MDTA-lite -> 残差
    - LayerNorm -> GDFN-lite -> 残差
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 4, 
        expand_ratio: int = 2,
        bias: bool = True
    ):
        super().__init__()
        
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTALite(dim, num_heads=num_heads, bias=bias)
        
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFNLite(dim, expand_ratio=expand_ratio, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MDTALiteChromaRefiner(nn.Module):
    """
    MDTA-lite 色度精炼网络
    
    在 Lab 的 Δab 空间做有界残差修正，使 Stage1 生成的 color_prior
    更加稳定、去噪、去伪影。
    
    Args:
        in_channels: 输入通道数 (默认6: delta_ab(2)+L(1)+conf(1)+mask(1)+conf_lut(1))
        hidden_channels: 隐藏层通道数 (默认32)
        num_blocks: Transformer block 数量 (默认1)
        num_heads: 注意力头数 (默认4)
        expand_ratio: GDFN 扩展比例 (默认2)
        output_scale: tanh 后的缩放因子 (默认0.3)
        learnable_scale: 缩放因子是否可学习 (默认True)
    
    输入:
        ref_in: [B, Cin, H, W] 拼接后的输入特征
    
    输出:
        delta_ab_update_norm: [B, 2, H, W] 归一化空间的 ab 更新量
    """
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 32,
        num_blocks: int = 1,
        num_heads: int = 4,
        expand_ratio: int = 2,
        output_scale: float = 0.3,
        learnable_scale: bool = True,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        
        # 输入投影: Cin -> C
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlockLite(
                dim=hidden_channels, 
                num_heads=num_heads, 
                expand_ratio=expand_ratio,
                bias=bias
            )
            for _ in range(num_blocks)
        ])
        
        # 输出投影: C -> 2 (delta_a, delta_b)
        self.output_proj = nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=bias)
        
        # 输出缩放因子
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(output_scale))
        else:
            self.register_buffer('scale', torch.tensor(output_scale))
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用较小的初始化值，保证初始输出接近零
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 输出投影使用更小的初始化
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, ref_in: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            ref_in: [B, Cin, H, W] 输入特征
                - 通道 0-1: delta_ab_norm = (ab_prior - ab_img) / 128
                - 通道 2: L_norm = L_img / 100
                - 通道 3: conf_final (置信度)
                - 通道 4: mask_hole (1=hole)
                - 通道 5: conf_lut (LUT置信度, 可选)
        
        Returns:
            delta_ab_update_norm: [B, 2, H, W] 归一化的 ab 更新量
        """
        # 输入投影
        x = self.input_proj(ref_in)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出投影
        delta_raw = self.output_proj(x)
        
        # tanh 限幅 + 缩放
        delta_ab_update_norm = torch.tanh(delta_raw) * self.scale
        
        return delta_ab_update_norm
    
    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_blocks={self.num_blocks}, "
            f"scale={self.scale.item():.3f}"
        )


def create_chroma_refiner(opt: dict) -> Optional[MDTALiteChromaRefiner]:
    """
    工厂函数：根据配置创建 ChromaRefiner
    
    Args:
        opt: 配置字典，包含 'chroma_refiner' 子字典:
            - enabled: 是否启用
            - in_channels: 输入通道数 (默认6)
            - hidden_channels: 隐藏层通道数 (默认32)
            - num_blocks: Transformer block 数量 (默认1)
            - output_scale: 输出缩放因子 (默认0.3)
    
    Returns:
        ChromaRefiner 模型或 None (如果未启用)
    """
    refiner_opt = opt.get('chroma_refiner', {})
    
    if not refiner_opt.get('enabled', False):
        return None
    
    return MDTALiteChromaRefiner(
        in_channels=refiner_opt.get('in_channels', 6),
        hidden_channels=refiner_opt.get('hidden_channels', 32),
        num_blocks=refiner_opt.get('num_blocks', 1),
        num_heads=refiner_opt.get('num_heads', 4),
        expand_ratio=refiner_opt.get('expand_ratio', 2),
        output_scale=refiner_opt.get('output_scale', 0.3),
        learnable_scale=refiner_opt.get('learnable_scale', True)
    )


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("MDTALiteChromaRefiner 单元测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    height, width = 256, 256
    in_channels = 6
    
    # 创建模型
    model = MDTALiteChromaRefiner(
        in_channels=in_channels,
        hidden_channels=32,
        num_blocks=1,
        num_heads=4,
        expand_ratio=2,
        output_scale=0.3,
        learnable_scale=True
    )
    
    print(f"\n模型结构:")
    print(model)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 测试前向传播
    print(f"\n测试前向传播...")
    # 构造模拟输入
    # delta_ab_norm (2ch) + L_norm (1ch) + conf (1ch) + mask (1ch) + conf_lut (1ch)
    ref_in = torch.randn(batch_size, in_channels, height, width)
    
    model.eval()
    with torch.no_grad():
        output = model(ref_in)
    
    print(f"  输入形状: {ref_in.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"  输出均值: {output.mean().item():.4f}")
    print(f"  输出标准差: {output.std().item():.4f}")
    
    # 验证输出范围 (应该在 tanh * scale 范围内)
    scale = model.scale.item()
    assert output.min() >= -scale - 0.01, f"输出最小值超出范围: {output.min()}"
    assert output.max() <= scale + 0.01, f"输出最大值超出范围: {output.max()}"
    print(f"  ✓ 输出在 [-{scale:.2f}, {scale:.2f}] 范围内")
    
    # 测试梯度
    print(f"\n测试梯度回传...")
    model.train()
    ref_in.requires_grad = True
    output = model(ref_in)
    loss = output.mean()
    loss.backward()
    
    assert ref_in.grad is not None, "输入梯度为 None"
    print(f"  ✓ 梯度正常回传")
    
    # 测试 GPU (如果可用)
    if torch.cuda.is_available():
        print(f"\n测试 GPU...")
        device = torch.device('cuda')
        model_gpu = model.to(device)
        ref_in_gpu = ref_in.detach().to(device)
        
        with torch.no_grad():
            output_gpu = model_gpu(ref_in_gpu)
        
        print(f"  ✓ GPU 推理成功")
        print(f"  输出形状: {output_gpu.shape}")
    
    # 测试工厂函数
    print(f"\n测试工厂函数...")
    opt_enabled = {
        'chroma_refiner': {
            'enabled': True,
            'in_channels': 6,
            'hidden_channels': 32,
            'num_blocks': 2,
            'output_scale': 0.5
        }
    }
    opt_disabled = {
        'chroma_refiner': {
            'enabled': False
        }
    }
    
    model_from_opt = create_chroma_refiner(opt_enabled)
    assert model_from_opt is not None, "工厂函数应返回模型"
    print(f"  ✓ 启用配置: 创建成功 (num_blocks={model_from_opt.num_blocks})")
    
    model_none = create_chroma_refiner(opt_disabled)
    assert model_none is None, "工厂函数应返回 None"
    print(f"  ✓ 禁用配置: 返回 None")
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
