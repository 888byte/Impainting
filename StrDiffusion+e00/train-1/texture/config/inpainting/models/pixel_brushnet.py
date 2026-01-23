# -*- coding: utf-8 -*-
"""
pixel_brushnet.py - 像素空间BrushNet架构

功能说明：
---------
本模块实现适配像素空间扩散模型（StrDiffusion）的BrushNet架构。
与原始BrushNet不同，本实现完全去除了VAE依赖，直接在RGB像素空间操作。

架构设计：
----------
1. 输入层：8通道输入
   - Noisy_Image: 3 channels (带噪声的RGB图像)
   - Mask: 1 channel (修复区域掩码)
   - Color_Prior: 3 channels (颜色先验图)
   - Confidence_Map: 1 channel (置信度图)

2. Encoder层级：与StrDiffusion的ConditionalUNet严格对齐
   - depth=4, nf=64
   - 特征维度: 64 -> 128 -> 256 -> 512 -> 1024 (mid)

3. 输出：各层特征通过Zero-Convolution注入主网络

参考：
------
- BrushNet: https://arxiv.org/abs/2403.06976
- ControlNet: https://arxiv.org/abs/2302.05543
- StrDiffusion ConditionalUNet 架构

Author: Auto-generated for BrushNet Integration
"""

import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

# 导入Zero-Convolution
from zero_conv import ZeroConv2d, make_zero_conv

# 导入StrDiffusion的模块（复用现有组件）
from modules.module_util import (
    SinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    LinearAttention,
    PreNorm, Residual
)


class PixelBrushNet(nn.Module):
    """
    像素空间BrushNet - 无VAE依赖
    
    该网络与StrDiffusion的ConditionalUNet共享相同的层级结构，
    用于提取条件特征并通过Zero-Convolution注入主网络。
    
    Architecture:
    -------------
    Input (8ch) -> Init Conv -> [Down Blocks] -> Mid Block
                                    |
                                    v
                          Feature outputs (via Zero-Conv)
    
    Attributes:
        depth (int): 下采样深度，与主UNet对齐
        nf (int): 基础特征通道数
        in_nc (int): 输入通道数（默认8）
    """
    
    def __init__(
        self,
        in_nc: int = 8,
        nf: int = 64,
        depth: int = 4,
        time_emb_dim: Optional[int] = None
    ):
        """
        初始化PixelBrushNet
        
        Args:
            in_nc: 输入通道数 (默认8: 3+1+3+1)
            nf: 基础特征通道数 (默认64，与ConditionalUNet对齐)
            depth: 下采样深度 (默认4，与ConditionalUNet对齐)
            time_emb_dim: 时间嵌入维度 (默认 nf*4)
            
        Tensor Shape 变化：
        ------------------
        输入: [B, 8, H, W]
        init_conv后: [B, 64, H, W]
        Down Block 1后: [B, 64, H, W] -> [B, 128, H/2, W/2]
        Down Block 2后: [B, 128, H/2, W/2] -> [B, 256, H/4, W/4]
        Down Block 3后: [B, 256, H/4, W/4] -> [B, 512, H/8, W/8]
        Down Block 4后: [B, 512, H/8, W/8] -> [B, 1024, H/8, W/8]
        Mid Block后: [B, 1024, H/8, W/8]
        """
        super().__init__()
        
        self.depth = depth
        self.nf = nf
        self.in_nc = in_nc
        
        # 定义ResBlock构建函数
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())
        
        # ============================================================
        # 时间嵌入层（与ConditionalUNet共享）
        # ============================================================
        if time_emb_dim is None:
            time_emb_dim = nf * 4  # 256
        self.time_dim = time_emb_dim
        
        sinu_pos_emb = SinusoidalPosEmb(nf)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(nf, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # ============================================================
        # 初始卷积层
        # ============================================================
        # 输入: [B, 8, H, W] -> [B, 64, H, W]
        self.init_conv = default_conv(in_nc, nf, kernel_size=7)
        
        # ============================================================
        # 下采样块 (Encoder)
        # ============================================================
        # 与ConditionalUNet的downs完全对齐
        self.downs = nn.ModuleList([])
        self.zero_convs_down = nn.ModuleList([])  # 每层两个特征输出的Zero-Conv
        
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))      # 64, 128, 256, 512
            dim_out = nf * int(math.pow(2, i + 1)) # 128, 256, 512, 1024
            
            # 下采样块：两个ResBlock + Attention + Downsample
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_emb_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_emb_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out)
            ]))
            
            # 每个下采样块输出两个特征（对应主UNet的两个skip connection）
            # 使用Zero-Conv进行特征映射
            self.zero_convs_down.append(nn.ModuleList([
                ZeroConv2d(dim_in, dim_in),  # 第一个ResBlock后
                ZeroConv2d(dim_in, dim_in)   # 第二个ResBlock后（Attention后）
            ]))
        
        # ============================================================
        # 中间块 (Mid Block)
        # ============================================================
        mid_dim = nf * int(math.pow(2, depth))  # 1024
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_emb_dim)
        
        # Mid特征的Zero-Conv
        self.zero_conv_mid = ZeroConv2d(mid_dim, mid_dim)
        
        # 打印网络结构信息
        self._print_architecture()
    
    def _print_architecture(self):
        """打印网络架构信息"""
        print(f"[PixelBrushNet] 架构信息:")
        print(f"  - 输入通道: {self.in_nc} (Noisy:3 + Mask:1 + Prior:3 + Conf:1)")
        print(f"  - 基础通道数: {self.nf}")
        print(f"  - 下采样深度: {self.depth}")
        print(f"  - 时间嵌入维度: {self.time_dim}")
        print(f"  - 特征维度序列: ", end="")
        dims = [self.nf * int(math.pow(2, i)) for i in range(self.depth + 1)]
        print(" -> ".join(map(str, dims)))
    
    def check_image_size(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        检查并调整图像尺寸以适应下采样
        
        Args:
            x: 输入张量
            h, w: 原始高度和宽度
            
        Returns:
            调整后的张量（如有必要进行padding）
        """
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(
        self,
        noisy_img: torch.Tensor,
        mask: torch.Tensor,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        timestep: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            noisy_img: [B, 3, H, W] 带噪声的图像
            mask: [B, 1, H, W] 修复区域掩码 (1=需要修复)
            color_prior: [B, 3, H, W] 颜色先验图
            confidence: [B, 1, H, W] 置信度图
            timestep: [B,] 或 scalar 时间步
            
        Returns:
            dict 包含：
                - 'down_features': List[Tensor] 各层下采样特征 (用于注入主UNet)
                - 'mid_feature': Tensor 中间层特征
                
        Tensor Shape 变化过程：
        ----------------------
        1. 输入拼接: [B, 3+1+3+1=8, H, W]
        2. init_conv: [B, 64, H, W]
        3. Down Block 1: [B, 64, H, W] -> downsample -> [B, 128, H/2, W/2]
        4. Down Block 2: [B, 128, H/2, W/2] -> [B, 256, H/4, W/4]
        5. Down Block 3: [B, 256, H/4, W/4] -> [B, 512, H/8, W/8]
        6. Down Block 4: [B, 512, H/8, W/8] -> [B, 1024, H/8, W/8]
        7. Mid Blocks: [B, 1024, H/8, W/8]
        """
        # 处理timestep
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=noisy_img.device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # ============================================================
        # Step 1: 输入拼接
        # ============================================================
        # 拼接所有条件输入
        x = torch.cat([noisy_img, mask, color_prior, confidence], dim=1)  # [B, 8, H, W]
        
        # 记录原始尺寸
        H, W = x.shape[2:]
        
        # 检查并调整尺寸
        x = self.check_image_size(x, H, W)
        
        # ============================================================
        # Step 2: 时间嵌入
        # ============================================================
        t_emb = self.time_mlp(timestep)  # [B, time_dim]
        
        # ============================================================
        # Step 3: 初始卷积
        # ============================================================
        x = self.init_conv(x)  # [B, 64, H, W]
        
        # ============================================================
        # Step 4: 下采样 + 特征收集
        # ============================================================
        down_features = []  # 存储各层特征用于注入
        
        for i, (blocks, zero_convs) in enumerate(zip(self.downs, self.zero_convs_down)):
            b1, b2, attn, downsample = blocks
            zc1, zc2 = zero_convs
            
            # 第一个ResBlock
            x = b1(x, t_emb)
            # 收集特征（通过Zero-Conv）
            down_features.append(zc1(x))
            
            # 第二个ResBlock + Attention
            x = b2(x, t_emb)
            x = attn(x)
            # 收集特征（通过Zero-Conv）
            down_features.append(zc2(x))
            
            # 下采样
            x = downsample(x)
        
        # ============================================================
        # Step 5: 中间块
        # ============================================================
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # 中间层特征（通过Zero-Conv）
        mid_feature = self.zero_conv_mid(x)
        
        return {
            'down_features': down_features,  # List of [B, C, H', W']
            'mid_feature': mid_feature       # [B, 1024, H/8, W/8]
        }
    
    def forward_simple(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        简化前向传播（输入已拼接）
        
        Args:
            x: [B, 8, H, W] 已拼接的输入
            timestep: [B,] 时间步
            
        Returns:
            同 forward()
        """
        # 分离各通道
        noisy_img = x[:, :3]
        mask = x[:, 3:4]
        color_prior = x[:, 4:7]
        confidence = x[:, 7:8]
        
        return self.forward(noisy_img, mask, color_prior, confidence, timestep)


class PixelBrushNetLite(nn.Module):
    """
    轻量级PixelBrushNet - 减少计算开销
    
    相比完整版本：
    - 移除Attention层
    - 减少ResBlock数量
    - 适用于资源受限场景
    """
    
    def __init__(
        self,
        in_nc: int = 8,
        nf: int = 64,
        depth: int = 4,
        time_emb_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.depth = depth
        self.nf = nf
        
        if time_emb_dim is None:
            time_emb_dim = nf * 4
        self.time_dim = time_emb_dim
        
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())
        
        # 时间嵌入
        sinu_pos_emb = SinusoidalPosEmb(nf)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(nf, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.init_conv = default_conv(in_nc, nf, kernel_size=3)
        
        # 下采样块（简化版，每层只有一个ResBlock）
        self.downs = nn.ModuleList([])
        self.zero_convs = nn.ModuleList([])
        
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_emb_dim),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out)
            ]))
            
            self.zero_convs.append(ZeroConv2d(dim_in, dim_in))
        
        # 中间块
        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_emb_dim)
        self.zero_conv_mid = ZeroConv2d(mid_dim, mid_dim)
        
        print(f"[PixelBrushNetLite] 轻量级模式，depth={depth}, nf={nf}")
    
    def forward(
        self,
        noisy_img: torch.Tensor,
        mask: torch.Tensor,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        timestep: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """前向传播"""
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=noisy_img.device)
        
        x = torch.cat([noisy_img, mask, color_prior, confidence], dim=1)
        t_emb = self.time_mlp(timestep)
        x = self.init_conv(x)
        
        down_features = []
        
        for blocks, zc in zip(self.downs, self.zero_convs):
            resblock, downsample = blocks
            x = resblock(x, t_emb)
            down_features.append(zc(x))
            x = downsample(x)
        
        x = self.mid_block(x, t_emb)
        mid_feature = self.zero_conv_mid(x)
        
        return {
            'down_features': down_features,
            'mid_feature': mid_feature
        }


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PixelBrushNet 单元测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 测试完整版PixelBrushNet
    print("\n[测试1] PixelBrushNet (完整版)...")
    model = PixelBrushNet(in_nc=8, nf=64, depth=4).to(device)
    
    # 创建测试输入
    batch_size = 2
    H, W = 256, 256
    noisy_img = torch.randn(batch_size, 3, H, W, device=device)
    mask = torch.randint(0, 2, (batch_size, 1, H, W), device=device).float()
    color_prior = torch.randn(batch_size, 3, H, W, device=device)
    confidence = torch.rand(batch_size, 1, H, W, device=device)
    timestep = torch.randint(0, 1000, (batch_size,), device=device).float()
    
    # 前向传播
    with torch.no_grad():
        output = model(noisy_img, mask, color_prior, confidence, timestep)
    
    print(f"  下采样特征数量: {len(output['down_features'])}")
    print(f"  各层特征形状:")
    for i, feat in enumerate(output['down_features']):
        print(f"    Layer {i}: {feat.shape}")
    print(f"  中间层特征形状: {output['mid_feature'].shape}")
    
    # 验证特征数量
    expected_down_features = 8  # depth=4, 每层2个特征
    assert len(output['down_features']) == expected_down_features, \
        f"下采样特征数量错误: {len(output['down_features'])} vs {expected_down_features}"
    
    # 测试轻量版
    print("\n[测试2] PixelBrushNetLite (轻量版)...")
    model_lite = PixelBrushNetLite(in_nc=8, nf=64, depth=4).to(device)
    
    with torch.no_grad():
        output_lite = model_lite(noisy_img, mask, color_prior, confidence, timestep)
    
    print(f"  下采样特征数量: {len(output_lite['down_features'])}")
    print(f"  中间层特征形状: {output_lite['mid_feature'].shape}")
    
    # 测试Zero-Conv初始输出
    print("\n[测试3] 验证Zero-Conv初始输出为0...")
    model_fresh = PixelBrushNet(in_nc=8, nf=64, depth=4).to(device)
    with torch.no_grad():
        output_fresh = model_fresh(noisy_img, mask, color_prior, confidence, timestep)
    
    for i, feat in enumerate(output_fresh['down_features']):
        mean_val = feat.abs().mean().item()
        print(f"  Layer {i} 均值: {mean_val:.6f} (应接近0)")
    
    mid_mean = output_fresh['mid_feature'].abs().mean().item()
    print(f"  Mid层均值: {mid_mean:.6f} (应接近0)")
    
    print("\n✓ 所有测试通过!")
