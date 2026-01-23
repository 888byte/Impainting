# -*- coding: utf-8 -*-
"""
brushnet_wrapper.py - BrushNet与StrDiffusion集成包装器

功能说明：
---------
本模块提供BrushNet与StrDiffusion主UNet的集成包装器。
通过Zero-Convolution将BrushNet的条件特征注入主UNet的对应层级。

集成方式：
----------
1. BrushNet接收条件输入（noisy_img, mask, color_prior, confidence）
2. BrushNet产生多尺度特征
3. 特征通过加法融合到主UNet的对应层级
4. 主UNet产生最终去噪预测

注意事项：
----------
- 本模块独立于原始代码，不修改ConditionalUNet
- 通过包装机制实现特征注入
- 可通过配置开关BrushNet

Author: Auto-generated for BrushNet Integration
"""

import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union

# 导入BrushNet
from pixel_brushnet import PixelBrushNet, PixelBrushNetLite
from zero_conv import ZeroConv2d

# 导入原始UNet模块
from modules.module_util import (
    SinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    LinearAttention,
    PreNorm, Residual
)


class ConditionalUNetWithBrushNet(nn.Module):
    """
    集成BrushNet的条件UNet
    
    该类包装原始的ConditionalUNet，添加BrushNet条件注入功能。
    
    Architecture:
    -------------
    1. BrushNet处理条件输入 -> 生成多尺度特征
    2. 主UNet正常前向传播
    3. 在主UNet的下采样阶段，通过加法融合BrushNet特征
    4. 中间层也进行特征融合
    
    这种设计确保：
    - 原始UNet代码完全不变
    - BrushNet特征通过Zero-Conv初始化为0，训练初期不影响主网络
    - 逐步学习有效的条件控制
    
    Attributes:
        base_unet: 主UNet网络
        brushnet: BrushNet条件网络
        brushnet_enabled: 是否启用BrushNet
    """
    
    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        depth: int = 4,
        brushnet_in_nc: int = 8,
        brushnet_enabled: bool = True,
        brushnet_lite: bool = False
    ):
        """
        初始化集成UNet
        
        Args:
            in_nc: 主UNet输入通道数 (默认3)
            out_nc: 主UNet输出通道数 (默认3)
            nf: 基础特征通道数 (默认64)
            depth: 下采样深度 (默认4)
            brushnet_in_nc: BrushNet输入通道数 (默认8)
            brushnet_enabled: 是否启用BrushNet (默认True)
            brushnet_lite: 是否使用轻量版BrushNet (默认False)
        """
        super().__init__()
        
        self.depth = depth
        self.nf = nf
        self.brushnet_enabled = brushnet_enabled
        
        # ============================================================
        # 主UNet（复用ConditionalUNet架构）
        # ============================================================
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())
        
        # 时间嵌入
        time_dim = nf * 4
        sinu_pos_emb = SinusoidalPosEmb(nf)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(nf, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # 初始卷积（输入 xt 和 cond 拼接）
        self.init_conv = default_conv(in_nc * 2, nf, 7)
        
        # 下采样块
        self.downs = nn.ModuleList([])
        
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out)
            ]))
        
        # 上采样块
        self.ups = nn.ModuleList([])
        
        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))
        
        # 中间块
        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        
        # 最终输出层
        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)
        
        # ============================================================
        # BrushNet
        # ============================================================
        if brushnet_enabled:
            if brushnet_lite:
                self.brushnet = PixelBrushNetLite(
                    in_nc=brushnet_in_nc, nf=nf, depth=depth
                )
            else:
                self.brushnet = PixelBrushNet(
                    in_nc=brushnet_in_nc, nf=nf, depth=depth
                )
            print(f"[ConditionalUNetWithBrushNet] BrushNet已启用 (lite={brushnet_lite})")
        else:
            self.brushnet = None
            print(f"[ConditionalUNetWithBrushNet] BrushNet未启用")
    
    def check_image_size(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """检查并调整图像尺寸"""
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(
        self,
        xt: torch.Tensor,
        cond: torch.Tensor,
        time: Union[int, float, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        color_prior: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        S: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            xt: [B, 3, H, W] 当前噪声状态
            cond: [B, 3, H, W] 条件输入
            time: 时间步
            mask: [B, 1, H, W] 修复区域掩码 (可选)
            color_prior: [B, 3, H, W] 颜色先验图 (可选)
            confidence: [B, 1, H, W] 置信度图 (可选)
            S: 结构引导图 (与原ConditionalUNet兼容，这里忽略)
            
        Returns:
            output: [B, 3, H, W] 去噪预测
            output: [B, 3, H, W] 同上（与原接口兼容）
        
        注意：
        -----
        当 brushnet_enabled=True 且提供了 mask, color_prior, confidence 时，
        将使用BrushNet特征注入。否则退化为普通UNet。
        """
        # 处理时间步
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=xt.device)
        
        # 输入预处理（与原ConditionalUNet一致）
        x = xt - cond
        x = torch.cat([x, cond], dim=1)  # [B, 6, H, W]
        
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        
        # 时间嵌入
        t = self.time_mlp(time)
        
        # ============================================================
        # 获取BrushNet特征（如果启用）
        # ============================================================
        brushnet_features = None
        brushnet_mid = None
        
        if (self.brushnet_enabled and self.brushnet is not None and 
            mask is not None and color_prior is not None and confidence is not None):
            
            # 调整条件输入尺寸
            mask = self.check_image_size(mask, H, W)
            color_prior = self.check_image_size(color_prior, H, W)
            confidence = self.check_image_size(confidence, H, W)
            
            # BrushNet前向传播
            bn_output = self.brushnet(xt, mask, color_prior, confidence, time)
            brushnet_features = bn_output['down_features']
            brushnet_mid = bn_output['mid_feature']
        
        # ============================================================
        # 主UNet下采样
        # ============================================================
        x = self.init_conv(x)
        x_ = x.clone()  # 保存用于最终拼接
        
        h = []  # skip connections
        
        bn_idx = 0  # BrushNet特征索引
        for i, blocks in enumerate(self.downs):
            b1, b2, attn, downsample = blocks
            
            # 第一个ResBlock
            x = b1(x, t)
            # 注入BrushNet特征
            if brushnet_features is not None and bn_idx < len(brushnet_features):
                x = x + brushnet_features[bn_idx]
                bn_idx += 1
            h.append(x)
            
            # 第二个ResBlock + Attention
            x = b2(x, t)
            x = attn(x)
            # 注入BrushNet特征
            if brushnet_features is not None and bn_idx < len(brushnet_features):
                x = x + brushnet_features[bn_idx]
                bn_idx += 1
            h.append(x)
            
            # 下采样
            x = downsample(x)
        
        # ============================================================
        # 中间块
        # ============================================================
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        # 注入BrushNet中间层特征
        if brushnet_mid is not None:
            x = x + brushnet_mid
        
        # ============================================================
        # 主UNet上采样
        # ============================================================
        for blocks in self.ups:
            b1, b2, attn, upsample = blocks
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)
            
            x = upsample(x)
        
        # ============================================================
        # 最终输出
        # ============================================================
        x = torch.cat([x, x_], dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        # 裁剪回原始尺寸
        x = x[..., :H, :W]
        
        return x, x


def create_brushnet_unet(opt: dict) -> nn.Module:
    """
    工厂函数：根据配置创建集成BrushNet的UNet
    
    Args:
        opt: 配置字典，包含以下键：
            - network_G.setting.in_nc: 输入通道
            - network_G.setting.out_nc: 输出通道
            - network_G.setting.nf: 基础通道数
            - network_G.setting.depth: 下采样深度
            - brushnet.enabled: 是否启用BrushNet
            - brushnet.in_nc: BrushNet输入通道数
            - brushnet.lite: 是否使用轻量版
            
    Returns:
        集成BrushNet的UNet模型
    """
    network_opt = opt.get('network_G', {}).get('setting', {})
    brushnet_opt = opt.get('brushnet', {})
    
    model = ConditionalUNetWithBrushNet(
        in_nc=network_opt.get('in_nc', 3),
        out_nc=network_opt.get('out_nc', 3),
        nf=network_opt.get('nf', 64),
        depth=network_opt.get('depth', 4),
        brushnet_in_nc=brushnet_opt.get('in_nc', 8),
        brushnet_enabled=brushnet_opt.get('enabled', True),
        brushnet_lite=brushnet_opt.get('lite', False)
    )
    
    return model


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ConditionalUNetWithBrushNet 单元测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 测试集成模型
    print("\n[测试1] ConditionalUNetWithBrushNet...")
    model = ConditionalUNetWithBrushNet(
        in_nc=3, out_nc=3, nf=64, depth=4,
        brushnet_in_nc=8, brushnet_enabled=True
    ).to(device)
    
    # 创建测试输入
    batch_size = 2
    H, W = 256, 256
    xt = torch.randn(batch_size, 3, H, W, device=device)
    cond = torch.randn(batch_size, 3, H, W, device=device)
    time = torch.randint(0, 1000, (batch_size,), device=device).float()
    mask = torch.randint(0, 2, (batch_size, 1, H, W), device=device).float()
    color_prior = torch.randn(batch_size, 3, H, W, device=device)
    confidence = torch.rand(batch_size, 1, H, W, device=device)
    
    # 带BrushNet的前向传播
    with torch.no_grad():
        output1, output2 = model(
            xt, cond, time,
            mask=mask, color_prior=color_prior, confidence=confidence
        )
    
    print(f"  输入形状: xt={xt.shape}, cond={cond.shape}")
    print(f"  输出形状: {output1.shape}")
    assert output1.shape == (batch_size, 3, H, W), "输出形状错误"
    
    # 不带BrushNet条件的前向传播（退化为普通UNet）
    print("\n[测试2] 不带BrushNet条件...")
    with torch.no_grad():
        output_no_bn, _ = model(xt, cond, time)
    
    print(f"  输出形状: {output_no_bn.shape}")
    
    # 测试BrushNet禁用模式
    print("\n[测试3] BrushNet禁用模式...")
    model_no_bn = ConditionalUNetWithBrushNet(
        in_nc=3, out_nc=3, nf=64, depth=4,
        brushnet_enabled=False
    ).to(device)
    
    with torch.no_grad():
        output_disabled, _ = model_no_bn(xt, cond, time)
    
    print(f"  输出形状: {output_disabled.shape}")
    
    print("\n✓ 所有测试通过!")
