"""
去噪 U-Net 结构定义（含注意力、条件归一化等模块）。
该文件是模型主干结构之一，供 networks.define_G 调用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools
import numpy as np
from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)

from torch.nn import init

# ✅ PixelBrushNet（像素空间 BrushNet 核心分支，不含 VAE）
# 你需要保证 ../pixel_brushnet.py 存在并实现 PixelBrushNet / PixelBrushNetConfig
from ..pixel_brushnet import PixelBrushNet, PixelBrushNetConfig


class BaseNetwork(nn.Module):
    """基础网络封装：提供初始化与参数统计。"""
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


import re
def PositionalNorm2d(x, epsilon=1e-5):
    """通道维度的归一化（用于稳定特征分布）。"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == "position":
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        nhidden = 16
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class SPADEBlock(BaseNetwork):
    def __init__(self, fin, fout, semantic_nc, norm_G='spectralspadeposition3x3'):
        super(SPADEBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        hidden = fin

        self.conv_0 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, hidden, kernel_size=1, bias=False)

        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        spade_config_str = norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, hidden, semantic_nc)
        self.norm_1 = SPADE(spade_config_str, hidden, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, semantic_nc)

        self.init_weights()

    def forward(self, x, seg):
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        return x + dx

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ConditionalUNet(nn.Module):
    """
    ✅ 已修改（加中文注释版）：
    1) 支持 PixelBrushNet 分支（像素空间、无 VAE）
    2) forward 预留两个输入：
       - color_prior: 颜色先验图 (B,3,H,W)
       - confidence : 置信度图   (B,1,H,W)
    3) Brush residual 的注入位置（与结构 guide 对齐）：
       - down：每个 level 的 downsample 后、guide(x,S) 前
       - mid ：mid_block2 后
       - up  ：每个 level 的 attn 后、upsample 前
    4) PixelBrushNet 的启用开关不在这里硬编码，
       由 networks.py 从 yml 读取后调用 enable_pixel_brushnet(cfg) 来启用。
    """
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1):
        super().__init__()
        self.depth = depth
        self.upscale = upscale  # not used
        self.nf = nf

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc * 2, nf, 7)

        # -----------------------
        # time embedding
        # -----------------------
        time_dim = nf * 4
        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # -----------------------
        # UNet 主干
        # -----------------------
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # 记录每个 down level 的输出通道（用于 brush residual 对齐）
        self._down_level_out_channels = []

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i + 1))
            self._down_level_out_channels.append(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth - 1) else default_conv(dim_in, dim_out),
                SPADEBlock(dim_out, dim_out, 1)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_dim = mid_dim
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

        # -----------------------
        # PixelBrushNet 分支（默认不创建！）
        # 说明：不创建才能保证 use_pixel_brushnet=false 时不增加参数，老 ckpt strict_load 也能加载。
        # -----------------------
        self.use_pixel_brushnet = False
        self.pixel_brushnet = None
        self.pixel_brush_cfg = None
        self._brush_up_projs = None
        self._brush_mid_proj = None

        # up path 里每个 level 注入时的通道（dim_out 的倒序）
        self._up_level_channels = [nf * int(math.pow(2, i + 1)) for i in range(depth)][::-1]

    # -----------------------
    # 由 networks.py 调用（从 yml 读取开关&超参）
    # -----------------------
    def enable_pixel_brushnet(self, cfg: PixelBrushNetConfig):
        """
        从 yml 读取到 cfg 后调用该函数启用 PixelBrushNet。
        cfg 建议包含：
          - in_ch=4
          - base_ch=nf（最好跟主干 nf 一致）
          - ch_mult 与 depth 对齐（比如 depth=4 -> [1,2,4,8]）
          - conditioning_scale 控制注入强度
        """
        self.use_pixel_brushnet = True
        self.pixel_brush_cfg = cfg

        # 1) 创建 PixelBrushNet：输出 residual 的通道对齐到 downsample 后的 dim_out
        self.pixel_brushnet = PixelBrushNet(cfg, out_ch_per_level=self._down_level_out_channels)

        # -----------------------
        # 2) mid residual 的投影（关键修复）
        # PixelBrushNet 的 mid 输出通道 = base_ch * ch_mult[-1]
        # UNet 的 mid_dim = nf * 2^depth
        # 所以这里应该是：mid_ch -> mid_dim
        # -----------------------
        brush_mid_ch = int(cfg.base_ch * cfg.ch_mult[-1])   # 例如 64*8=512
        self._brush_mid_proj = nn.Conv2d(brush_mid_ch, self.mid_dim, 1, 1, 0)

        # zero init：保证刚接入时不破坏原模型（初始接近 no-op）
        nn.init.zeros_(self._brush_mid_proj.weight)
        if self._brush_mid_proj.bias is not None:
            nn.init.zeros_(self._brush_mid_proj.bias)

        # 3) up residual 的投影（同样 zero init）
        self._brush_up_projs = nn.ModuleList([
            nn.Conv2d(ch, ch, 1, 1, 0) for ch in self._up_level_channels
        ])
        for conv in self._brush_up_projs:
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def _get_brush_residuals(self, color_prior, confidence):
        """
        计算 BrushNet 分支输出的多尺度 residual。
        返回：
          - down_residuals: list[L]，按 down level 顺序对齐（0..depth-1），每个通道=dim_out
          - mid_residual : (B, mid_dim, h, w)
          - up_residuals : list[L]，按 up level 顺序对齐（0..depth-1），每个通道=dim_out
        """
        if (not self.use_pixel_brushnet) or (self.pixel_brushnet is None):
            return None, None, None
        if (color_prior is None) or (confidence is None):
            return None, None, None

        down_residuals, mid_residual = self.pixel_brushnet(
            color_prior=color_prior,
            confidence=confidence,
            conditioning_scale=getattr(self.pixel_brush_cfg, "conditioning_scale", 1.0)
        )

        assert mid_residual.shape[1] == self._brush_mid_proj.in_channels, \
            f"Brush mid_residual 通道={mid_residual.shape[1]}，但 _brush_mid_proj 期望输入通道={self._brush_mid_proj.in_channels}，请检查 cfg.base_ch/ch_mult 或 mid_proj 构造"
        # mid residual 投影（zero init，初始不影响主干）
        if self._brush_mid_proj is not None:
            mid_residual = self._brush_mid_proj(mid_residual)

        # 生成 up_residuals：将 down_residuals 反向对齐到 up levels，并用 zero 1x1 做投影
        up_residuals = []
        if self._brush_up_projs is not None:
            for i, _ch in enumerate(self._up_level_channels):
                src = down_residuals[self.depth - 1 - i]  # 对齐到对应尺度
                up_residuals.append(self._brush_up_projs[i](src))

        return down_residuals, mid_residual, up_residuals

    def forward(
        self,
        xt,
        cond,
        time=-1,
        S=None,
        color_prior=None,   # ✅ 颜色先验图（B,3,H,W）
        confidence=None,    # ✅ 置信度图（B,1,H,W）
        **kwargs
    ):
        # time 转 tensor
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        # StrDiffusion 纹理输入：拼 [xt-cond, cond]
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()
        t = self.time_mlp(time)

        if self.use_pixel_brushnet:
            B, _, H, W = cond.shape

            # ① 无颜色先验（no-prior 消融）：默认用全 0 颜色先验
            #    这样就“没有颜色信息”，但分支仍然参与计算，不会断图
            if color_prior is None:
                color_prior = torch.zeros_like(cond)  # (B,3,H,W)

            # ② 无置信度：默认全 1（分支作用于全图）
            #    如果你希望只在洞区域起作用，训练时传 confidence=mask 即可
            if confidence is None:
                confidence = torch.ones((B, 1, H, W), device=cond.device, dtype=cond.dtype)

        # ✅ 计算 brush residual（只算一次，节省时间）
        down_residuals, mid_residual, up_residuals = self._get_brush_residuals(color_prior, confidence)

        h = []

        # -----------------------
        # down path
        # 注入点：downsample 后、guide(x,S) 前
        # -----------------------
        for level_idx, (b1, b2, attn, downsample, guide) in enumerate(self.downs):
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

            # ✅ 注入 brush residual（颜色/置信度引导）
            if down_residuals is not None and level_idx < len(down_residuals):
                res = down_residuals[level_idx]
                if res.shape[-2:] != x.shape[-2:]:
                    res = F.interpolate(res, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x + res

            # ✅ 原本的结构引导（保持不变）
            x = guide(x, S)

        # -----------------------
        # mid
        # 注入点：mid_block2 后
        # -----------------------
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        if mid_residual is not None:
            if mid_residual.shape[-2:] != x.shape[-2:]:
                mid_residual = F.interpolate(mid_residual, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = x + mid_residual

        # -----------------------
        # up path
        # 注入点：attn 后、upsample 前
        # -----------------------
        for up_idx, (b1, b2, attn, upsample) in enumerate(self.ups):
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)

            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)

            x = attn(x)

            # ✅ 注入 up residual（和 up level 尺度对齐）
            if up_residuals is not None and up_idx < len(up_residuals):
                res = up_residuals[up_idx]
                if res.shape[-2:] != x.shape[-2:]:
                    res = F.interpolate(res, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x + res

            x = upsample(x)

        # head
        x = torch.cat([x, x_], dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W]

        # ✅ 你这份是训练侧接口：返回 (x, x) 以兼容 noise_fn 的 unpack
        return x, x


class ConditionalUNets(nn.Module):
    # 你原来的第二个 UNet（不含结构 guide），保持不动
    def __init__(self, in_nc, out_nc, nf, depth=4, upscale=1):
        super().__init__()
        self.depth = depth
        self.upscale = upscale # not used

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)

        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for i in range(depth):
            dim_in = nf * int(math.pow(2, i))
            dim_out = nf * int(math.pow(2, i+1))
            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if i != (depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * int(math.pow(2, depth))
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time)

        h = []

        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)

            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W]
        return x
