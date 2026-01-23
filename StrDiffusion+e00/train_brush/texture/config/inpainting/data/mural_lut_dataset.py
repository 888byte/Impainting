import os, glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.color_prior_generator import ColorPriorGenerator, Stage1Config

class MuralLUTDataset(Dataset):
    """
    动态 GT 生成：
    ModeA: 全图 LUT 映射作为 GT
    ModeB: 仅 mask 区域 LUT 映射，mask 外保持原图
    同时生成 Stage1 的 color_prior/conf_map，保证同源 LUT 逻辑
    """
    def __init__(self, img_root, mask_root, lut_path, gt_mode="A", stage1_cfg=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_root, "*.*")))
        self.mask_root = mask_root
        self.gt_mode = gt_mode.upper()
        cfg = Stage1Config(**(stage1_cfg or {}))
        self.stage1 = ColorPriorGenerator(lut_path, cfg=cfg)

    def __len__(self):
        return len(self.img_paths)

    def _load_mask(self, img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mpath = os.path.join(self.mask_root, name + ".png")
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(mpath)
        m = (m > 127).astype(np.float32)  # 1=mask? 这里你按你的约定修正
        # 建议统一为：mask_keep=1 表示已知区域
        mask_keep = 1.0 - m
        return mask_keep

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mask_keep = self._load_mask(img_path)  # (H,W) 1=known

        # GT 构建：同源 LUT（三线性）
        full_lut = self.stage1.apply_lut_full_np(rgb)  # (H,W,3) uint8
        if self.gt_mode == "A":
            gt = full_lut
        else:
            # ModeB：仅 mask 区域 LUT 映射
            gt = rgb.copy()
            hole = (1.0 - mask_keep)[..., None]
            gt = (gt.astype(np.float32) * (1-hole) + full_lut.astype(np.float32) * hole).round().astype(np.uint8)

        # Stage1 先验（按你第一阶段定义：known 用 LUT，hole 用 inpaint）
        prior_u8, conf = self.stage1.generate_np(rgb, mask_keep)

        # 输入给扩散模型：masked_input
        masked = (rgb.astype(np.float32) * mask_keep[..., None]).round().astype(np.uint8)

        # to torch
        def to_t(x):
            return torch.from_numpy(x.astype(np.float32)/255.).permute(2,0,1)
        rgb_t = to_t(rgb)
        gt_t = to_t(gt)
        prior_t = to_t(prior_u8)
        conf_t = torch.from_numpy(conf.astype(np.float32))[None, ...]
        mask_t = torch.from_numpy(mask_keep.astype(np.float32))[None, ...]

        return {
            "input": rgb_t,          # 原图（faded）
            "GT": gt_t,              # 动态GT（ModeA/B）
            "mask": mask_t,          # (1,H,W) keep=1
            "color_prior": prior_t,  # (3,H,W)
            "conf_map": conf_t,      # (1,H,W)
            "masked_input": to_t(masked)
        }
