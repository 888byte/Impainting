# -*- coding: utf-8 -*-
"""
mural_inpainting_dataset.py - 壁画修复数据集

功能说明：
---------
本数据集专为双阶段壁画修复训练设计，支持三种GT生成模式：
1. Mode A (full): 全图复原 - 整张图经过LUT映射，模型学习将褪色图恢复到原始颜色
2. Mode B (partial): 局部复原 - 仅Mask区域经过LUT映射，保留背景纹理
3. Mode C (mixed): 混合模式 - 训练时随机选择Mode A或B

任务定义：
--------
- `degraded_full`：完整褪色图，只用于生成训练目标 GT
- `degraded`：按 mask 合成后的真实缺损输入，用于模拟推理时的观测图
- `color_prior/confidence`：基于真实缺损输入生成

这样训练时的条件链与推理保持一致，而监督目标仍然来自完整参考图。

输出格式：
----------
{
    'degraded': [3, H, W],       # 当前训练输入（已按 mask 合成真实缺损外观）
    'degraded_full': [3, H, W],  # 完整褪色图（仅用于生成训练目标）
    'gt': [3, H, W],             # 目标GT（根据模式生成）
    'mask': [1, H, W],           # 修复区域掩码 (1=需要修复)
    'color_prior': [3, H, W],    # 颜色先验图（基于真实缺损输入生成）
    'confidence': [1, H, W],     # 置信度图
    'mode': str,                 # 当前样本使用的模式
    'path': str                  # 图像路径
}

Author: Auto-generated for BrushNet Integration
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

# 导入颜色先验生成器
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from color_prior_generator import ColorPriorGenerator
except ImportError:
    from ..color_prior_generator import ColorPriorGenerator


class MuralInpaintingDataset(Dataset):
    """
    壁画修复数据集
    
    支持三种GT生成模式：
    - 'full': 全图LUT映射
    - 'partial': 仅Mask区域LUT映射
    - 'mixed': 随机选择 (各50%概率)
    
    Attributes:
        opt: 配置字典
        color_prior_gen: 颜色先验生成器
        gt_mode: GT生成模式
        image_paths: 图像文件路径列表
        mask_paths: 掩码文件路径列表
    """
    
    def __init__(
        self,
        opt: dict,
        lut_path: str,
        gt_mode: str = 'mixed',
        prior_method: str = 'fast',
        debug_mode: bool = False,
        debug_dir: str = './debug_logs',
        lut_alpha: float = 0.7,
        lut_beta: float = 0.3,
        lut_inpaint_method: str = 'telea'
    ):
        """
        初始化数据集
        
        Args:
            opt: 配置字典，包含以下键：
                - dataroot_GT: GT图像目录
                - dataroot_mask: 掩码目录 (可选，如果不提供则动态生成)
                - GT_size: 图像尺寸
                - use_flip: 是否水平翻转
                - use_rot: 是否旋转
            lut_path: LUT文件路径
            gt_mode: GT生成模式 ('full', 'partial', 'mixed')
            debug_mode: 是否开启调试模式
            debug_dir: 调试输出目录
            lut_alpha: LUT置信度权重
            lut_beta: 修复置信度权重
            lut_inpaint_method: 修复方法 ('telea' 或 'ns')
            prior_method: 先验生成方法 ('fast' 或 'quality')
        """
        super().__init__()
        
        self.opt = opt
        self.gt_mode = gt_mode.lower()
        self.prior_method = prior_method.lower()
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        
        # 验证gt_mode
        valid_modes = ['full', 'partial', 'mixed']
        if self.gt_mode not in valid_modes:
            raise ValueError(f"gt_mode必须是 {valid_modes} 之一，实际为 {self.gt_mode}")
        
        # 初始化颜色先验生成器（使用直接传入的参数）
        self.color_prior_gen = ColorPriorGenerator(
            lut_path=lut_path,
            alpha=lut_alpha,
            beta=lut_beta,
            inpaint_method=lut_inpaint_method
        )
        
        # 获取图像尺寸
        self.GT_size = opt.get('GT_size', 256)
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.observed_hole_fill = str(opt.get('observed_hole_fill', 'white')).lower()
        self.observed_hole_fill_value = int(opt.get('observed_hole_fill_value', 255))
        self.max_crop_retry = int(opt.get('max_crop_retry', 20))
        self.min_hole_ratio = float(opt.get('min_hole_ratio', 0.01))
        self.max_hole_ratio = float(opt.get('max_hole_ratio', 0.75))
        self._crop_retry_fail_count = 0
        
        # 加载图像路径
        self.image_paths = self._get_image_paths(opt.get('dataroot_GT', ''))
        
        # 加载或准备掩码
        mask_root = opt.get('dataroot_mask', None)
        if mask_root and os.path.exists(mask_root):
            self.mask_paths = self._get_image_paths(mask_root)
            self.use_dynamic_mask = False
        else:
            self.mask_paths = []
            self.use_dynamic_mask = True
            print("[MuralInpaintingDataset] 未找到掩码目录，将动态生成掩码")
        
        # 创建调试目录
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        print(f"[MuralInpaintingDataset] 初始化完成:")
        print(f"  - 图像数量: {len(self.image_paths)}")
        print(f"  - 掩码数量: {len(self.mask_paths)}")
        print(f"  - GT模式: {self.gt_mode}")
        print(f"  - 图像尺寸: {self.GT_size}")
        print(f"  - Debug模式: {self.debug_mode}")

    def _build_observed_input(self, degraded_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        构造训练时的“真实缺损输入”。

        训练目标仍然来自完整褪色图，但提供给 prior/LUT/Mu-Denoiser 的输入
        需要尽量模拟推理时已经真实缺损的观测图像。

        Args:
            degraded_img: [H, W, 3] uint8 完整褪色图
            mask: [H, W] uint8 掩码，255=缺失区域

        Returns:
            observed: [H, W, 3] uint8 合成后的缺损输入
        """
        observed = degraded_img.copy()
        hole = mask > 127

        if self.observed_hole_fill == 'white':
            observed[hole] = self.observed_hole_fill_value
        elif self.observed_hole_fill == 'black':
            observed[hole] = 0
        else:
            raise ValueError(
                f"不支持的 observed_hole_fill: {self.observed_hole_fill}，"
                "请使用 'white' 或 'black'"
            )

        return observed
    
    def _get_image_paths(self, root: str) -> List[str]:
        """获取目录下所有图像文件路径"""
        if not os.path.exists(root):
            return []
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        paths = []
        
        for root_dir, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    paths.append(os.path.join(root_dir, f))
        
        return sorted(paths)
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载并预处理图像"""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法加载图像: {path}")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def _load_mask(self, path: str) -> np.ndarray:
        """加载掩码"""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {path}")
        
        return mask
    
    def _generate_dynamic_mask(self, h: int, w: int) -> np.ndarray:
        """
        动态生成修复掩码
        
        生成随机矩形/椭圆形掩码区域
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 随机选择掩码类型
        mask_type = random.choice(['rect', 'ellipse', 'irregular'])
        
        if mask_type == 'rect':
            # 随机矩形
            x1 = random.randint(int(w * 0.1), int(w * 0.5))
            y1 = random.randint(int(h * 0.1), int(h * 0.5))
            x2 = random.randint(x1 + int(w * 0.1), min(x1 + int(w * 0.5), w - 1))
            y2 = random.randint(y1 + int(h * 0.1), min(y1 + int(h * 0.5), h - 1))
            mask[y1:y2, x1:x2] = 255
            
        elif mask_type == 'ellipse':
            # 随机椭圆
            center_x = random.randint(int(w * 0.3), int(w * 0.7))
            center_y = random.randint(int(h * 0.3), int(h * 0.7))
            axis_x = random.randint(int(w * 0.1), int(w * 0.3))
            axis_y = random.randint(int(h * 0.1), int(h * 0.3))
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 
                       angle, 0, 360, 255, -1)
            
        else:
            # 不规则形状（多个随机线段）
            num_strokes = random.randint(3, 8)
            for _ in range(num_strokes):
                x1 = random.randint(0, w - 1)
                y1 = random.randint(0, h - 1)
                x2 = random.randint(0, w - 1)
                y2 = random.randint(0, h - 1)
                thickness = random.randint(int(min(w, h) * 0.03), int(min(w, h) * 0.08))
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
        return mask
    
    def _random_crop(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random crop with lightweight hole-ratio retry; output interface unchanged."""
        h, w = img.shape[:2]
        crop_size = self.GT_size
        
        if h < crop_size or w < crop_size:
            # Resize small images first.
            scale = max(crop_size / h, crop_size / w) * 1.1
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = new_h, new_w

        best_img_crop = None
        best_mask_crop = None
        best_ratio = None
        best_score = float("inf")
        max_retry = max(1, self.max_crop_retry)

        for _ in range(max_retry):
            y = random.randint(0, h - crop_size)
            x = random.randint(0, w - crop_size)
            img_crop = img[y:y + crop_size, x:x + crop_size]
            mask_crop = mask[y:y + crop_size, x:x + crop_size]
            hole_ratio = float(np.mean(mask_crop > 127))

            if hole_ratio < self.min_hole_ratio:
                score = self.min_hole_ratio - hole_ratio
            elif hole_ratio > self.max_hole_ratio:
                score = hole_ratio - self.max_hole_ratio
            else:
                return img_crop, mask_crop

            if score < best_score:
                best_score = score
                best_img_crop = img_crop
                best_mask_crop = mask_crop
                best_ratio = hole_ratio

        self._crop_retry_fail_count += 1
        if self.debug_mode or self._crop_retry_fail_count <= 5:
            print(
                "[MuralInpaintingDataset] crop retry fallback(best): "
                f"hole_ratio={best_ratio:.4f}, expected "
                f"[{self.min_hole_ratio:.4f}, {self.max_hole_ratio:.4f}], "
                f"max_crop_retry={max_retry}"
            )
        return best_img_crop, best_mask_crop

    def _feather_blend(
        self,
        original: np.ndarray,
        transformed: np.ndarray,
        mask: np.ndarray,
        feather_radius: int = 7
    ) -> np.ndarray:
        """
        使用羽化融合将 transformed 区域平滑混合到 original
        
        参考 t7.py 的实现，使用高斯模糊创建软边缘
        
        Args:
            original: [H, W, 3] 原始图像
            transformed: [H, W, 3] 变换后的图像
            mask: [H, W] uint8 掩码 (255=需要替换)
            feather_radius: 羽化半径（像素）
            
        Returns:
            blended: [H, W, 3] 融合后的图像
        """
        # 将 mask 转换为 0-1 浮点数
        mask_float = (mask.astype(np.float32) / 255.0)
        
        # 使用高斯模糊进行羽化
        if feather_radius > 0:
            k = 2 * feather_radius + 1
            mask_float = cv2.GaussianBlur(mask_float, (k, k), sigmaX=0, sigmaY=0)
        
        mask_float = np.clip(mask_float, 0.0, 1.0)
        
        # 扩展为3通道用于融合
        mask_3ch = mask_float[:, :, np.newaxis]
        
        # 软融合: result = original * (1 - mask) + transformed * mask
        original_f = original.astype(np.float32)
        transformed_f = transformed.astype(np.float32)
        
        blended = original_f * (1.0 - mask_3ch) + transformed_f * mask_3ch
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def _augment(
        self, 
        img: np.ndarray, 
        mask: np.ndarray,
        color_prior: np.ndarray,
        gt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据增强"""
        # 水平翻转
        if self.use_flip and random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            color_prior = np.fliplr(color_prior).copy()
            gt = np.fliplr(gt).copy()
        
        # 旋转 (0, 90, 180, 270度)
        if self.use_rot:
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
                color_prior = np.rot90(color_prior, k).copy()
                gt = np.rot90(gt, k).copy()
        
        return img, mask, color_prior, gt
    
    def _augment_with_confidence(
        self, 
        img: np.ndarray, 
        mask: np.ndarray,
        observed_img: np.ndarray,
        color_prior: np.ndarray,
        gt: np.ndarray,
        confidence: np.ndarray,
        conf_lut: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据增强（包含 observed_img、confidence 和 conf_lut）"""
        # 水平翻转
        if self.use_flip and random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            observed_img = np.fliplr(observed_img).copy()
            color_prior = np.fliplr(color_prior).copy()
            gt = np.fliplr(gt).copy()
            confidence = np.fliplr(confidence).copy()
            if conf_lut is not None:
                conf_lut = np.fliplr(conf_lut).copy()
        
        # 旋转 (0, 90, 180, 270度)
        if self.use_rot:
            k = random.randint(0, 3)
            if k > 0:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()
                observed_img = np.rot90(observed_img, k).copy()
                color_prior = np.rot90(color_prior, k).copy()
                gt = np.rot90(gt, k).copy()
                confidence = np.rot90(confidence, k).copy()
                if conf_lut is not None:
                    conf_lut = np.rot90(conf_lut, k).copy()
        
        return img, mask, observed_img, color_prior, gt, confidence, conf_lut
    
    def _generate_gt(
        self, 
        degraded_img: np.ndarray,
        mask: np.ndarray,
        mode: str
    ) -> np.ndarray:
        """
        根据模式生成GT
        
        Args:
            degraded_img: [H, W, 3] 褪色图像
            mask: [H, W] 掩码
            mode: 'full' 或 'partial'
            
        Returns:
            gt: [H, W, 3] 目标GT
        """
        return self.color_prior_gen.build_target(
            degraded_img,
            mask,
            mode=mode,
            feather_radius=7,
        )
    
    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """转换为PyTorch张量 [H, W, C] uint8 -> [C, H, W] float32 [0, 1]"""
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img.copy())
    
    def _save_debug(self, idx: int, data: Dict[str, np.ndarray]):
        """保存调试图像"""
        if not self.debug_mode:
            return
        
        for key, value in data.items():
            if value.ndim == 2:
                # 灰度图
                cv2.imwrite(
                    os.path.join(self.debug_dir, f"sample_{idx}_{key}.png"),
                    value
                )
            elif value.ndim == 3:
                # RGB图 -> BGR保存
                cv2.imwrite(
                    os.path.join(self.debug_dir, f"sample_{idx}_{key}.png"),
                    cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_RGB2BGR)
                )
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            dict 包含：
                - 'degraded': [3, H, W] 当前训练输入（真实缺损外观）
                - 'degraded_full': [3, H, W] 完整褪色图
                - 'gt': [3, H, W] 目标GT
                - 'mask': [1, H, W] 修复掩码
                - 'color_prior': [3, H, W] 颜色先验
                - 'confidence': [1, H, W] 置信度图
                - 'mode': str 当前GT模式
                - 'path': str 图像路径
        """
        # ============================================================
        # Step 1: 加载图像和掩码
        # ============================================================
        img_path = self.image_paths[index]
        degraded_img = self._load_image(img_path)  # [H, W, 3] RGB
        
        # 加载或生成掩码
        if self.use_dynamic_mask or len(self.mask_paths) == 0:
            h, w = degraded_img.shape[:2]
            mask = self._generate_dynamic_mask(h, w)
        else:
            mask_idx = index % len(self.mask_paths)
            mask = self._load_mask(self.mask_paths[mask_idx])
            # 调整掩码尺寸与图像一致
            if mask.shape[:2] != degraded_img.shape[:2]:
                mask = cv2.resize(mask, (degraded_img.shape[1], degraded_img.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        # ============================================================
        # Step 2: 随机裁剪
        # ============================================================
        degraded_img, mask = self._random_crop(degraded_img, mask)
        
        # ============================================================
        # Step 3: 构造真实缺损输入，并确定 GT 模式
        # ============================================================
        observed_degraded = self._build_observed_input(degraded_img, mask)

        if self.gt_mode == 'mixed':
            current_mode = 'full' if random.random() > 0.5 else 'partial'
        else:
            current_mode = self.gt_mode
        
        # ============================================================
        # Step 4: 生成颜色先验和置信度（遵循 gt_mode）
        # ============================================================
        # 获取LUT映射结果
        prior_result = self.color_prior_gen.generate(observed_degraded, mask, method=self.prior_method)
        lut_mapped = prior_result['color_prior']     # [H, W, 3] float32 - 全图LUT+修复
        confidence = prior_result['confidence']       # [H, W] float32
        conf_lut = prior_result['conf_lut']           # [H, W] float32 - LUT原始置信度
        
        # 根据 current_mode 决定 color_prior 的内容
        if current_mode == 'full':
            # Full 模式: Prior 全图使用LUT映射结果
            color_prior = lut_mapped
        else:
            # Partial 模式: Prior 只在mask区域使用LUT（使用羽化融合）
            lut_mapped_uint8 = np.clip(lut_mapped, 0, 255).astype(np.uint8)
            color_prior_blended = self._feather_blend(
                original=observed_degraded,
                transformed=lut_mapped_uint8,
                mask=mask,
                feather_radius=7
            )
            color_prior = color_prior_blended.astype(np.float32)
        
        # ============================================================
        # Step 5: 生成GT
        # ============================================================
        gt = self._generate_gt(degraded_img, mask, current_mode)
        
        # ============================================================
        # Step 6: 数据增强（同时增强 confidence 和 conf_lut）
        # ============================================================
        degraded_img, mask, observed_degraded, color_prior, gt, confidence, conf_lut = self._augment_with_confidence(
            degraded_img, mask, observed_degraded, color_prior, gt, confidence, conf_lut
        )
        
        # ============================================================
        # Step 7: 调试输出
        # ============================================================
        if self.debug_mode and index < 5:  # 只保存前5个样本
            self._save_debug(index, {
                'degraded': observed_degraded,
                'degraded_full': degraded_img,
                'gt': gt,
                'mask': mask,
                'color_prior': color_prior,
                'confidence': (confidence * 255).astype(np.uint8)
            })
        
        # ============================================================
        # Step 8: 转换为张量
        # ============================================================
        degraded_tensor = self._to_tensor(observed_degraded)      # [3, H, W]
        degraded_full_tensor = self._to_tensor(degraded_img)      # [3, H, W]
        gt_tensor = self._to_tensor(gt)                           # [3, H, W]
        color_prior_tensor = self._to_tensor(
            np.clip(color_prior, 0, 255).astype(np.uint8)
        )                                                          # [3, H, W]
        
        mask_tensor = torch.from_numpy(
            mask.astype(np.float32) / 255.0
        ).unsqueeze(0)                                             # [1, H, W]
        
        confidence_tensor = torch.from_numpy(
            confidence.astype(np.float32)
        ).unsqueeze(0)                                             # [1, H, W]
        
        # conf_lut 用于 ChromaRefiner
        conf_lut_tensor = torch.from_numpy(
            conf_lut.astype(np.float32)
        ).unsqueeze(0)                                             # [1, H, W]
        
        # 生成灰度图和边缘图（与原训练代码兼容）
        gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
        gt_gray_tensor = torch.from_numpy(
            gt_gray.astype(np.float32) / 255.0
        ).unsqueeze(0)                                             # [1, H, W]
        
        # 边缘检测 (Canny)
        gt_edge = cv2.Canny(gt, 50, 150)
        gt_edge_tensor = torch.from_numpy(
            gt_edge.astype(np.float32) / 255.0
        ).unsqueeze(0)                                             # [1, H, W]
        
        return {
            'degraded': degraded_tensor,
            'degraded_full': degraded_full_tensor,
            'GT': gt_tensor,  # 与原数据集兼容
            'gt': gt_tensor,
            'GT_gray': gt_gray_tensor,   # 灰度图（与原训练代码兼容）
            'GT_edge': gt_edge_tensor,   # 边缘图（与原训练代码兼容）
            'mask': mask_tensor,
            'color_prior': color_prior_tensor,
            'confidence': confidence_tensor,
            'conf_lut': conf_lut_tensor,  # ChromaRefiner 需要
            'mode': current_mode,
            'path': img_path,
            'GT_path': img_path  # 与原数据集兼容
        }
    
    def __len__(self) -> int:
        return len(self.image_paths)


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("MuralInpaintingDataset 单元测试")
    print("=" * 60)
    
    # 配置
    opt = {
        'dataroot_GT': './test_images',  # 替换为实际路径
        'GT_size': 256,
        'use_flip': True,
        'use_rot': True,
        'lut': {
            'alpha': 0.7,
            'beta': 0.3,
            'inpaint_method': 'telea'
        }
    }
    
    LUT_PATH = './example_lut.npz'
    
    # 检查LUT文件
    if not os.path.exists(LUT_PATH):
        print(f"\n[警告] LUT文件未找到: {LUT_PATH}")
        print("请先运行 lut_processor.py 创建测试LUT")
        sys.exit(1)
    
    # 创建测试图像目录
    test_dir = './test_images'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        # 创建测试图像
        for i in range(5):
            test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(test_dir, f'test_{i}.png'), test_img)
        print(f"已创建测试图像目录: {test_dir}")
    
    # 测试三种模式
    for mode in ['full', 'partial', 'mixed']:
        print(f"\n[测试] gt_mode='{mode}'...")
        
        try:
            dataset = MuralInpaintingDataset(
                opt=opt,
                lut_path=LUT_PATH,
                gt_mode=mode,
                debug_mode=True,
                debug_dir=f'./debug_logs_{mode}'
            )
            
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  degraded: {sample['degraded'].shape}")
                print(f"  gt: {sample['gt'].shape}")
                print(f"  mask: {sample['mask'].shape}")
                print(f"  color_prior: {sample['color_prior'].shape}")
                print(f"  confidence: {sample['confidence'].shape}")
                print(f"  mode: {sample['mode']}")
                print(f"  ✓ 测试通过")
            else:
                print(f"  数据集为空")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    print("\n测试完成!")
