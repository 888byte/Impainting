# -*- coding: utf-8 -*-
"""
debug_utils.py - 调试工具模块

功能说明：
---------
提供调试模式下的张量可视化和保存功能。
开启debug_mode后，自动保存训练过程中的关键张量到拼接图像文件。

修改记录：
---------
- 修复 masked_input 计算错误（应为已知区域显示）
- 修改输出为拼接图像（方便对比查看）
- 添加标签标注

Author: Auto-generated for BrushNet Integration
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, Optional, Union, List
from datetime import datetime


class DebugLogger:
    """
    调试日志记录器
    
    自动保存训练过程中的张量为可视化拼接图像
    
    Attributes:
        log_dir: 日志保存目录
        enabled: 是否启用
        save_freq: 保存频率
    """
    
    def __init__(
        self,
        log_dir: str = './debug_logs',
        enabled: bool = False,
        save_freq: int = 500
    ):
        """
        初始化调试记录器
        
        Args:
            log_dir: 日志目录
            enabled: 是否启用
            save_freq: 每N个step保存一次
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.save_freq = save_freq
        
        if self.enabled:
            os.makedirs(self.log_dir, exist_ok=True)
            # 创建带时间戳的子目录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_dir = os.path.join(self.log_dir, f'session_{timestamp}')
            os.makedirs(self.session_dir, exist_ok=True)
            print(f"[DebugLogger] 已启用，输出目录: {self.session_dir}")
        else:
            self.session_dir = None
    
    def should_save(self, step: int) -> bool:
        """判断当前step是否需要保存"""
        return self.enabled and (step % self.save_freq == 0)
    
    def _tensor_to_image(
        self,
        tensor: torch.Tensor,
        normalize: bool = True,
        is_mask: bool = False
    ) -> np.ndarray:
        """
        将张量转换为可保存的图像
        
        Args:
            tensor: [C, H, W] 或 [B, C, H, W] 张量
            normalize: 是否归一化到0-255
            is_mask: 是否是掩码图（二值图）
            
        Returns:
            [H, W, 3] uint8 图像 (BGR格式)
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个batch
        
        # 转到CPU并转为numpy
        img = tensor.detach().cpu().numpy()
        
        if img.shape[0] == 1:
            # 单通道 -> 灰度图
            img = img[0]
            if is_mask:
                # 掩码直接二值化显示
                img = (img > 0.5).astype(np.float32)
            if normalize:
                img_min, img_max = img.min(), img.max()
                if img_max - img_min > 1e-8:
                    img = (img - img_min) / (img_max - img_min)
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            # 转为3通道灰度图便于拼接
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        
        elif img.shape[0] == 3:
            # 三通道 -> RGB -> BGR
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            if normalize and (img.max() <= 1.0):
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        
        else:
            raise ValueError(f"不支持的通道数: {img.shape[0]}")
    
    def _add_label(self, img: np.ndarray, label: str) -> np.ndarray:
        """
        在图像左上角添加标签
        
        Args:
            img: BGR图像
            label: 标签文本
            
        Returns:
            带标签的图像
        """
        img = img.copy()
        # 添加半透明背景
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (0, 0), (label_size[0] + 10, label_size[1] + 10), (0, 0, 0), -1)
        cv2.putText(img, label, (5, label_size[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img
    
    def save_training_state_concatenated(
        self,
        step: int,
        input_image: torch.Tensor,
        gt: torch.Tensor,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        refined_gt: Optional[torch.Tensor] = None
    ):
        """
        保存完整的训练状态为拼接大图
        
        布局：
        [原图] [GT] [RefinedGT] [Color Prior] [Confidence] [Mask] [Masked Input]
        
        Args:
            step: 当前步数
            input_image: 输入原始图像 (含mask区域的原图)
            gt: Ground Truth (原始GT，可能带噪声)
            refined_gt: 精炼后的GT (ChromaRefiner 输出, 可选)
            color_prior: 颜色先验 (第一阶段生成)
            confidence: 置信度图 (mask区域应为均匀低值)
            mask: 掩码 (1=缺失, 0=已知)
            output: 模型输出 (可选)
        
        说明：
        - input_image: 退化图像（训练时这是含mask的输入）
        - masked_input: input_image 与 mask 结合，mask区域置黑
        """
        if not self.should_save(step):
            return
        
        # 计算 masked_input
        masked_input = input_image * (1 - mask)
        
        # 准备所有图像
        images = []
        labels = ['Input', 'GT']
        tensors = [input_image, gt]
        
        # 如果有精炼后的 GT，添加对比
        if refined_gt is not None:
            labels.append('RefinedGT')
            tensors.append(refined_gt)
            # Debug: 打印 refined_gt 的范围
            print(f"[DebugLogger] refined_gt: min={refined_gt.min().item():.4f}, max={refined_gt.max().item():.4f}, shape={refined_gt.shape}")
        
        labels.extend(['Prior', 'Confidence', 'Mask', 'MaskedInput'])
        tensors.extend([color_prior, confidence, mask, masked_input])
        
        # 转换并添加标签
        for tensor, label in zip(tensors, labels):
            try:
                is_mask = (label == 'Mask')
                # Confidence 不要归一化，直接显示实际值（0.3会显示为灰色）
                is_confidence = (label == 'Confidence')
                if is_confidence:
                    # 不归一化，直接 * 255 显示
                    img = self._tensor_to_image(tensor, normalize=False, is_mask=False)
                else:
                    img = self._tensor_to_image(tensor, normalize=True, is_mask=is_mask)
                img = self._add_label(img, label)
                images.append(img)
            except Exception as e:
                print(f"[DebugLogger] 转换 {label} 失败: {e}")
                # 创建错误占位图
                h, w = 256, 256
                if len(images) > 0:
                    h, w = images[0].shape[:2]
                error_img = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Error: {label}", (10, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                images.append(error_img)
        
        # 确保所有图像尺寸一致
        if images:
            target_h = max(img.shape[0] for img in images)
            target_w = max(img.shape[1] for img in images)
            
            resized_images = []
            for img in images:
                if img.shape[:2] != (target_h, target_w):
                    img = cv2.resize(img, (target_w, target_h))
                resized_images.append(img)
            
            # 横向拼接
            concatenated = np.hstack(resized_images)
            
            # 保存
            filename = f"step_{step:08d}_debug.png"
            filepath = os.path.join(self.session_dir, filename)
            cv2.imwrite(filepath, concatenated)
            print(f"[DebugLogger] Step {step}: 已保存拼接调试图像 -> {filepath}")
    
    def save_tensors(
        self,
        step: int,
        tensors: Dict[str, torch.Tensor],
        prefix: str = ''
    ):
        """
        保存多个张量为单独图像（保留旧接口）
        
        Args:
            step: 当前训练步数
            tensors: 张量字典 {名称: 张量}
            prefix: 文件名前缀
        """
        if not self.should_save(step):
            return
        
        for name, tensor in tensors.items():
            try:
                img = self._tensor_to_image(tensor)
                filename = f"step_{step:08d}_{prefix}{name}.png"
                filepath = os.path.join(self.session_dir, filename)
                cv2.imwrite(filepath, img)
            except Exception as e:
                print(f"[DebugLogger] 保存 {name} 失败: {e}")
    
    # 兼容旧接口
    def save_training_state(
        self,
        step: int,
        input_image: torch.Tensor,
        gt: torch.Tensor,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        refined_gt: Optional[torch.Tensor] = None
    ):
        """
        保存完整的训练状态（调用拼接版本）
        
        Args:
            refined_gt: ChromaRefiner 精炼后的 GT (可选)
        """
        self.save_training_state_concatenated(
            step, input_image, gt, color_prior, confidence, mask, output, refined_gt
        )


def save_debug_concatenated(
    input_image: torch.Tensor,
    transformed_gt: torch.Tensor,
    generated_prior: torch.Tensor,
    confidence_map: torch.Tensor,
    mask: torch.Tensor,
    step: int,
    log_dir: str = './debug_logs'
):
    """
    独立函数：保存拼接调试图像
    
    可直接调用，无需DebugLogger实例
    
    Args:
        input_image: 输入图像
        transformed_gt: 变换后的GT
        generated_prior: 生成的颜色先验
        confidence_map: 置信度图
        mask: 掩码
        step: 当前步数
        log_dir: 保存目录
    """
    os.makedirs(log_dir, exist_ok=True)
    
    def tensor_to_img(t, is_mask=False):
        if t.dim() == 4:
            t = t[0]
        img = t.detach().cpu().numpy()
        if img.shape[0] in [1, 3]:
            if img.shape[0] == 1:
                img = img[0]
                if is_mask:
                    img = (img > 0.5).astype(np.float32)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = np.transpose(img, (1, 2, 0))
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def add_label(img, label):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (100, 20), (0, 0, 0), -1)
        cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return img
    
    # 计算 masked_input
    masked_input = input_image * (1 - mask)
    
    items = [
        ('Input', input_image, False),
        ('GT', transformed_gt, False),
        ('Prior', generated_prior, False),
        ('Confidence', confidence_map, False),
        ('Mask', mask, True),
        ('MaskedInput', masked_input, False)
    ]
    
    images = []
    for name, tensor, is_mask in items:
        try:
            img = tensor_to_img(tensor, is_mask)
            img = add_label(img, name)
            images.append(img)
        except Exception as e:
            print(f"[Debug] 转换 {name} 失败: {e}")
    
    if images:
        # 调整尺寸
        target_size = images[0].shape[:2]
        for i, img in enumerate(images):
            if img.shape[:2] != target_size:
                images[i] = cv2.resize(img, (target_size[1], target_size[0]))
        
        # 拼接
        concatenated = np.hstack(images)
        cv2.imwrite(os.path.join(log_dir, f"step_{step:08d}_debug.png"), concatenated)


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DebugLogger 单元测试")
    print("=" * 60)
    
    # 创建测试目录
    test_dir = './test_debug_logs'
    
    # 测试DebugLogger
    logger = DebugLogger(log_dir=test_dir, enabled=True, save_freq=1)
    
    # 创建测试张量
    input_img = torch.rand(1, 3, 256, 256)
    gt = torch.rand(1, 3, 256, 256)
    prior = torch.rand(1, 3, 256, 256)
    confidence = torch.rand(1, 1, 256, 256)
    mask = torch.zeros(1, 1, 256, 256)
    mask[:, :, 100:150, 100:150] = 1  # 中间区域是mask
    
    # 保存拼接图
    logger.save_training_state(
        step=1,
        input_image=input_img,
        gt=gt,
        color_prior=prior,
        confidence=confidence,
        mask=mask
    )
    
    print(f"\n已保存测试图像到: {logger.session_dir}")
    print("✓ 测试通过!")
