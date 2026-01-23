# -*- coding: utf-8 -*-
"""
debug_utils.py - 调试工具模块

功能说明：
---------
提供调试模式下的张量可视化和保存功能。
开启debug_mode后，自动保存训练过程中的关键张量到图像文件。

Author: Auto-generated for BrushNet Integration
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, Optional, Union
from datetime import datetime


class DebugLogger:
    """
    调试日志记录器
    
    自动保存训练过程中的张量为可视化图像
    
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
        normalize: bool = True
    ) -> np.ndarray:
        """
        将张量转换为可保存的图像
        
        Args:
            tensor: [C, H, W] 或 [B, C, H, W] 张量
            normalize: 是否归一化到0-255
            
        Returns:
            [H, W, C] uint8 图像 (BGR格式)
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个batch
        
        # 转到CPU并转为numpy
        img = tensor.detach().cpu().numpy()
        
        if img.shape[0] == 1:
            # 单通道 -> 灰度图
            img = img[0]
            if normalize:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
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
    
    def save_tensors(
        self,
        step: int,
        tensors: Dict[str, torch.Tensor],
        prefix: str = ''
    ):
        """
        保存多个张量为图像
        
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
    
    def save_training_state(
        self,
        step: int,
        input_image: torch.Tensor,
        gt: torch.Tensor,
        color_prior: torch.Tensor,
        confidence: torch.Tensor,
        mask: torch.Tensor,
        output: Optional[torch.Tensor] = None
    ):
        """
        保存完整的训练状态
        
        Args:
            step: 当前步数
            input_image: 输入图像
            gt: Ground Truth
            color_prior: 颜色先验
            confidence: 置信度图
            mask: 掩码
            output: 模型输出 (可选)
        """
        if not self.should_save(step):
            return
        
        tensors = {
            'input_image': input_image,
            'transformed_gt': gt,
            'generated_prior': color_prior,
            'confidence_map': confidence,
            'masked_input': input_image * (1 - mask)
        }
        
        if output is not None:
            tensors['model_output'] = output
        
        self.save_tensors(step, tensors)
        print(f"[DebugLogger] Step {step}: 已保存调试图像到 {self.session_dir}")


def save_debug_tensors(
    input_image: torch.Tensor,
    transformed_gt: torch.Tensor,
    generated_prior: torch.Tensor,
    confidence_map: torch.Tensor,
    masked_input: torch.Tensor,
    step: int,
    log_dir: str = './debug_logs'
):
    """
    独立函数：保存调试张量
    
    可直接调用，无需DebugLogger实例
    
    Args:
        所有参数同 DebugLogger.save_training_state
    """
    os.makedirs(log_dir, exist_ok=True)
    
    def tensor_to_img(t):
        if t.dim() == 4:
            t = t[0]
        img = t.detach().cpu().numpy()
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    items = {
        'input_image': input_image,
        'transformed_gt': transformed_gt,
        'generated_prior': generated_prior,
        'confidence_map': confidence_map,
        'masked_input': masked_input
    }
    
    for name, tensor in items.items():
        try:
            img = tensor_to_img(tensor)
            cv2.imwrite(os.path.join(log_dir, f"step_{step}_{name}.png"), img)
        except Exception as e:
            print(f"[Debug] 保存 {name} 失败: {e}")


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
    test_tensors = {
        'rgb_image': torch.rand(1, 3, 256, 256),
        'grayscale': torch.rand(1, 1, 256, 256),
        'mask': torch.randint(0, 2, (1, 1, 256, 256)).float()
    }
    
    # 保存
    logger.save_tensors(step=1, tensors=test_tensors)
    
    print(f"\n已保存测试图像到: {logger.session_dir}")
    print("✓ 测试通过!")
