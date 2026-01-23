# -*- coding: utf-8 -*-
"""
color_prior_generator.py - 颜色先验与置信度生成器

功能说明：
---------
本模块整合LUT颜色映射和缺失区域填充，生成高质量的颜色先验图和置信度图。

核心功能：
----------
1. 对已知像素区域应用LUT三线性插值进行颜色映射
2. 对Mask缺失区域使用多尺度cv2.inpaint进行填充
3. 融合LUT置信度和修复置信度，生成最终置信度图

多尺度修复策略：
----------------
- 如果 Mask 区域占比 > 30%：先下采样图像进行修复，再上采样回原分辨率
- 这样可以保证大面积缺失时的全局颜色一致性

置信度融合公式：
---------------
Conf_final = α * Conf_LUT + β * Conf_Inpaint

其中：
- α (alpha): LUT置信度权重，默认0.7
- β (beta): 修复置信度权重，默认0.3
- Mask区域的 Conf_Inpaint 显著低于已知区域

Author: Auto-generated for BrushNet Integration
"""

import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# 导入LUT处理器
import sys
import os
# 添加当前目录到路径以支持相对导入
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from lut_processor import LUTProcessor


class ColorPriorGenerator:
    """
    颜色先验与置信度生成器
    
    该类负责：
    1. 加载并应用LUT进行颜色映射
    2. 对缺失区域进行多尺度修复填充
    3. 生成融合的置信度图
    
    Attributes:
        lut (LUTProcessor): LUT处理器实例
        alpha (float): LUT置信度权重
        beta (float): 修复置信度权重
        inpaint_method (int): cv2.inpaint 方法 (TELEA 或 NS)
        large_mask_threshold (float): 大面积缺失阈值（默认0.3）
        inpaint_conf_known (float): 已知区域的修复置信度
        inpaint_conf_inpainted (float): 修复区域的修复置信度
    """
    
    def __init__(
        self,
        lut_path: str,
        alpha: float = 0.7,
        beta: float = 0.3,
        inpaint_method: str = 'telea',
        large_mask_threshold: float = 0.3,
        inpaint_conf_known: float = 1.0,
        inpaint_conf_inpainted: float = 0.3
    ):
        """
        初始化颜色先验生成器
        
        Args:
            lut_path: pigment_lut33.npz 文件路径
            alpha: LUT置信度权重 (默认0.7)
            beta: 修复置信度权重 (默认0.3)
            inpaint_method: 修复方法 'telea' 或 'ns' (默认'telea')
            large_mask_threshold: 大面积缺失阈值 (默认0.3, 即30%)
            inpaint_conf_known: 已知区域的修复置信度 (默认1.0)
            inpaint_conf_inpainted: 修复区域的修复置信度 (默认0.3)
        """
        # 加载LUT处理器
        self.lut = LUTProcessor(lut_path)
        
        # 置信度融合参数
        self.alpha = alpha
        self.beta = beta
        
        # 多尺度修复参数
        self.large_mask_threshold = large_mask_threshold
        
        # 修复方法选择
        if inpaint_method.lower() == 'telea':
            self.inpaint_method = cv2.INPAINT_TELEA
        elif inpaint_method.lower() == 'ns':
            self.inpaint_method = cv2.INPAINT_NS
        else:
            raise ValueError(f"不支持的修复方法: {inpaint_method}，请使用 'telea' 或 'ns'")
        
        # 修复置信度参数
        self.inpaint_conf_known = inpaint_conf_known
        self.inpaint_conf_inpainted = inpaint_conf_inpainted
        
        print(f"[ColorPriorGenerator] 初始化完成:")
        print(f"  - α (LUT权重): {self.alpha}")
        print(f"  - β (修复权重): {self.beta}")
        print(f"  - 修复方法: {inpaint_method}")
        print(f"  - 大面积阈值: {self.large_mask_threshold * 100:.0f}%")
    
    def _calculate_mask_ratio(self, mask: np.ndarray) -> float:
        """
        计算Mask区域占比
        
        Args:
            mask: [H, W] uint8 掩码（255=缺失, 0=已知）
            
        Returns:
            ratio: 缺失区域占比 (0.0 ~ 1.0)
        """
        total_pixels = mask.size
        masked_pixels = np.sum(mask > 127)
        return masked_pixels / total_pixels
    
    def multi_scale_inpaint(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        inpaint_radius: int = 3
    ) -> np.ndarray:
        """
        多尺度修复策略
        
        策略说明：
        ---------
        1. 计算Mask区域占比
        2. 如果 > large_mask_threshold：
           - 将图像下采样到1/2尺寸
           - 在低分辨率上进行修复
           - 上采样回原分辨率
           - 再次进行精细修复
        3. 否则直接使用cv2.inpaint
        
        Args:
            image: [H, W, 3] uint8 输入图像 (BGR或RGB)
            mask: [H, W] uint8 掩码 (255=缺失, 0=已知)
            inpaint_radius: 修复半径 (默认3)
            
        Returns:
            inpainted: [H, W, 3] uint8 修复后的图像
        """
        H, W = image.shape[:2]
        mask_ratio = self._calculate_mask_ratio(mask)
        
        # 确保mask是uint8类型
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        if mask_ratio > self.large_mask_threshold:
            # ============================================================
            # 大面积缺失：使用多尺度策略
            # ============================================================
            
            # 第一阶段：在1/2分辨率上修复（获取全局颜色一致性）
            scale_factor = 0.5
            small_H, small_W = int(H * scale_factor), int(W * scale_factor)
            
            # 下采样
            small_image = cv2.resize(image, (small_W, small_H), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, (small_W, small_H), interpolation=cv2.INTER_NEAREST)
            
            # 低分辨率修复
            small_inpainted = cv2.inpaint(
                small_image, small_mask, 
                inpaintRadius=inpaint_radius * 2,  # 低分辨率用更大半径
                flags=self.inpaint_method
            )
            
            # 上采样回原分辨率
            upscaled = cv2.resize(small_inpainted, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # 将上采样结果填充到原图的mask区域
            merged = image.copy()
            mask_bool = mask > 127
            merged[mask_bool] = upscaled[mask_bool]
            
            # 第二阶段：在原分辨率上精细修复边缘
            # 创建边缘mask（只修复边缘过渡区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            edge_mask = dilated - eroded
            
            # 精细修复边缘
            final_inpainted = cv2.inpaint(
                merged, edge_mask,
                inpaintRadius=inpaint_radius,
                flags=self.inpaint_method
            )
            
            return final_inpainted
        
        else:
            # ============================================================
            # 小面积缺失：直接修复
            # ============================================================
            return cv2.inpaint(
                image, mask,
                inpaintRadius=inpaint_radius,
                flags=self.inpaint_method
            )
    
    def generate(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        debug: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        生成颜色先验和置信度图
        
        处理流程：
        ---------
        1. 对整张图像应用LUT映射，获得初始颜色先验和LUT置信度
        2. 对mask区域使用多尺度修复填充颜色先验
        3. 构建修复置信度图（mask区域置信度较低）
        4. 融合两种置信度：Conf_final = α * Conf_LUT + β * Conf_Inpaint
        
        Args:
            image: [H, W, 3] uint8 输入RGB图像 (0-255)
            mask: [H, W] uint8 掩码 (255=缺失需要修复, 0=已知)
            debug: 是否返回调试信息
            
        Returns:
            result: dict 包含以下键：
                - 'color_prior': [H, W, 3] float32 颜色先验图 (0-255)
                - 'confidence': [H, W] float32 融合置信度图 (0-1)
                - 'conf_lut': [H, W] float32 LUT原始置信度 (0-1)
                - 'conf_inpaint': [H, W] float32 修复区域置信度 (0-1)
                
        Tensor Shape 变化：
        ------------------
        输入图像: [H, W, 3] uint8
        LUT映射后: [H, W, 3] float32
        修复填充后: [H, W, 3] float32
        最终颜色先验: [H, W, 3] float32
        最终置信度: [H, W] float32
        """
        # 验证输入
        assert image.ndim == 3 and image.shape[2] == 3, \
            f"输入图像形状应为 [H, W, 3]，实际为 {image.shape}"
        assert mask.ndim == 2, \
            f"掩码形状应为 [H, W]，实际为 {mask.shape}"
        assert image.shape[:2] == mask.shape, \
            f"图像和掩码尺寸不匹配: {image.shape[:2]} vs {mask.shape}"
        
        H, W = image.shape[:2]
        
        # ============================================================
        # Step 1: 对整张图像应用LUT映射
        # ============================================================
        color_prior_lut, conf_lut = self.lut.trilinear_interpolate(image)
        # color_prior_lut: [H, W, 3] float32, 范围 0-255
        # conf_lut: [H, W] float32, 范围 0-1
        
        # ============================================================
        # Step 2: 对mask区域进行多尺度修复填充
        # ============================================================
        # 将颜色先验转为uint8用于修复
        color_prior_uint8 = np.clip(color_prior_lut, 0, 255).astype(np.uint8)
        
        # 多尺度修复
        color_prior_inpainted = self.multi_scale_inpaint(color_prior_uint8, mask)
        color_prior_inpainted = color_prior_inpainted.astype(np.float32)
        
        # ============================================================
        # Step 3: 构建修复置信度图
        # ============================================================
        # 已知区域置信度高，修复区域置信度低
        mask_normalized = mask.astype(np.float32) / 255.0  # [H, W], 0=已知, 1=缺失
        
        conf_inpaint = (
            (1 - mask_normalized) * self.inpaint_conf_known +  # 已知区域置信度
            mask_normalized * self.inpaint_conf_inpainted       # 修复区域置信度
        )  # [H, W] float32, 范围 0-1
        
        # ============================================================
        # Step 4: 融合置信度
        # ============================================================
        # 对于已知区域，使用LUT置信度
        # 对于修复区域，进一步降低置信度
        confidence = self.alpha * conf_lut + self.beta * conf_inpaint
        confidence = np.clip(confidence, 0, 1)
        
        # ============================================================
        # Step 5: 组装最终颜色先验
        # ============================================================
        # 已知区域使用LUT映射结果，修复区域使用inpaint结果
        color_prior = color_prior_lut.copy()
        mask_bool = mask > 127
        color_prior[mask_bool] = color_prior_inpainted[mask_bool]
        
        # 构建返回结果
        result = {
            'color_prior': color_prior,      # [H, W, 3] float32, 0-255
            'confidence': confidence,         # [H, W] float32, 0-1
            'conf_lut': conf_lut,             # [H, W] float32, 0-1
            'conf_inpaint': conf_inpaint      # [H, W] float32, 0-1
        }
        
        if debug:
            result['color_prior_lut'] = color_prior_lut
            result['color_prior_inpainted'] = color_prior_inpainted
            result['mask_ratio'] = self._calculate_mask_ratio(mask)
        
        return result
    
    def generate_tensor(
        self,
        image_tensor,
        mask_tensor,
        device=None
    ):
        """
        对PyTorch张量生成颜色先验和置信度图
        
        Args:
            image_tensor: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            mask_tensor: [B, 1, H, W] torch.Tensor，范围 [0, 1]（1=缺失）
            device: 输出设备
            
        Returns:
            color_prior: [B, 3, H, W] torch.Tensor，范围 [0, 1]
            confidence: [B, 1, H, W] torch.Tensor，范围 [0, 1]
        """
        import torch
        
        if device is None:
            device = image_tensor.device
        
        B = image_tensor.shape[0]
        
        # 转换到numpy
        img_np = image_tensor.detach().cpu().numpy()  # [B, 3, H, W]
        img_np = np.transpose(img_np, (0, 2, 3, 1))    # [B, H, W, 3]
        img_np = (img_np * 255).astype(np.uint8)
        
        mask_np = mask_tensor.detach().cpu().numpy()   # [B, 1, H, W]
        mask_np = mask_np[:, 0, :, :]                  # [B, H, W]
        mask_np = (mask_np * 255).astype(np.uint8)
        
        # 批量处理
        color_priors = []
        confidences = []
        
        for i in range(B):
            result = self.generate(img_np[i], mask_np[i])
            color_priors.append(result['color_prior'])
            confidences.append(result['confidence'])
        
        # 转回tensor
        color_prior = np.stack(color_priors, axis=0)   # [B, H, W, 3]
        color_prior = np.transpose(color_prior, (0, 3, 1, 2)) / 255.0  # [B, 3, H, W]
        
        confidence = np.stack(confidences, axis=0)     # [B, H, W]
        confidence = confidence[:, np.newaxis, :, :]   # [B, 1, H, W]
        
        color_prior = torch.from_numpy(color_prior.astype(np.float32)).to(device)
        confidence = torch.from_numpy(confidence.astype(np.float32)).to(device)
        
        return color_prior, confidence


# =============================================================================
# 单元测试
# =============================================================================
if __name__ == "__main__":
    import sys
    
    # 测试用占位LUT路径
    LUT_PATH = "./pigment_lut33.npz"
    
    print("=" * 60)
    print("Color Prior Generator 单元测试")
    print("=" * 60)
    
    # 确保LUT文件存在（复用lut_processor的测试文件）
    if not os.path.exists(LUT_PATH):
        print(f"\n[警告] LUT文件未找到: {LUT_PATH}")
        print("请先运行 lut_processor.py 创建测试LUT")
        sys.exit(1)
    
    # 测试初始化
    print("\n[测试1] 初始化生成器...")
    gen = ColorPriorGenerator(
        LUT_PATH,
        alpha=0.7,
        beta=0.3,
        inpaint_method='telea'
    )
    
    # 创建测试数据
    print("\n[测试2] 创建测试数据...")
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # 小面积mask
    small_mask = np.zeros((256, 256), dtype=np.uint8)
    small_mask[100:150, 100:150] = 255  # 约10%缺失
    
    # 大面积mask
    large_mask = np.zeros((256, 256), dtype=np.uint8)
    large_mask[50:200, 50:200] = 255  # 约35%缺失
    
    # 测试小面积修复
    print("\n[测试3] 小面积修复...")
    result_small = gen.generate(test_img, small_mask, debug=True)
    print(f"  Mask占比: {result_small['mask_ratio']*100:.1f}%")
    print(f"  颜色先验形状: {result_small['color_prior'].shape}")
    print(f"  置信度形状: {result_small['confidence'].shape}")
    print(f"  置信度范围: [{result_small['confidence'].min():.4f}, {result_small['confidence'].max():.4f}]")
    
    # 测试大面积修复（多尺度）
    print("\n[测试4] 大面积修复（多尺度策略）...")
    result_large = gen.generate(test_img, large_mask, debug=True)
    print(f"  Mask占比: {result_large['mask_ratio']*100:.1f}%")
    print(f"  颜色先验形状: {result_large['color_prior'].shape}")
    print(f"  置信度形状: {result_large['confidence'].shape}")
    
    # 验证置信度在mask区域较低
    mask_bool = large_mask > 127
    conf_in_mask = result_large['confidence'][mask_bool].mean()
    conf_out_mask = result_large['confidence'][~mask_bool].mean()
    print(f"  Mask内平均置信度: {conf_in_mask:.4f}")
    print(f"  Mask外平均置信度: {conf_out_mask:.4f}")
    
    assert conf_in_mask < conf_out_mask, "修复区域置信度应低于已知区域"
    
    print("\n✓ 所有测试通过!")
