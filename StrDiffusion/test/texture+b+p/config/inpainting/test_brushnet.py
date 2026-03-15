# -*- coding: utf-8 -*-
"""
test_brushnet.py - 壁画颜色修复推理脚本（BrushNet + Mu-Denoiser 版）

运行命令：
python test_brushnet.py -opt options/test/ir-sde-brushnet.yml

功能：
- 加载训练好的 BrushNet + Mu-Denoiser 模型
- 对输入图像进行去噪 + 置信度加权 LUT 变换
- 生成颜色先验（传统填充 mask 区域 + 新 LUT 对齐非 mask 区域）
- 使用 SDE 反向采样进行修复
- 保存结果
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os.path as osp

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import options as option
from models import create_model
import utils as util
from data import create_dataloader, create_dataset
import str_utils as str_util

# 导入颜色先验生成器
try:
    from color_prior_generator import ColorPriorGenerator
except ImportError:
    ColorPriorGenerator = None
    print("[WARNING] ColorPriorGenerator not found, color prior will be disabled")

# 导入LUT处理器（用于置信度加权LUT变换）
try:
    from lut_processor import LUTProcessor
except ImportError:
    LUTProcessor = None
    print("[WARNING] LUTProcessor not found, confidence-weighted LUT will be disabled")


def denoise_image(img_np: np.ndarray, d=9, sigma_color=75, sigma_space=75) -> np.ndarray:
    """
    对图像进行边缘保持去噪（双边滤波）
    
    Args:
        img_np: [H, W, 3] uint8 输入图像
        d: 滤波直径
        sigma_color: 颜色空间sigma
        sigma_space: 坐标空间sigma
    
    Returns:
        denoised: [H, W, 3] uint8 去噪后图像
    """
    return cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)


def apply_lut_with_confidence(lut_processor, img_np: np.ndarray, lut_strength: float = 0.7,
                               smooth_radius: int = 5) -> tuple:
    """
    应用 LUT 变换并使用置信度加权混合
    （与训练代码 denoising_model.py 的逻辑对齐）
    
    Args:
        lut_processor: LUTProcessor 实例
        img_np: [H, W, 3] uint8 输入图像（去噪后）
        lut_strength: 全局 LUT 强度 (0-1)
        smooth_radius: 平滑半径
    
    Returns:
        lut_result: [H, W, 3] float32 LUT 变换结果 (0-255)
        confidence: [H, W] float32 置信度 (0-1)
    """
    # 应用 LUT 获取颜色映射和置信度
    lut_color, confidence = lut_processor.trilinear_interpolate(img_np)
    
    # 平滑处理（减少颜色割裂）
    if smooth_radius > 0:
        d = smooth_radius * 2 + 1
        lut_color_uint8 = np.clip(lut_color, 0, 255).astype(np.uint8)
        for c in range(3):
            lut_color[:, :, c] = cv2.bilateralFilter(
                lut_color_uint8[:, :, c], d=d, sigmaColor=50, sigmaSpace=smooth_radius
            ).astype(np.float32)
    
    # 置信度加权混合
    # 有效权重 = 全局强度 × 逐像素置信度
    effective_weight = lut_strength * confidence  # [H, W]
    effective_weight_3ch = effective_weight[:, :, np.newaxis]  # [H, W, 1]
    
    # 混合：低置信度保持原色，高置信度使用 LUT
    img_float = img_np.astype(np.float32)
    result = img_float * (1 - effective_weight_3ch) + lut_color * effective_weight_3ch
    
    return result, confidence


class MaskDataset(Dataset):
    """掩码数据集"""
    def __init__(self, mask_root):
        self.data = []
        for root, dirs, files in os.walk(mask_root, topdown=True):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.data.append(osp.join(root, name))
        print(f"[MaskDataset] Loaded {len(self.data)} masks from {mask_root}")
        
        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        mask = self.transform(Image.open(path).convert('1'))
        return 1 - mask  # 0=masked(需要修复), 1=unmasked(已知)


def _feather_blend(original: np.ndarray, transformed: np.ndarray, 
                   mask: np.ndarray, feather_radius: int = 7) -> np.ndarray:
    """
    羽化融合：在mask边界进行平滑过渡
    
    Args:
        original: [H, W, 3] uint8 原始图像
        transformed: [H, W, 3] uint8 变换后的图像（如LUT映射）
        mask: [H, W] uint8 掩码 (255=需要修复, 0=已知)
        feather_radius: 羽化半径
        
    Returns:
        blended: [H, W, 3] uint8 融合后的图像
    """
    # 创建羽化掩码
    mask_float = mask.astype(np.float32) / 255.0
    
    # 高斯模糊实现羽化效果
    if feather_radius > 0:
        kernel_size = feather_radius * 2 + 1
        mask_feathered = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
    else:
        mask_feathered = mask_float
    
    # 扩展维度用于广播
    mask_3ch = mask_feathered[:, :, np.newaxis]
    
    # 融合：mask区域使用transformed，非mask区域使用original
    blended = (original.astype(np.float32) * (1 - mask_3ch) + 
               transformed.astype(np.float32) * mask_3ch)
    
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    # ============================================================
    # 解析配置
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", 
                       default="./test/texture+b+p/config/inpainting/options/test/ir-sde-brushnet.yml", 
                       type=str, help="Path to option YAML file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # 创建结果目录
    util.mkdirs((
        path for key, path in opt["path"].items()
        if not key == "experiments_root"
        and "pretrain_model" not in key
        and "resume" not in key
    ))

    # 设置日志
    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    # ============================================================
    # 初始化颜色先验生成器 和 LUT 处理器
    # ============================================================
    color_prior_gen = None
    lut_processor = None
    lut_opt = opt.get("lut", None)
    if lut_opt and ColorPriorGenerator is not None:
        lut_path = lut_opt.get("path", None)
        if lut_path and os.path.exists(lut_path):
            color_prior_gen = ColorPriorGenerator(
                lut_path=lut_path,
                alpha=lut_opt.get("alpha", 0.7),
                beta=lut_opt.get("beta", 0.3),
                inpaint_method=lut_opt.get("inpaint_method", "telea")
            )
            logger.info(f"[ColorPriorGenerator] Loaded LUT from {lut_path}")
            
            # 同时加载 LUT 处理器（用于置信度加权变换）
            if LUTProcessor is not None:
                lut_processor = LUTProcessor(lut_path)
                logger.info(f"[LUTProcessor] Loaded for confidence-weighted blending")
        else:
            logger.warning(f"[ColorPriorGenerator] LUT path not found: {lut_path}")
    else:
        logger.warning("[ColorPriorGenerator] LUT config not provided, using original mode")

    prior_method = lut_opt.get("prior_method", "quality") if lut_opt else "quality"
    gt_mode = lut_opt.get("gt_mode", "partial") if lut_opt else "partial"
    lut_strength = lut_opt.get("lut_strength", 0.7) if lut_opt else 0.7
    lut_smooth_radius = lut_opt.get("lut_smooth_radius", 5) if lut_opt else 5
    logger.info(f"[Config] gt_mode={gt_mode}, prior_method={prior_method}, lut_strength={lut_strength}, smooth_radius={lut_smooth_radius}")
    
    # ============================================================
    # 加载数据集
    # ============================================================
    mask_root = opt['degradation']['mask_root']
    mask_dataset = MaskDataset(mask_root)
    
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        mask_loader = DataLoader(mask_dataset, batch_size=1, shuffle=True)
        logger.info(f"Number of test images in [{dataset_opt['name']}]: {len(test_set)}")

    # ============================================================
    # 创建模型
    # ============================================================
    model = create_model(opt)
    device = model.device

    # SDE设置
    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device
    )
    sde.set_model(model.model)

    S_sde = str_util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device
    )
    S_sde.set_model(model.models)

    # ============================================================
    # 开始测试
    # ============================================================
    test_times = []
    mask_iterator = iter(mask_loader)

    for g, test_data in enumerate(tqdm(test_loader, desc="Testing")):
        test_set_name = test_loader.dataset.opt["name"]
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name, 'brushnet')
        util.mkdir(dataset_dir)

        img_path = test_data["GT_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 获取输入数据
        Y_GT = test_data["GT"]           # [B, 3, H, W]
        X_GT = test_data["GT_gray"]      # [B, 1, H, W]
        X_LQ = test_data["GT_edge"]      # [B, 1, H, W]

        # 获取mask
        try:
            mask = next(mask_iterator)
        except StopIteration:
            mask_iterator = iter(mask_loader)
            mask = next(mask_iterator)
        
        # mask 语义: 1=已知, 0=缺失（SDE 约定）
        # mask_np: 255=需要修复, 0=已知（cv2 约定，用于 ColorPriorGenerator）

        # ============================================================
        # 第一步：对输入图像去噪
        # ============================================================
        img_np = (Y_GT[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3]
        mask_np = ((1 - mask[0, 0]).numpy() * 255).astype(np.uint8)  # [H, W], 255=需要修复
        
        # 去噪（边缘保持双边滤波）
        denoised_np = denoise_image(img_np, d=9, sigma_color=75, sigma_space=75)
        logger.info(f"[{img_name}] Denoised input image")
        
        # ============================================================
        # 第二步：置信度加权 LUT 变换（与训练一致）
        # ============================================================
        lut_transformed_np = None  # 新 LUT 逻辑的结果（用于 prior 同步）
        color_prior_np = None
        confidence_np = None
        
        if lut_processor is not None:
            # 使用置信度加权 LUT 变换（与训练 denoising_model.py 逻辑对齐）
            lut_transformed_np, confidence_np = apply_lut_with_confidence(
                lut_processor, 
                denoised_np, 
                lut_strength=lut_strength,
                smooth_radius=lut_smooth_radius
            )
            logger.info(f"[{img_name}] Applied confidence-weighted LUT (strength={lut_strength})")
        
        # ============================================================
        # 第三步：生成颜色先验
        # ============================================================
        # 颜色先验 = 非 mask 区域使用新 LUT 结果 + mask 区域使用传统修复
        
        if color_prior_gen is not None:
            # 使用 ColorPriorGenerator 生成传统先验（含 mask 区域修复）
            prior_result = color_prior_gen.generate(denoised_np, mask_np, method=prior_method)
            prior_from_gen = prior_result['color_prior']  # [H, W, 3] float32 (0-255)
            
            if confidence_np is None:
                confidence_np = prior_result['confidence']
            
            # ============ 关键：同步 color_prior 的非 mask 区域 ============
            # 与训练代码一致：非 mask 区域使用新 LUT 逻辑，mask 区域保留传统填充
            # mask_np: 255=需要修复, 0=已知
            mask_bool = mask_np > 127  # True=需要修复
            
            if lut_transformed_np is not None:
                # 非 mask 区域（已知区域）：使用新 LUT 变换结果
                # mask 区域（缺失区域）：保留传统 inpaint 填充
                color_prior_np = lut_transformed_np.copy()
                color_prior_np[mask_bool] = prior_from_gen[mask_bool]
            else:
                # 没有 LUT 处理器，回退到完全使用 ColorPriorGenerator
                color_prior_np = prior_from_gen
            
            logger.info(f"[{img_name}] Generated color prior (non-mask: new LUT, mask: traditional inpaint)")
            
        elif lut_transformed_np is not None:
            # 有 LUT 但没有 ColorPriorGenerator
            # 全图使用 LUT 结果（无法填充 mask 区域）
            color_prior_np = lut_transformed_np
            logger.info(f"[{img_name}] Using LUT result as prior (no inpaint for mask region)")
        else:
            color_prior_np = None
            confidence_np = None
        
        # 转换为 tensor
        color_prior = None
        confidence = None
        
        if color_prior_np is not None:
            color_prior = torch.from_numpy(
                np.clip(color_prior_np, 0, 255).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
        if confidence_np is not None:
            confidence = torch.from_numpy(
                confidence_np.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # ============================================================
        # 第四步：准备原始褪色图 tensor（用于 Mu-Denoiser）
        # ============================================================
        original_degraded = Y_GT.clone()  # [B, 3, H, W] 原始褪色图

        # ============================================================
        # 第五步：推理
        # ============================================================
        noisy_state = sde.noise_state(Y_GT * mask)
        noisy_states = S_sde.noise_state(X_LQ * mask)
        
        model.feed_data(
            noisy_state, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ * mask,
            color_prior=color_prior,
            confidence=confidence,
            original_degraded=original_degraded
        )
        
        tic = time.time()
        model.test(
            sde, 
            save_states=False, 
            save_dir=dataset_dir, 
            GT=Y_GT, 
            mask=mask, 
            S_sde=S_sde, 
            S_GT=X_GT, 
            S_LQ=noisy_states, 
            dis=model.dis
        )
        toc = time.time()
        test_times.append(toc - tic)

        # ============================================================
        # 保存结果
        # ============================================================
        visuals = model.get_current_visuals()
        SR_img = visuals["Output"]
        output = util.tensor2img(SR_img.squeeze())  # uint8
        LQ_ = util.tensor2img(visuals["Input"].squeeze())
        GT_ = util.tensor2img(visuals["GT"].squeeze())

        # 保存修复结果
        suffix = opt.get("suffix", None)
        if suffix:
            save_path = os.path.join(dataset_dir, f"{img_name}{suffix}.png")
        else:
            save_path = os.path.join(dataset_dir, f"{img_name}_restored.png")
        util.save_img(output, save_path)

        # 保存对比图
        GT_path = os.path.join(dataset_dir, f"{img_name}_input.png")
        util.save_img(GT_, GT_path)
        
        # 保存颜色先验（如果有）
        if color_prior is not None:
            prior_img = util.tensor2img(color_prior.squeeze())
            prior_path = os.path.join(dataset_dir, f"{img_name}_prior.png")
            util.save_img(prior_img, prior_path)
        
        # 保存 LUT 变换结果（如果有）
        if lut_transformed_np is not None:
            lut_img = np.clip(lut_transformed_np, 0, 255).astype(np.uint8)
            lut_path = os.path.join(dataset_dir, f"{img_name}_lut.png")
            cv2.imwrite(lut_path, cv2.cvtColor(lut_img, cv2.COLOR_RGB2BGR))

        logger.info(f"[{img_name}] Saved to {save_path}, time: {toc-tic:.3f}s")

    # 统计
    avg_time = np.mean(test_times) if test_times else 0
    logger.info(f"\nTest completed! Average time per image: {avg_time:.3f}s")
    logger.info(f"Results saved to: {opt['path']['results_root']}")


if __name__ == "__main__":
    import os
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
