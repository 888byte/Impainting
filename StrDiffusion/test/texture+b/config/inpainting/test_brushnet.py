# -*- coding: utf-8 -*-
"""
test_brushnet.py - 壁画颜色修复推理脚本（BrushNet集成版）

运行命令：
python test/texture+b/config/inpainting/test_brushnet.py -opt test/texture+b/config/inpainting/options/test/ir-sde-brushnet.yml

功能：
- 加载训练好的BrushNet模型
- 对每张输入图像生成颜色先验和置信度
- 使用SDE反向采样进行修复
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


def main():
    # ============================================================
    # 解析配置
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", 
                       default="./test/texture+b/config/inpainting/options/test/ir-sde-brushnet.yml", 
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
    # 初始化颜色先验生成器
    # ============================================================
    color_prior_gen = None
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
        else:
            logger.warning(f"[ColorPriorGenerator] LUT path not found: {lut_path}")
    else:
        logger.warning("[ColorPriorGenerator] LUT config not provided, using original mode")

    prior_method = lut_opt.get("prior_method", "quality") if lut_opt else "quality"
    
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

        # ============================================================
        # 生成颜色先验和置信度
        # ============================================================
        color_prior = None
        confidence = None
        
        if color_prior_gen is not None:
            # 将tensor转换为numpy进行处理
            img_np = (Y_GT[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # [H, W, 3]
            mask_np = ((1 - mask[0, 0]).numpy() * 255).astype(np.uint8)  # [H, W], 255=需要修复
            
            # 生成颜色先验
            prior_result = color_prior_gen.generate(img_np, mask_np, method=prior_method)
            color_prior_np = prior_result['color_prior']    # [H, W, 3] float32
            confidence_np = prior_result['confidence']       # [H, W] float32
            
            # 转换为tensor
            color_prior = torch.from_numpy(
                np.clip(color_prior_np, 0, 255).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
            confidence = torch.from_numpy(
                confidence_np.astype(np.float32)
            ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            logger.info(f"[{img_name}] Generated color prior and confidence")

        # ============================================================
        # 推理
        # ============================================================
        noisy_state = sde.noise_state(Y_GT * mask)
        noisy_states = S_sde.noise_state(X_LQ * mask)
        
        model.feed_data(
            noisy_state, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ * mask,
            color_prior=color_prior,
            confidence=confidence
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
