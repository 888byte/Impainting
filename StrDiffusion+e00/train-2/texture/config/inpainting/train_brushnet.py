# -*- coding: utf-8 -*-
"""
train_brushnet.py - BrushNet集成训练脚本

功能说明：
---------
本脚本为BrushNet集成版本的训练入口，与原始train.py分离，
确保不影响原有训练流程。

使用方法：
---------
python train_brushnet.py -opt options/train/ir-sde-brushnet.yml

Author: Auto-generated for BrushNet Integration
"""

import argparse
import logging
import math
import os
import random
import sys
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2

import options as option
from models import create_model

import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import str_utils as str_util

import os.path as osp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from itertools import cycle

# 导入调试工具
from debug_utils import DebugLogger, save_debug_tensors


def init_dist(backend="nccl", **kwargs):
    """分布式训练初始化"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


class MaskDataset(Dataset):
    """掩码数据集"""
    def __init__(self, mask_root, image_size=256):
        data = []
        for root, dirs, files in os.walk(mask_root, topdown=True):
            for name in files:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    data.append(osp.join(root, name))
        
        self.data = data
        print(f"[MaskDataset] 加载掩码数量: {len(self.data)}")
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size), interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        path = self.data[i]
        mask = self.transform(Image.open(path).convert('1'))
        return 1 - mask  # 0 is masked, 1 is unmasked


def main():
    """主训练函数"""
    # ============================================================
    # 解析命令行参数
    # ============================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, required=True, help="Path to option YAML file.")
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    opt = option.parse(args.opt, is_train=True)
    opt = option.dict_to_nonedict(opt)
    
    # ============================================================
    # 分布式设置
    # ============================================================
    if args.launcher == "none":
        opt["dist"] = False
        rank = -1
        print("[Train] 单卡训练模式")
    else:
        opt["dist"] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(f"[Train] 分布式训练模式, rank={rank}, world_size={world_size}")
    
    torch.backends.cudnn.benchmark = True
    seed = opt["train"]["manual_seed"]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # ============================================================
    # 日志设置
    # ============================================================
    if rank <= 0:
        if opt["path"].get("resume_state", None) is None:
            util.mkdir_and_rename(opt["path"]["experiments_root"])
            util.mkdirs((
                path for key, path in opt["path"].items()
                if not key == "experiments_root"
                and "pretrain_model" not in key
                and "resume" not in key
            ))
        
        util.setup_logger(
            "base", opt["path"]["log"], "train_" + opt["name"],
            level=logging.INFO, screen=True, tofile=True
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        
        # TensorBoard
        if opt.get("use_tb_logger", True):
            from torch.utils.tensorboard import SummaryWriter
            tb_logger = SummaryWriter(log_dir=f"log/{opt['name']}/tb_logger/")
        else:
            tb_logger = None
    else:
        util.setup_logger("base", opt["path"]["log"], "train", level=logging.INFO, screen=False)
        logger = logging.getLogger("base")
        tb_logger = None
    
    # ============================================================
    # Debug模式设置
    # ============================================================
    debug_opt = opt.get("debug", {})
    debug_logger = DebugLogger(
        log_dir=debug_opt.get("log_dir", "./debug_logs"),
        enabled=debug_opt.get("enabled", False),
        save_freq=debug_opt.get("save_freq", 500)
    )
    
    # ============================================================
    # 检查点保存设置
    # ============================================================
    save_freq = opt.get("logger", {}).get("save_checkpoint_freq", 5000)
    best_loss = float("inf")
    ema_loss = None
    ema_beta = 0.98
    
    if rank <= 0 and opt.get("path", {}).get("training_state", None):
        os.makedirs(opt["path"]["training_state"], exist_ok=True)
    
    # ============================================================
    # 数据集创建
    # ============================================================
    # 掩码数据集
    mask_root = opt.get('degradation', {}).get('mask_root', None)
    if mask_root and os.path.exists(mask_root):
        train_set_mask = MaskDataset(mask_root, opt["datasets"]["train"].get("GT_size", 256))
    else:
        train_set_mask = None
        logger.warning("[Train] 未找到掩码目录，将使用数据集内置掩码")
    
    # 主数据集
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            # 添加LUT路径到dataset_opt
            lut_opt = opt.get("lut", {})
            dataset_opt["lut_path"] = lut_opt.get("path", "./pigment_lut33.npz")
            dataset_opt["gt_mode"] = dataset_opt.get("gt_mode", "mixed")
            dataset_opt["debug_mode"] = debug_opt.get("enabled", False)
            
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            
            if train_set_mask is not None:
                train_loader_mask = DataLoader(
                    train_set_mask,
                    batch_size=dataset_opt["batch_size"],
                    shuffle=True,
                    num_workers=dataset_opt["n_workers"],
                    drop_last=True
                )
            else:
                train_loader_mask = None
            
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            
            if opt["dist"]:
                train_sampler = DistIterSampler(train_set, world_size, rank, 1)
                total_epochs = int(math.ceil(total_iters / train_size))
            else:
                train_sampler = None
            
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            
            if rank <= 0:
                logger.info(f"[Train] 训练图像数量: {len(train_set)}, 迭代数: {train_size}")
                logger.info(f"[Train] 总Epochs: {total_epochs}, 总迭代: {total_iters}")
    
    # ============================================================
    # 模型创建
    # ============================================================
    torch.cuda.set_device(opt["gpu_ids"][0])
    model = create_model(opt)
    device = model.device
    
    # ============================================================
    # SDE设置
    # ============================================================
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
    
    # ============================================================
    # Resume训练
    # ============================================================
    if opt["path"].get("resume_state", None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id)
        )
        option.check_resume(opt, resume_state["iter"])
        start_epoch = int(float(resume_state["epoch"]))
        current_step = int(float(resume_state["iter"]))
        model.resume_training(resume_state)
        logger.info(f"[Train] 从 epoch {start_epoch}, iter {current_step} 恢复训练")
    else:
        start_epoch = 0
        current_step = 0
    
    # ============================================================
    # 训练循环
    # ============================================================
    logger.info(f"[Train] 开始训练, epoch: {start_epoch}, iter: {current_step}")
    
    error = mp.Value('b', False)
    
    for epoch in range(start_epoch, total_epochs + 1):
        if train_loader_mask is not None:
            mask_iterator = iter(train_loader_mask)
        
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        
        for gg, train_data in enumerate(train_loader):
            current_step += 1
            
            if current_step > total_iters:
                break
            
            # ============================================================
            # 获取训练数据
            # ============================================================
            # 检查数据集类型（mural_inpainting vs GT）
            if "color_prior" in train_data:
                # MuralInpaintingDataset格式
                Y_GT = train_data["gt"]
                degraded = train_data["degraded"]
                color_prior = train_data["color_prior"]
                confidence = train_data["confidence"]
                mask = train_data["mask"]
                
                # 对于边缘引导，使用灰度和边缘
                X_GT = degraded.mean(dim=1, keepdim=True)  # 灰度
                X_LQ = X_GT  # 简化处理
            else:
                # 原始GTDataset格式
                Y_GT = train_data["GT"]
                X_GT = train_data.get("GT_gray", Y_GT.mean(dim=1, keepdim=True))
                X_LQ = train_data.get("GT_edge", X_GT)
                color_prior = None
                confidence = None
                
                # 从mask数据集获取mask
                if train_loader_mask is not None:
                    try:
                        mask = next(mask_iterator)
                    except StopIteration:
                        mask_iterator = iter(train_loader_mask)
                        mask = next(mask_iterator)
                else:
                    # 生成随机mask
                    B, _, H, W = Y_GT.shape
                    mask = torch.ones(B, 1, H, W)
            
            # 确保张量在正确设备上
            Y_GT = Y_GT.to(device)
            mask = mask.to(device)
            
            if color_prior is not None:
                color_prior = color_prior.to(device)
                confidence = confidence.to(device)
            
            # ============================================================
            # SDE采样
            # ============================================================
            timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_GT * mask)
            
            # ============================================================
            # 模型前向传播
            # ============================================================
            if color_prior is not None:
                # BrushNet模式
                model.feed_data(
                    states, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ,
                    color_prior=color_prior, confidence=confidence
                )
            else:
                # 原始模式
                model.feed_data(states, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ)
            
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(current_step, warmup_iter=opt["train"]["warmup_iter"])
            
            # ============================================================
            # Debug保存
            # ============================================================
            if debug_logger.should_save(current_step) and color_prior is not None:
                debug_logger.save_training_state(
                    step=current_step,
                    input_image=Y_GT * mask,
                    gt=Y_GT,
                    color_prior=color_prior,
                    confidence=confidence,
                    mask=mask
                )
            
            # ============================================================
            # 检查点保存
            # ============================================================
            logs = model.get_current_log()
            
            if "loss" in logs:
                cur_loss = float(logs["loss"])
            elif "l_total" in logs:
                cur_loss = float(logs["l_total"])
            else:
                cur_loss = sum(float(v) for k, v in logs.items() if str(k).startswith("l_"))
            
            if ema_loss is None:
                ema_loss = cur_loss
            else:
                ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * cur_loss
            
            if rank <= 0:
                if save_freq > 0 and (current_step % save_freq == 0):
                    logger.info(f"[Checkpoint] iter={current_step} 保存模型和训练状态")
                    model.save(str(current_step))
                    model.save_training_state(epoch, current_step)
                
                if ema_loss < best_loss:
                    best_loss = ema_loss
                    logger.info(f"[Best] iter={current_step} best_ema_loss={best_loss:.6e}")
                    model.save("best")
                    model.save_training_state(epoch, current_step)
            
            # ============================================================
            # 日志打印
            # ============================================================
            if current_step % opt["logger"]["print_freq"] == 0:
                message = f"<epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{model.get_current_learning_rate():.3e}> "
                for k, v in logs.items():
                    message += f"{k}: {v:.4e} "
                    if tb_logger is not None and rank <= 0:
                        tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            
            if error.value:
                sys.exit(0)
    
    # ============================================================
    # 训练完成
    # ============================================================
    if rank <= 0:
        logger.info("[Train] 保存最终模型")
        model.save("latest")
        try:
            model.save_training_state(total_epochs, "latest")
        except Exception as e:
            logger.warning(f"[Train] 保存训练状态失败: {e}")
        
        logger.info("[Train] 训练完成")
        
        if tb_logger is not None:
            tb_logger.close()


if __name__ == "__main__":
    main()
