"""
训练入口：纹理去噪模型（Texture Denoising）训练脚本。
主要流程：
1) 读取配置 -> 设置分布式与随机种子
2) 创建训练集/掩码集 -> DataLoader
3) 构建模型与 SDE -> 进入训练循环
4) 记录日志、保存模型
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
# from IPython import embed
import cv2
import options as option
from models import create_model

#sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler

from data.util import bgr2ycbcr

import str_utils as str_util

import os.path as osp
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
# from PIL import Imageoptim
from torch.utils.data import DataLoader
from itertools import cycle



def init_dist(backend="nccl", **kwargs):
    """分布式训练初始化（多进程/多卡）。"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # 当前进程 rank
    num_gpus = torch.cuda.device_count()  # 可用 GPU 数量
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # 初始化默认分布式进程组


def main():
    # 1) 解析配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="./train/texture/config/inpainting/options/train/ir-sde.yml", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # 将 dict 转为 NoneDict：缺失 key 返回 None，避免 KeyError
    opt = option.dict_to_nonedict(opt)

    # 训练随机种子
    seed = opt["train"]["manual_seed"]
    # 2) 分布式训练设置
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 3) 断点恢复（resume）处理

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    # 4) 输出目录与日志
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                )
            )
            os.system("rm ./log")
            os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # 配置日志（必须在 log 目录准备好后执行）
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")


    # 5) 构建掩码数据集（inpainting 需要 mask）
    mask_root = opt['degradation']['mask_root']
    train_set_mask = Datasetset_mask(mask_root)
    
    # 6) 创建训练集/数据加载器
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            train_loader_mask = DataLoader(train_set_mask, batch_size=dataset_opt["batch_size"], shuffle=True, num_workers=dataset_opt["n_workers"], drop_last=True)
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))


    assert train_loader is not None

    # 7) 创建模型与 SDE（扩散过程）
    model = create_model(opt) 
    device = model.device
    # 断点恢复训练
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
    
    S_sde = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    
    
    # 8) 训练主循环
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    for epoch in range(start_epoch, total_epochs + 1):
        
        mask_iterator = iter(train_loader_mask)
        if opt["dist"]:
            train_sampler.set_epoch(epoch)
        for gg, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break

            # 纹理训练：Y_GT 为 RGB，X_GT/X_LQ 为结构引导（灰度/边缘）
            Y_GT, X_GT, X_LQ = train_data["GT"],train_data["GT_gray"],train_data["GT_edge"] ##completed grayscale and edge images
            
            # 读取 mask（随机遮挡）
            try:
                mask = next(mask_iterator)
            except StopIteration:
                mask_iterator = iter(train_loader_mask)
                mask = next(mask_iterator)
            
            # 采样扩散过程的随机时间步与状态
            timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_GT*mask)###timestep>2
            # 送入模型：states=xt，mu=masked 图像，x0=原图
            model.feed_data(states,Y_GT*mask, Y_GT, mask, S_sde, X_GT, X_LQ) # xt, mu, x0, mask
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # 添加：按频率保存训练状态
            if current_step % opt.get("logger", {}).get("save_training_state_freq", 5000) == 0:
                if rank <= 0:
                    logger.info("Saving training states for step {}.".format(current_step))
                    model.save_training_state(epoch, current_step)

            if error.value:
                sys.exit(0)
        
        # 每个 epoch 结束保存模型
        logger.info("Saving models and training states.")
        model.save("new")
        #model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()





from PIL import Image
class Datasetset_mask(Dataset):
    """读取 mask 数据集（用于 inpainting 遮挡）。"""
    def __init__(self, THE_PATH):
        data = []
        for root, dirs, files in os.walk(THE_PATH, topdown=True):
            for name in files:
                data.append(osp.join(root, name))
                
        self.data = data   
        print("mask dataset length: {}".format(len(self.data)))
        self.image_size = 256

        self.transform = transforms.Compose([
        	transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
            transforms.RandomHorizontalFlip(),
        	transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path = self.data[i]
        mask = self.transform(Image.open(path).convert('1'))
        return 1 - mask   # 0 表示被遮挡，1 表示保留

if __name__ == "__main__":
    import os

    cuda_home = os.getenv("CUDA_HOME")

    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
