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

# ============ BrushNet 支持 ============
# 颜色先验生成器 (按需导入)
try:
    from color_prior_generator import ColorPriorGenerator
    HAS_COLOR_PRIOR = True
except ImportError:
    HAS_COLOR_PRIOR = False
    print("[Warning] ColorPriorGenerator 未找到，BrushNet模式不可用")

# 调试工具
try:
    from debug_utils import DebugLogger
    HAS_DEBUG_UTILS = True
except ImportError:
    HAS_DEBUG_UTILS = False
    print("[Warning] DebugLogger 未找到，调试模式不可用")
# ============ BrushNet 支持 ============


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group


def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="./train/texture/config/inpainting/options/train/ir-sde.yml", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### set random seed
    seed = opt["train"]["manual_seed"]
    #### distributed training settings
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

    torch.backends.cudnn.benchmark = True

    ###### Predictor&Corrector train ######

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

    #### mkdir and loggers
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

        # config loggers. Before it, the log will not work
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


    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count()=", torch.cuda.device_count())
    print("opt[gpu_ids]=", opt["gpu_ids"])


    #———————————————————检查点保存—————————————————————————
    #———————————————————检查点保存—————————————————————————
    # 1) 读取保存频率（你要每5000次保存一次，就在 yml 里设 logger.save_checkpoint_freq: 5000）
    #save_freq = int(opt.get("logger", {}).get("save_checkpoint_freq", 5000))
    save_freq = 5000

    # 2) 训练loss最优保存（best），用EMA平滑避免抖动
    best_loss = float("inf")
    ema_loss = None
    ema_beta = 0.98  # 越大越平滑（0.98~0.995都可）

    # 3) 确保 training_state 目录存在（避免 save_training_state 失败）
    if rank <= 0 and opt.get("path", {}).get("training_state", None):
        os.makedirs(opt["path"]["training_state"], exist_ok=True)
    #———————————————————检查点保存—————————————————————————
    #———————————————————检查点保存—————————————————————————

    # ============ BrushNet 初始化 ============
    use_brushnet = opt.get('brushnet', {}).get('enabled', False)
    color_prior_gen = None
    debug_logger = None
    
    if use_brushnet and HAS_COLOR_PRIOR:
        # LUT配置现在在 datasets.train 下
        train_opt = opt.get('datasets', {}).get('train', {})
        lut_path = train_opt.get('lut_path', None)
        if lut_path and os.path.exists(lut_path):
            color_prior_gen = ColorPriorGenerator(
                lut_path=lut_path,
                alpha=train_opt.get('lut_alpha', 0.7),
                beta=train_opt.get('lut_beta', 0.3),
                inpaint_method=train_opt.get('lut_inpaint_method', 'telea')
            )
            logger.info(f"[BrushNet] ColorPriorGenerator 已初始化: {lut_path}")
        else:
            logger.warning(f"[BrushNet] LUT文件未找到: {lut_path}")
    
    # 调试模式
    debug_cfg = opt.get('debug', {})
    if debug_cfg.get('enabled', False) and HAS_DEBUG_UTILS:
        debug_logger = DebugLogger(
            log_dir=debug_cfg.get('log_dir', './debug_logs'),
            enabled=True,
            save_freq=debug_cfg.get('save_freq', 500)
        )
        logger.info(f"[Debug] 调试模式已启用，保存频率: {debug_cfg.get('save_freq', 500)}")
    # ============ BrushNet 初始化 ============

    mask_root = opt['degradation']['mask_root']
    train_set_mask = Datasetset_mask(mask_root)

    #### create train and val dataloader
    dataset_ratio = 1  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            train_loader_mask = DataLoader(
                train_set_mask,
                batch_size=dataset_opt["batch_size"],
                shuffle=True,
                num_workers=dataset_opt["n_workers"],
                drop_last=True
            )
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


    torch.cuda.set_device(opt["gpu_ids"][0])
    #### create model
    model = create_model(opt)
    device = model.device

    #### resume training
    if resume_state:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )
        start_epoch = int(float(resume_state["epoch"]))
        current_step = int(float(resume_state["iter"]))
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

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

    #### training
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

            # ============ 区分数据集模式 ============
            is_mural_mode = 'degraded' in train_data  # mural_inpainting 数据集特有
            
            if is_mural_mode:
                # Mural Inpainting 模式：使用数据集提供的完整数据
                Y_degraded = train_data["degraded"]  # 褪色图像（输入）
                Y_GT = train_data["GT"]              # LUT变换后的GT（目标）
                X_GT = train_data["GT_gray"]         # 灰度图
                X_LQ = train_data["GT_edge"]         # 边缘图
                mask = train_data["mask"]            # 使用数据集的mask
                color_prior = train_data["color_prior"]
                confidence = train_data["confidence"]
            else:
                # 原始模式：兼容旧数据集
                Y_GT, X_GT, X_LQ = train_data["GT"], train_data["GT_gray"], train_data["GT_edge"]
                Y_degraded = Y_GT  # 原始模式下 degraded = GT
                
                # 从外部加载mask
                try:
                    mask = next(mask_iterator)
                except StopIteration:
                    mask_iterator = iter(train_loader_mask)
                    mask = next(mask_iterator)
                
                # 使用 ColorPriorGenerator 生成先验
                color_prior = None
                confidence = None
                if color_prior_gen is not None:
                    color_prior, confidence = color_prior_gen.generate_tensor(
                        Y_GT, 1 - mask, device=device
                    )
            
            # 确保数据在正确设备上
            Y_degraded = Y_degraded.to(device)
            Y_GT = Y_GT.to(device)
            X_GT = X_GT.to(device)
            X_LQ = X_LQ.to(device)
            mask = mask.to(device)
            if color_prior is not None:
                color_prior = color_prior.to(device)
            if confidence is not None:
                confidence = confidence.to(device)
            
            # ============ SDE训练 ============
            # 注意：mask约定为 1=已知, 0=缺失
            # 对于mural数据集，mask来自数据集（1=缺失），需要取反
            if is_mural_mode:
                mask_for_sde = 1 - mask  # 转换为 1=已知, 0=缺失
            else:
                mask_for_sde = mask  # 原始数据集的mask已经是 1=已知
            
            timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_degraded*mask_for_sde)
            model.feed_data(states, Y_degraded*mask_for_sde, Y_GT, mask_for_sde, S_sde, X_GT, X_LQ, 
                           color_prior=color_prior, confidence=confidence)
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            
            # ============ 调试保存 ============
            if debug_logger is not None and debug_logger.should_save(current_step):
                debug_logger.save_training_state(
                    step=current_step,
                    input_image=Y_degraded,  # 褪色图像（输入）
                    gt=Y_GT,                  # LUT变换后的GT（目标）
                    color_prior=color_prior if color_prior is not None else Y_degraded,
                    confidence=confidence if confidence is not None else torch.zeros_like(mask),
                    mask=mask if is_mural_mode else (1 - mask)  # 统一为 1=缺失
                )
            # ============ 调试保存完成 ============

            #———————————————————检查点保存—————————————————————————
            #———————————————————检查点保存—————————————————————————
            # 统一在每个iter取一次log，后面打印/保存都复用，避免重复调用
            logs = model.get_current_log()

            # 从logs中尽量稳健地取“总loss”
            if "loss" in logs:
                cur_loss = float(logs["loss"])
            elif "l_total" in logs:
                cur_loss = float(logs["l_total"])
            else:
                # 把所有以 l_ 开头的项加起来作为总loss
                cur_loss = 0.0
                for k, v in logs.items():
                    if str(k).startswith("l_"):
                        cur_loss += float(v)

            # EMA平滑（用于挑 best）
            if ema_loss is None:
                ema_loss = cur_loss
            else:
                ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * cur_loss

            # 仅rank0保存，避免多卡写冲突
            if rank <= 0:
                # 每 save_freq 次保存一次（权重+state）
                if save_freq > 0 and (current_step % save_freq == 0):
                    logger.info(f"[ckpt] iter={current_step} Saving models and training states.")
                    # 用iter命名，避免覆盖
                    model.save(str(current_step))
                    model.save_training_state(epoch, current_step)

                # 保存loss最低(best)（用EMA判定更稳）
                if ema_loss < best_loss:
                    best_loss = ema_loss
                    logger.info(f"[best] iter={current_step} best_ema_loss={best_loss:.6e}. Saving best model/state.")
                    model.save("best")
                    model.save_training_state(epoch, current_step)
            #———————————————————检查点保存—————————————————————————
            #———————————————————检查点保存—————————————————————————

            if current_step % opt["logger"]["print_freq"] == 0:
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

            if error.value:
                sys.exit(0)

        #———————————————————检查点保存—————————————————————————
        #———————————————————检查点保存—————————————————————————
        # 原本这里“每个epoch都保存一次”，已移除。
        # 现在只在 iter%save_freq==0 时保存，以及loss最优(best)时保存。
        #———————————————————检查点保存—————————————————————————
        #———————————————————检查点保存—————————————————————————

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        # 如你希望最后也落一份state（便于resume），可以保留：
        #———————————————————检查点保存—————————————————————————
        #———————————————————检查点保存—————————————————————————
        try:
            model.save_training_state(total_epochs, "latest")
        except Exception as e:
            logger.info(f"[warn] save_training_state(latest) failed: {e}")
        #———————————————————检查点保存—————————————————————————
        #———————————————————检查点保存—————————————————————————

        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


from PIL import Image
class Datasetset_mask(Dataset):
    """The class to load the dataset"""
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
        return 1 - mask   #  0 is masked, 1 is unmasked


if __name__ == "__main__":
    import os

    cuda_home = os.getenv("CUDA_HOME")

    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)

    
    main()
