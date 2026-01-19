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
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
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
    # 【注意】这里保留了 Structure 的配置文件路径
    parser.add_argument("-opt", default="./train/structure/config/inpainting/options/train/ir-sde.yml", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
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
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

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


    #———————————————————检查点保存参数设置—————————————————————————
    #———————————————————检查点保存参数设置—————————————————————————
    # 1) 设置保存频率
    save_freq = 5000

    # 2) 训练loss最优保存（best），用EMA平滑避免抖动
    best_loss = float("inf")
    ema_loss = None
    ema_beta = 0.98  # 越大越平滑

    # 3) 确保 training_state 目录存在
    if rank <= 0 and opt.get("path", {}).get("training_state", None):
        os.makedirs(opt["path"]["training_state"], exist_ok=True)
    #———————————————————检查点保存参数设置—————————————————————————
    #———————————————————检查点保存参数设置—————————————————————————


    mask_root = opt['degradation']['mask_root']
    train_set_mask = Datasetset_mask(mask_root)
    
    #### create train and val dataloader
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
        
        # 【关键修改】判断 iter 是否为 "best"
        # 如果是 "best"，无法转为 int，通常意味着这是最优模型
        # 这里选择将 current_step 重置为 0（或者你可以手动指定一个值）
        if str(resume_state["iter"]) == "best":
            logger.warning("Resume iter is 'best', resetting current_step to 0.")
            current_step = 0
            # 尝试转换 epoch，如果也是字符串可能也需要处理，但通常 epoch 是数字
            start_epoch = int(resume_state["epoch"]) 
        else:
            current_step = int(resume_state["iter"])
            start_epoch = int(resume_state["epoch"])
            
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)
    
    S_sde = str_util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    
    
    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    # best_psnr 暂时没用到，保留以防万一
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

            # 【重要】保留 Structure 特有的数据解包逻辑
            _, X_GT, X_LQ = train_data["GT"],train_data["GT_gray"],train_data["GT_edge"] ##completed grayscale and edge images
            
            ## load mask information
            try:
                mask = next(mask_iterator)
            except StopIteration:
                mask_iterator = iter(train_loader_mask)
                mask = next(mask_iterator)
            
            timesteps, states = sde.generate_random_states(x0=X_GT, mu=X_LQ*mask)###timestep>2
            model.feed_data(states,X_LQ*mask, X_GT, mask, S_sde, X_GT, X_LQ) # xt, mu, x0, mask
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )

            #———————————————————检查点保存逻辑（从Texture移植）—————————————————————————
            #———————————————————检查点保存逻辑（从Texture移植）—————————————————————————
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
                # 1. 每 save_freq 次保存一次（权重+state）
                if save_freq > 0 and (current_step % save_freq == 0):
                    logger.info(f"[ckpt] iter={current_step} Saving models and training states.")
                    # 用iter命名，避免覆盖
                    model.save(str(current_step))
                    model.save_training_state(epoch, current_step)

                # 2. 保存loss最低(best)（用EMA判定更稳）
                if ema_loss < best_loss:
                    best_loss = ema_loss
                    logger.info(f"[best] iter={current_step} best_ema_loss={best_loss:.6e}. Saving best model/state.")
                    model.save("best")
                    model.save_training_state(epoch, current_step) 
            #———————————————————检查点保存逻辑（从Texture移植）—————————————————————————
            #———————————————————检查点保存逻辑（从Texture移植）—————————————————————————

            if current_step % opt["logger"]["print_freq"] == 0:
                # logs = model.get_current_log() # 上面已经取过了，直接用
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
        
        #———————————————————End of Epoch 保存逻辑—————————————————————————
        # 移除了每个epoch强制保存的逻辑，完全依赖 iter 计数保存
        #———————————————————————————————————————————————————————————————

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        # 尝试保存最后的 state，方便后续 resume
        try:
            model.save_training_state(total_epochs, "latest")
        except Exception as e:
            logger.info(f"[warn] save_training_state(latest) failed: {e}")

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