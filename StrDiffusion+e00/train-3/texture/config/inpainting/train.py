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


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_tb_scalar_map(logs):
    scalar_map = {}

    def add(tag, key):
        value = _as_float(logs.get(key))
        if value is not None:
            scalar_map[tag] = value

    add("train/loss_main", "loss_main")
    add("train/loss_total", "loss_total")
    add("train/loss_hole", "loss_hole")
    add("train/loss_known", "loss_known")
    add("train/loss_hole_weighted", "loss_hole_weighted")
    add("train/loss_mu_total", "loss_mu_total")
    add("train/loss_mu_ss", "loss_mu_ss")
    add("train/loss_mu_tv", "loss_mu_tv")
    add("train/lr_main", "lr_main")
    add("train/lr_mu", "lr_mu")
    add("train/ema_texture_loss", "ema_texture_loss")
    add("train/ema_total_loss", "ema_total_loss")
    add("train/best_metric", "best_metric")
    add("train/best_total_metric", "best_total_metric")
    add("train/mask_hole_ratio", "mask_hole_ratio")
    add("train/texture_condition_gap", "texture_condition_gap")
    add("stats/color_prior_hole_mean", "stats_color_prior_hole_mean")
    add("stats/color_prior_hole_std", "stats_color_prior_hole_std")
    add("stats/color_prior_hole_white_ratio", "stats_color_prior_hole_white_ratio")
    add("stats/confidence_hole_mean", "stats_confidence_hole_mean")
    add("stats/condition_known_mean", "stats_condition_known_mean")
    add("stats/condition_known_std", "stats_condition_known_std")
    add("stats/mu_known_mean", "stats_mu_known_mean")
    add("stats/mu_known_std", "stats_mu_known_std")
    add("train/condition_target_gap", "condition_target_gap")
    add("train/degraded_target_gap", "degraded_target_gap")

    return scalar_map


def _prepare_tb_image(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return None

    image = tensor.detach().float().cpu()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 2:
        image = image.unsqueeze(0)
    if image.dim() != 3:
        return None
    if image.shape[0] not in (1, 3):
        image = image[:1]
    return image.clamp(0.0, 1.0)


def _log_tb_training_images(tb_logger, debug_info, current_step):
    if not debug_info:
        return

    image_tags = {
        "train_vis/original_degraded": "original_degraded",
        "train_vis/reference_degraded": "reference_degraded",
        "train_vis/denoised_observed_mask_aware": "denoised_observed_mask_aware",
        "train_vis/condition_lut": "condition_lut",
        "train_vis/condition_mu": "condition_mu",
        "train_vis/mu_clean_lut": "mu_clean_lut",
        "train_vis/training_target": "training_target",
        "train_vis/color_prior": "color_prior",
        "train_vis/confidence": "confidence",
        "train_vis/mask_known": "mask_known",
        "train_vis/mask_hole": "mask_hole",
        "train_vis/structure_gray_from_target": "structure_gray_from_target",
        "train_vis/structure_edge_from_target": "structure_edge_from_target",
    }

    for tag, key in image_tags.items():
        image = _prepare_tb_image(debug_info.get(key))
        if image is not None:
            tb_logger.add_image(tag, image, current_step)



def _build_structure_from_target(training_target, device):
    """Build S_GT(gray) and S_LQ(edge) from the target-domain training target."""
    gray_list = []
    edge_list = []
    for batch_idx in range(training_target.shape[0]):
        rgb = training_target[batch_idx].detach().cpu().permute(1, 2, 0).numpy()
        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        gray_list.append(torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0))
        edge_list.append(torch.from_numpy(edge.astype(np.float32) / 255.0).unsqueeze(0))
    structure_gray = torch.stack(gray_list, dim=0).to(device)
    structure_edge = torch.stack(edge_list, dim=0).to(device)
    return structure_gray, structure_edge

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
            tb_image_freq = int(opt.get("logger", {}).get("tb_image_freq", opt["logger"]["print_freq"] * 10))
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
    # best:     以主纹理损失为准，更贴近最终修复质量
    # best_total: 以总损失为准，便于追踪包含 D_mu 的整体最优点
    best_texture_loss = float("inf")
    best_total_loss = float("inf")
    ema_texture_loss = None
    ema_total_loss = None
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
        best_texture_loss = float(resume_state.get("best_texture_loss", best_texture_loss))
        best_total_loss = float(resume_state.get("best_total_loss", best_total_loss))
        ema_texture_loss = resume_state.get("ema_texture_loss", ema_texture_loss)
        ema_total_loss = resume_state.get("ema_total_loss", ema_total_loss)
        if ema_texture_loss is not None:
            ema_texture_loss = float(ema_texture_loss)
        if ema_total_loss is not None:
            ema_total_loss = float(ema_total_loss)
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
                # Mural Inpainting 模式：
                # degraded      -> 真实缺损外观输入（给条件链）
                # degraded_full -> 完整褪色图（仅用于训练目标生成）
                Y_degraded = train_data["degraded"]
                Y_degraded_full = train_data.get("degraded_full", train_data["degraded"])
                Y_GT = train_data["GT"]              # LUT变换后的GT（目标）
                X_GT = train_data["GT_gray"]         # 灰度图
                X_LQ = train_data["GT_edge"]         # 边缘图
                mask = train_data["mask"]            # 使用数据集的mask
                color_prior = train_data["color_prior"]
                confidence = train_data["confidence"]
                # ChromaRefiner 需要的 conf_lut
                conf_lut = train_data.get("conf_lut", None)
            else:
                # 原始模式：兼容旧数据集
                Y_GT, X_GT, X_LQ = train_data["GT"], train_data["GT_gray"], train_data["GT_edge"]
                Y_degraded = Y_GT  # 原始模式下 degraded = GT
                Y_degraded_full = Y_GT
                
                # 从外部加载mask
                try:
                    mask = next(mask_iterator)
                except StopIteration:
                    mask_iterator = iter(train_loader_mask)
                    mask = next(mask_iterator)
                
                # 使用 ColorPriorGenerator 生成先验
                color_prior = None
                confidence = None
                conf_lut = None
                if color_prior_gen is not None:
                    color_prior, confidence = color_prior_gen.generate_tensor(
                        Y_GT,
                        1 - mask,
                        device=device,
                        method=opt["datasets"]["train"].get("prior_method", "quality"),
                    )
            
            # 确保数据在正确设备上
            Y_degraded = Y_degraded.to(device)
            Y_degraded_full = Y_degraded_full.to(device)
            Y_GT = Y_GT.to(device)
            X_GT = X_GT.to(device)
            X_LQ = X_LQ.to(device)
            mask = mask.to(device)
            if color_prior is not None:
                color_prior = color_prior.to(device)
            if confidence is not None:
                confidence = confidence.to(device)
            if conf_lut is not None:
                conf_lut = conf_lut.to(device)
            
            # ============ SDE?? ============
            # Mask semantics:
            #   mural dataset mask: 1=hole; mask_for_sde/self.mask: 1=known.
            #   BrushNet receives 1-self.mask, i.e. 1=hole.
            if is_mural_mode:
                mask_for_sde = 1 - mask
                complement_error = torch.max(torch.abs(mask_for_sde + mask - 1.0)).item()
                assert complement_error < 1e-4, (
                    f"mural mask and mask_for_sde must be complementary, error={complement_error:.6f}"
                )
            else:
                mask_for_sde = mask

            mask_min = float(mask_for_sde.min().item())
            mask_max = float(mask_for_sde.max().item())
            assert mask_min >= -1e-4 and mask_max <= 1.0 + 1e-4, (
                f"mask_for_sde must stay in [0,1], got min={mask_min:.6f}, max={mask_max:.6f}"
            )
            hole_ratio = float((1 - mask_for_sde).mean().item())

            if is_mural_mode:
                assert model.lut_processor is not None, "mural mode requires LUTProcessor for target-domain training_target/condition_lut"
                # Mural target domain:
                #   x0/GT/reverse target = LUT(denoised(degraded_full)).
                # Condition/mu uses only inference-observable input:
                #   condition_lut = LUT(mask-aware denoised(observed_degraded)).
                with torch.no_grad():
                    denoised_ref = model._denoise_image(Y_degraded_full, mask_known=None)
                    training_target, _ = model._build_lut_transformed(denoised_ref)

                    denoised_observed_mask_aware = model._denoise_image(
                        Y_degraded, mask_known=mask_for_sde
                    )
                    condition_lut, _ = model._build_lut_transformed(
                        denoised_observed_mask_aware
                    )

                    if color_prior is not None:
                        color_prior_for_sde = (
                            condition_lut * mask_for_sde
                            + color_prior * (1 - mask_for_sde)
                        )
                    else:
                        color_prior_for_sde = None

                    if confidence is not None:
                        confidence_for_sde = (
                            torch.ones_like(mask_for_sde) * mask_for_sde
                            + confidence * (1 - mask_for_sde)
                        )
                    else:
                        confidence_for_sde = None

                    # MuCleanr/Mu-Denoiser now cleans target-domain condition_lut.
                    # If disabled or no usable weights are loaded/trained yet, the
                    # helper falls back to condition_lut; the final SDE mu is masked.
                    mu_clean_lut = model.compute_mu_clean_no_grad(
                        condition_lut, mask_for_sde, confidence_for_sde
                    )
                    condition_mu = mu_clean_lut * mask_for_sde

                timesteps, states = sde.generate_random_states(
                    x0=training_target, mu=condition_mu
                )

                X_GT_for_sde, X_LQ_for_sde = _build_structure_from_target(
                    training_target, device
                )

                model.feed_data(
                    states, condition_mu, training_target, mask_for_sde,
                    S_sde, X_GT_for_sde, X_LQ_for_sde,
                    color_prior=color_prior_for_sde,
                    confidence=confidence_for_sde,
                    conf_lut=conf_lut,
                    original_degraded=Y_degraded,
                    reference_degraded=Y_degraded_full,
                    condition_lut=condition_lut,
                    mu_clean_lut=mu_clean_lut,
                    denoised_observed_mask_aware=denoised_observed_mask_aware,
                )
            else:
                # Legacy path remains unchanged: x0/GT/reverse target are Y_GT.
                training_target = Y_GT
                condition_lut = Y_GT
                mu_clean_lut = Y_GT
                denoised_observed_mask_aware = None
                condition_mu = Y_GT * mask_for_sde
                color_prior_for_sde = color_prior
                confidence_for_sde = confidence

                timesteps, states = sde.generate_random_states(
                    x0=training_target, mu=condition_mu
                )
                model.feed_data(
                    states, condition_mu, training_target, mask_for_sde,
                    S_sde, X_GT, X_LQ,
                    color_prior=color_prior_for_sde,
                    confidence=confidence_for_sde,
                    conf_lut=conf_lut,
                    original_degraded=Y_degraded,
                    reference_degraded=Y_degraded_full,
                    condition_lut=condition_lut,
                    mu_clean_lut=mu_clean_lut,
                    denoised_observed_mask_aware=denoised_observed_mask_aware,
                )
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            
            # ============ 调试保存 ============
            if debug_logger is not None and debug_logger.should_save(current_step):
                # 获取调试信息
                debug_info = getattr(model, '_debug_refiner_info', None)
                
                if debug_info is not None:
                    # 新的调试格式：
                    # Input -> Denoised -> ColorChanged -> Prior -> Original+Mask -> Mask
                    original = debug_info.get('original_degraded', Y_degraded)
                    denoised = debug_info.get('denoised_observed_mask_aware', debug_info.get('denoised_original', None))
                    color_changed = debug_info.get('training_target', debug_info.get('color_changed', training_target))
                    prior = debug_info.get('color_prior', color_prior_for_sde)
                    orig_with_mask = debug_info.get('condition_mu', condition_mu)
                    mask_img = debug_info.get('mask_known', mask_for_sde)
                    
                    # 计算去噪效果
                    if denoised is not None:
                        diff = (denoised - original).abs().mean().item()
                        logger.info(f"[Denoise Debug] step={current_step}, denoise_diff={diff:.6f}")
                else:
                    original = Y_degraded
                    denoised = None
                    color_changed = training_target
                    prior = color_prior_for_sde
                    orig_with_mask = condition_mu
                    mask_img = mask_for_sde
                
                debug_logger.save_training_state_v2(
                    step=current_step,
                    original=original,
                    denoised=denoised,
                    color_changed=color_changed,
                    color_prior=prior if prior is not None else Y_degraded,
                    original_with_mask=orig_with_mask,
                    mask=mask if is_mural_mode else (1 - mask),
                )
            # ============ 调试保存完成 ============

            #———————————————————检查点保存—————————————————————————
            #———————————————————检查点保存—————————————————————————
            # 统一在每个iter取一次log，后面打印/保存都复用，避免重复调用
            logs = model.get_current_log()

            # 主纹理损失：用于 best checkpoint
            if "loss_main" in logs:
                cur_texture_loss = float(logs["loss_main"])
            elif "loss" in logs:
                cur_texture_loss = float(logs["loss"])
            elif "l_total" in logs:
                cur_texture_loss = float(logs["l_total"])
            else:
                cur_texture_loss = 0.0
                for k, v in logs.items():
                    if str(k).startswith("l_"):
                        cur_texture_loss += float(v)

            # 总损失：用于观测整体优化过程
            if "loss_total" in logs:
                cur_total_loss = float(logs["loss_total"])
            elif "loss" in logs:
                cur_total_loss = float(logs["loss"])
            elif "l_total" in logs:
                cur_total_loss = float(logs["l_total"])
            else:
                cur_total_loss = 0.0
                for k, v in logs.items():
                    if str(k).startswith("l_"):
                        cur_total_loss += float(v)

            # EMA平滑（用于挑 best）
            if ema_texture_loss is None:
                ema_texture_loss = cur_texture_loss
            else:
                ema_texture_loss = ema_beta * ema_texture_loss + (1.0 - ema_beta) * cur_texture_loss
            if ema_total_loss is None:
                ema_total_loss = cur_total_loss
            else:
                ema_total_loss = ema_beta * ema_total_loss + (1.0 - ema_beta) * cur_total_loss
            logs["ema_texture_loss"] = float(ema_texture_loss)
            logs["ema_total_loss"] = float(ema_total_loss)
            logs["best_metric"] = float(min(best_texture_loss, ema_texture_loss))
            logs["best_total_metric"] = float(min(best_total_loss, ema_total_loss))

            # 仅rank0保存，避免多卡写冲突
            if rank <= 0:
                # 每 save_freq 次保存一次（权重+state）- 使用相同标签
                if save_freq > 0 and (current_step % save_freq == 0):
                    logger.info(f"[ckpt] iter={current_step} Saving models and training states.")
                    # 使用iter作为标签，确保权重和state文件对应
                    # 生成: {iter}_G.pth, {iter}_D.pth, {iter}.state
                    model.save(str(current_step))
                    model.save_training_state(
                        epoch,
                        current_step,
                        label=str(current_step),
                        extra_state={
                            "best_texture_loss": float(best_texture_loss),
                            "best_total_loss": float(best_total_loss),
                            "ema_texture_loss": float(ema_texture_loss),
                            "ema_total_loss": float(ema_total_loss),
                        },
                    )

                # 保存loss最低(best)（用EMA判定更稳）- 使用 "best" 标签
                if ema_texture_loss < best_texture_loss:
                    best_texture_loss = ema_texture_loss
                    logs["best_metric"] = float(best_texture_loss)
                    log_msg = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> [best-texture] loss_main: {:.4e} loss_total: {:.4e}".format(
                        epoch,
                        current_step,
                        model.get_current_learning_rate(),
                        logs.get('loss_main', logs.get('loss', 0)),
                        logs.get('loss_total', logs.get('loss', 0)),
                    )
                    if 'loss_mu_ss' in logs:
                        log_msg += f" loss_mu_ss={logs['loss_mu_ss']:.4f}"
                    if 'loss_mu_total' in logs:
                        log_msg += f" loss_mu={logs['loss_mu_total']:.4f}"
                    logger.info(log_msg)
                    model.save("best")
                    model.save_training_state(
                        epoch,
                        current_step,
                        label="best",
                        extra_state={
                            "best_texture_loss": float(best_texture_loss),
                            "best_total_loss": float(best_total_loss),
                            "ema_texture_loss": float(ema_texture_loss),
                            "ema_total_loss": float(ema_total_loss),
                        },
                    )

                if ema_total_loss < best_total_loss:
                    best_total_loss = ema_total_loss
                    logs["best_total_metric"] = float(best_total_loss)
                    log_msg = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> [best-total] loss_main: {:.4e} loss_total: {:.4e}".format(
                        epoch,
                        current_step,
                        model.get_current_learning_rate(),
                        logs.get('loss_main', logs.get('loss', 0)),
                        logs.get('loss_total', logs.get('loss', 0)),
                    )
                    if 'loss_mu_ss' in logs:
                        log_msg += f" loss_mu_ss={logs['loss_mu_ss']:.4f}"
                    if 'loss_mu_total' in logs:
                        log_msg += f" loss_mu={logs['loss_mu_total']:.4f}"
                    logger.info(log_msg)
                    model.save("best_total")
                    model.save_training_state(
                        epoch,
                        current_step,
                        label="best_total",
                        extra_state={
                            "best_texture_loss": float(best_texture_loss),
                            "best_total_loss": float(best_total_loss),
                            "ema_texture_loss": float(ema_texture_loss),
                            "ema_total_loss": float(ema_total_loss),
                        },
                    )
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
                            tb_logger.add_scalar(f"raw/{k}", v, current_step)
                if opt["use_tb_logger"] and "debug" not in opt["name"] and rank <= 0:
                    for tag, value in _build_tb_scalar_map(logs).items():
                        tb_logger.add_scalar(tag, value, current_step)
                    if current_step % tb_image_freq == 0:
                        _log_tb_training_images(
                            tb_logger, model.get_current_training_debug(), current_step
                        )
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
            model.save_training_state(
                total_epochs,
                current_step,
                label="latest",
                extra_state={
                    "best_texture_loss": float(best_texture_loss),
                    "best_total_loss": float(best_total_loss),
                    "ema_texture_loss": float(ema_texture_loss) if ema_texture_loss is not None else None,
                    "ema_total_loss": float(ema_total_loss) if ema_total_loss is not None else None,
                },
            )
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
