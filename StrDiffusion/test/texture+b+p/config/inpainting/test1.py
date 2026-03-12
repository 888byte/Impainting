# python test/texture/config/inpainting/test.py -opt test/texture/config/inpainting/options/test/ir-sde.yml
import argparse
import logging
import math
import os
import random
import sys
import copy
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import options as option
from models import create_model
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
import str_utils as str_util
import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import os.path as osp
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image


def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    #dist.init_process_group(backend=backend, **kwargs)


def fft_compute_color(img_col, center=False):
    assert img_col.shape[0] != 1, "Should be color image"
    c, h, w = img_col.shape
    idx_list_ = []
    x_mag = np.zeros((c, h, w))
    x_phase = np.zeros((c, h, w))
    x_fft = []
    for i in range(c):
        img = img_col[i]
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        if center:
            dft = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])
        idx = (mag == 0)
        mag[idx] = 1.0
        magnitude_spectrum = np.log(mag)
        phase_spectrum = cv2.phase(dft[:, :, 0], dft[:, :, 1])
        x_mag[i] = magnitude_spectrum
        x_phase[i] = phase_spectrum
        idx_list_.append(idx)
    return x_fft, x_mag, x_phase, idx_list_


# ================= [MOD 1] 新增：按文件名建立 mask 索引，用于“图像名 + _mask”精确匹配 =================
# 为什么要这么改：
#   你原来的写法是单独开一个 mask dataloader + shuffle，每张图 next() 一个 mask -> 随机 mask，必然不对应文件名
#   推理/测试通常要“指定 mask 对应指定图”，所以必须按 img_name 去找 mask 文件
class MaskByName:
    def __init__(self, mask_root: str, image_size: int = 256):
        self.mask_root = mask_root
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        # 建一个 {stem: full_path} 的索引，stem = 不含扩展名的文件名
        self.mask_map = {}
        for root, _, files in os.walk(mask_root, topdown=True):
            for name in files:
                stem = os.path.splitext(name)[0]
                self.mask_map[stem] = osp.join(root, name)

        print(f"[MaskByName] indexed masks: {len(self.mask_map)} from {mask_root}")

    def get_mask_tensor(self, img_stem: str) -> torch.Tensor:
        """
        img_stem: 原图不含扩展名的名字，例如 real-147_center
        约定 mask: img_stem + "_mask"（扩展名不固定）
        """
        key = img_stem + "_mask"
        if key not in self.mask_map:
            # 给一个更友好的报错：列出一些候选 key 方便你排查命名
            sample_keys = list(self.mask_map.keys())[:10]
            raise FileNotFoundError(
                f"Mask not found for image '{img_stem}'. Expected mask stem '{key}'. "
                f"mask_root='{self.mask_root}'. Example mask stems: {sample_keys}"
            )

        path = self.mask_map[key]
        # 注意：convert('1') 得到 0/1 二值图，ToTensor 后是 0/1 float
        mask = self.transform(Image.open(path).convert('1'))
        # 保持你原逻辑：return 1-mask（0表示masked, 1表示unmasked）
        return 1 - mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", default="./test/texture/config/inpainting/options/test/ir-sde.yml",
                        type=str, help="Path to option YMAL file.")
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # ================= [MOD 2] 新增：测试阶段强制只用一张卡（避免 DataParallel/多卡占用） =================
    # 为什么要这么改：
    #   你的环境里经常被“GPU动态调度”塞进多个可见GPU，某些代码会误用多卡（尤其 gpu_ids 变成多元素时）
    #   这里在 create_model 之前把 gpu_ids 截断成 1 个，确保不会走 DataParallel
    if opt.get("gpu_ids", None):
        opt["gpu_ids"] = [opt["gpu_ids"][0]]
    else:
        opt["gpu_ids"] = [0]
    torch.cuda.set_device(opt["gpu_ids"][0])

    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )

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

    mask_root = opt['degradation']['mask_root']

    # ================= [MOD 3] 新增：用 MaskByName 替代“随机mask dataloader” =================
    mask_provider = MaskByName(mask_root, image_size=256)

    # create test dataloader
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info(
            "Number of test images in [{:s}]: {:d}".format(dataset_opt["name"], len(test_set))
        )

    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],                 # 不改 T（你训练=400，这里保持一致）
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device
    )
    sde.set_model(model.model)

    S_sde = str_util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],                 # 不改 T
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device
    )
    S_sde.set_model(model.models)

    # ================= [MOD 4] 新增：save_states 做成可控开关，默认 False（显著提速） =================
    # 为什么要这么改：
    #   save_states=True 可能会保存每个 diffusion step 的中间状态（T=400），IO 极慢
    #   默认关掉，确实需要再在 yml 里加 save_states: true 或直接改这里
    save_states = bool(opt.get("save_states", False))

    test_times = []
    torch.backends.cudnn.benchmark = True

    for epoch in range(0, 1):
        for g, train_data in enumerate(test_loader):
            test_set_name = test_loader.dataset.opt["name"]
            logger.info("\nTesting [{:s}]...".format(test_set_name))

            dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
            util.mkdir(dataset_dir)

            img_path = train_data["GT_path"][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            Y_GT, X_GT, X_LQ = train_data["GT"], train_data["GT_gray"], train_data["GT_edge"]

            dataset_dir = os.path.join(dataset_dir, 'new')
            util.mkdir(dataset_dir)

            # ================= [MOD 5] 核心修改：按文件名取对应 mask，而不是 next() 随机取 =================
            # 约定：mask 文件名 stem = img_name + "_mask"
            mask = mask_provider.get_mask_tensor(img_name)

            noisy_state = sde.noise_state(Y_GT * mask)
            noisy_states = S_sde.noise_state(X_LQ * mask)

            model.feed_data(noisy_state, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ * mask)

            tic = time.time()
            with torch.no_grad():
                model.test(
                    sde,
                    save_states=save_states,   # ================= [MOD 4] 用开关控制 =================
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

            visuals = model.get_current_visuals()
            SR_img = visuals["Output"]
            output = util.tensor2img(SR_img.squeeze())
            GT_ = util.tensor2img(visuals["GT"].squeeze())

            suffix = opt["suffix"]
            if suffix:
                save_img_path = os.path.join(dataset_dir, img_name + suffix + ".png")
            else:
                save_img_path = os.path.join(dataset_dir, img_name + "_f.png")
            util.save_img(output, save_img_path)

            SR_img_y = SR_img * mask
            output_y = util.tensor2img(SR_img_y.squeeze())
            save_img_path = os.path.join(dataset_dir, img_name + "_m.png")
            util.save_img(output_y, save_img_path)

            GT_img_path = os.path.join(dataset_dir, img_name + "_r.png")
            util.save_img(GT_, GT_img_path)

    if len(test_times) > 0:
        logger.info(f"[Test] Avg time per image: {sum(test_times)/len(test_times):.3f}s, "
                    f"min={min(test_times):.3f}s, max={max(test_times):.3f}s")


if __name__ == "__main__":
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
