"""
Same GT + same mask -> multiple stochastic inpainting results.

Example:
python test/texture/config/inpainting/test_multi_sample.py ^
    -opt test/texture/config/inpainting/options/test/ir-sde-multi-sample.yml
"""

import argparse
import logging
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

import options as option
import str_utils as str_util
import utils as util
from data import create_dataloader, create_dataset
from file_utils import OrderedYaml
from models import create_model


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_diverse_init(known_tensor, mask, sigma, noise_scale=1.0, noise_known_region=False):
    """
    known_tensor: already-masked known region, e.g. Y_GT * mask
    mask: 1=known, 0=hole
    sigma: base stochastic scale
    noise_scale: amplify randomness; >1 means much more diverse but usually lower quality
    """
    noise = torch.randn_like(known_tensor) * sigma * noise_scale
    if noise_known_region:
        return known_tensor + noise
    return known_tensor + noise * (1 - mask)


def parse_test_options(opt_path):
    loader, _ = OrderedYaml()
    with open(opt_path, mode="r", encoding="utf-8") as f:
        opt = yaml.load(f, Loader=loader)

    gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
    print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    opt["is_train"] = False

    scale = 1
    if opt["distortion"] == "sr":
        scale = opt["degradation"]["scale"]
        opt["network_G"]["setting"]["upscale"] = scale

    for phase, dataset in opt["datasets"].items():
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        dataset["scale"] = scale

        is_lmdb = False
        if dataset.get("dataroot_GT", None) is not None:
            dataset["dataroot_GT"] = osp.expanduser(dataset["dataroot_GT"])
            if dataset["dataroot_GT"].endswith("lmdb"):
                is_lmdb = True
        if dataset.get("dataroot_LQ", None) is not None:
            dataset["dataroot_LQ"] = osp.expanduser(dataset["dataroot_LQ"])
            if dataset["dataroot_LQ"].endswith("lmdb"):
                is_lmdb = True
        dataset["data_type"] = "lmdb" if is_lmdb else "img"
        if dataset["mode"].endswith("mc"):
            dataset["data_type"] = "mc"
            dataset["mode"] = dataset["mode"].replace("_mc", "")

    for key, path in opt["path"].items():
        if path and key != "strict_load":
            opt["path"][key] = osp.expanduser(path)

    current_dir = osp.dirname(osp.abspath(__file__))
    opt["path"]["root"] = osp.abspath(osp.join(current_dir, osp.pardir, osp.pardir, osp.pardir))
    config_dir = osp.basename(current_dir)
    results_root = osp.join(opt["path"]["root"], "results", config_dir)
    opt["path"]["results_root"] = osp.join(results_root, opt["name"])
    opt["path"]["log"] = osp.join(results_root, opt["name"])

    return option.dict_to_nonedict(opt)


class SingleMaskProvider:
    def __init__(self, opt):
        image_size = int(opt.get("mask_size", 256))
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(image_size, image_size), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self.fixed_mask_path = osp.expanduser(opt.get("fixed_mask_path")) if opt.get("fixed_mask_path") else None
        self.mask_root = opt["degradation"].get("mask_root")

    def _load_mask_from_path(self, path):
        mask = self.transform(Image.open(path).convert("1"))
        return 1 - mask  # 0=masked, 1=visible

    def _find_mask_by_name(self, img_name):
        if not self.mask_root:
            raise FileNotFoundError("mask_root is empty and fixed_mask_path is not set.")

        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
        candidates = [osp.join(self.mask_root, img_name + "_mask" + ext) for ext in exts]
        for path in candidates:
            if osp.exists(path):
                return path

        raise FileNotFoundError(
            f"Can not find mask for image '{img_name}'. "
            f"Tried: {candidates}. "
            f"You can also set fixed_mask_path in yml."
        )

    def get_mask(self, img_name):
        if self.fixed_mask_path:
            path = self.fixed_mask_path
        else:
            path = self._find_mask_by_name(img_name)
        return osp.splitext(osp.basename(path))[0], self._load_mask_from_path(path)


def save_once(dataset_dir, img_name, sample_idx, mask_name, output, output_masked, gt_img, mask_img):
    sample_tag = f"s{sample_idx:03d}"
    util.save_img(output, osp.join(dataset_dir, f"{img_name}__{mask_name}__{sample_tag}_f.png"))
    util.save_img(output_masked, osp.join(dataset_dir, f"{img_name}__{mask_name}__{sample_tag}_m.png"))

    gt_path = osp.join(dataset_dir, f"{img_name}_r.png")
    if not osp.exists(gt_path):
        util.save_img(gt_img, gt_path)

    mask_path = osp.join(dataset_dir, f"{img_name}__{mask_name}_mask.png")
    if not osp.exists(mask_path):
        util.save_img(mask_img, mask_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        default="./test/texture/config/inpainting/options/test/ir-sde-multi-sample.yml",
        type=str,
        help="Path to option YAML file.",
    )
    args = parser.parse_args()

    opt = parse_test_options(args.opt)

    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if key != "experiments_root"
            and "pretrain_model" not in key
            and "resume" not in key
        )
    )

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=False,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    num_samples = int(opt.get("num_samples", 4))
    base_seed = int(opt.get("base_seed", 1234))
    seed_stride = int(opt.get("seed_stride", 1))
    save_states = bool(opt.get("save_states", False))
    output_tag = opt.get("output_tag", "multi_sample")
    texture_noise_scale = float(opt.get("texture_noise_scale", 1.0))
    structure_noise_scale = float(opt.get("structure_noise_scale", 1.0))
    noise_known_region = bool(opt.get("noise_known_region", False))
    fixed_image_name = opt.get("fixed_image_name")
    normalized_fixed_image_name = None
    if fixed_image_name:
        normalized_fixed_image_name = fixed_image_name[:-5] if fixed_image_name.endswith("_mask") else fixed_image_name

    mask_provider = SingleMaskProvider(opt)

    test_loader = None
    for _, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info("Number of test images in [{:s}]: {:d}".format(dataset_opt["name"], len(test_set)))

    if test_loader is None:
        raise RuntimeError("No test dataset found in options file.")

    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    sde.set_model(model.model)

    S_sde = str_util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    S_sde.set_model(model.models)

    test_times = []

    processed_images = 0
    scanned_image_names = []

    for train_data in test_loader:
        img_path = train_data["GT_path"][0]
        img_name = osp.splitext(osp.basename(img_path))[0]
        if len(scanned_image_names) < 10:
            scanned_image_names.append(img_name)

        if normalized_fixed_image_name and img_name != normalized_fixed_image_name:
            continue

        test_set_name = test_loader.dataset.opt["name"]
        dataset_dir = osp.join(opt["path"]["results_root"], test_set_name, output_tag, img_name)
        util.mkdir(dataset_dir)
        processed_images += 1

        Y_GT, X_GT, X_LQ = train_data["GT"], train_data["GT_gray"], train_data["GT_edge"]
        mask_name, mask = mask_provider.get_mask(img_name)
        mask = mask.unsqueeze(0)

        logger.info(
            "\nTesting image [%s] with fixed mask [%s], num_samples=%d, texture_noise_scale=%.3f, structure_noise_scale=%.3f, noise_known_region=%s",
            img_name,
            mask_name,
            num_samples,
            texture_noise_scale,
            structure_noise_scale,
            noise_known_region,
        )

        for sample_idx in range(num_samples):
            seed = base_seed + sample_idx * seed_stride
            set_random_seed(seed)

            noisy_state = build_diverse_init(
                Y_GT * mask,
                mask,
                sde.max_sigma,
                noise_scale=texture_noise_scale,
                noise_known_region=noise_known_region,
            )
            noisy_states = build_diverse_init(
                X_LQ * mask,
                mask,
                S_sde.max_sigma,
                noise_scale=structure_noise_scale,
                noise_known_region=noise_known_region,
            )
            model.feed_data(noisy_state, Y_GT * mask, Y_GT, mask, S_sde, X_GT, X_LQ * mask)

            tic = time.time()
            with torch.no_grad():
                model.test(
                    sde,
                    save_states=save_states,
                    save_dir=dataset_dir,
                    GT=Y_GT,
                    mask=mask,
                    S_sde=S_sde,
                    S_GT=X_GT,
                    S_LQ=noisy_states,
                    dis=model.dis,
                )
            toc = time.time()
            test_times.append(toc - tic)

            visuals = model.get_current_visuals()
            sr_img = visuals["Output"]
            output = util.tensor2img(sr_img.squeeze())
            output_masked = util.tensor2img((sr_img * mask).squeeze())
            gt_img = util.tensor2img(visuals["GT"].squeeze())
            mask_img = util.tensor2img(mask.squeeze())

            save_once(dataset_dir, img_name, sample_idx, mask_name, output, output_masked, gt_img, mask_img)
            logger.info("Saved sample %d/%d for [%s], seed=%d, time=%.3fs", sample_idx + 1, num_samples, img_name, seed, toc - tic)

    if processed_images == 0:
        logger.error(
            "No image was processed. fixed_image_name=%s, normalized_fixed_image_name=%s, first_dataset_image_names=%s",
            fixed_image_name,
            normalized_fixed_image_name,
            scanned_image_names,
        )

    if test_times:
        logger.info(
            "[Test] Avg time per sample: %.3fs, min=%.3fs, max=%.3fs",
            sum(test_times) / len(test_times),
            min(test_times),
            max(test_times),
        )


if __name__ == "__main__":
    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        print("CUDA_HOME environment variable is not set.")
    else:
        print("CUDA_HOME:", cuda_home)
    main()
