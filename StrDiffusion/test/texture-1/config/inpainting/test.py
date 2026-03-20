# -*- coding: utf-8 -*-
"""官方骨架兼容的推理入口。

启动方式:
    python test.py -opt options/test/ir-sde.yml
    python test.py -opt options/test/ir-sde-brushnet.yml
    python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false

Mask 语义:
    - mural 推理模式下，输入 mask 白色表示待修复区域
    - dataset 输出:
        mask_hole  = 1 表示待修复区域
        mask_known = 1 表示已知区域
    - 官方结构链继续使用 mask_known
    - BrushNet / MGLC / 最终纹理网络只使用 mask_hole

读取规则:
    - 保留官方配置骨架，不改 YAML 字段
    - mural 模式下，mask_root 优先使用 degradation.mask_root
    - 若 dataroot_degraded 为空，则回退使用 dataroot_GT 作为样本图目录
"""

import argparse
import logging
import os
import os.path as osp
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import options as option
import str_utils as str_util
import utils as util
from data import create_dataloader, create_dataset
from models import create_model


class DatasetsetMask(Dataset):
    """Legacy random-mask loader kept for the original official test path."""

    def __init__(self, root_path):
        self.data = []
        for root, _, files in os.walk(root_path, topdown=True):
            for name in files:
                self.data.append(osp.join(root, name))
        self.data = sorted(self.data)
        self.transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256), interpolation=Image.NEAREST),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index]
        mask = self.transform(Image.open(path).convert("1"))
        return 1 - mask  # legacy semantics: 1=known, 0=masked


def _flatten_overrides(nested_overrides):
    overrides = []
    for item in nested_overrides:
        if isinstance(item, list):
            overrides.extend(item)
        else:
            overrides.append(item)
    return overrides


def _save_visuals_for_sample(visuals, sample_dir, img_name, suffix=None, save_raw=False):
    os.makedirs(sample_dir, exist_ok=True)
    output = util.tensor2img(visuals["Output"].squeeze())
    final_name = "final.png" if suffix is None else img_name + suffix + ".png"
    util.save_img(output, os.path.join(sample_dir, final_name))

    if save_raw and "RawOutput" in visuals:
        raw = util.tensor2img(visuals["RawOutput"].squeeze())
        util.save_img(raw, os.path.join(sample_dir, "raw_pred.png"))

    if "GT" in visuals:
        gt_img = util.tensor2img(visuals["GT"].squeeze())
        util.save_img(gt_img, os.path.join(sample_dir, "gt.png"))


def _resolve_runtime_dataset_opt(dataset_opt, opt):
    """Resolve runtime roots without mutating the YAML file on disk."""
    runtime_opt = dict(dataset_opt)
    if runtime_opt.get("mode") != "mural_inference":
        return runtime_opt

    degradation_opt = opt.get("degradation", {})
    mask_root = degradation_opt.get("mask_root")
    if mask_root:
        runtime_opt["dataroot_mask"] = mask_root

    if not runtime_opt.get("dataroot_degraded"):
        fallback_root = runtime_opt.get("dataroot_GT")
        if fallback_root:
            runtime_opt["dataroot_degraded"] = fallback_root

    return runtime_opt


def _run_legacy_test(test_loader, opt, logger, model, sde, s_sde):
    mask_root = opt["degradation"]["mask_root"]
    mask_loader = DataLoader(DatasetsetMask(mask_root), batch_size=1, shuffle=True)
    mask_iterator = iter(mask_loader)

    for _, train_data in enumerate(test_loader):
        test_set_name = test_loader.dataset.opt["name"]
        logger.info("\nTesting [%s]...", test_set_name)
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name, "new")
        util.mkdir(dataset_dir)

        img_path = train_data["GT_path"][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        y_gt = train_data["GT"]
        x_gt = train_data["GT_gray"]
        x_lq = train_data["GT_edge"]

        try:
            mask = next(mask_iterator)
        except StopIteration:
            mask_iterator = iter(mask_loader)
            mask = next(mask_iterator)

        noisy_state = sde.noise_state(y_gt * mask)
        noisy_states = s_sde.noise_state(x_lq * mask)
        model.feed_data(noisy_state, y_gt * mask, y_gt, mask, s_sde, x_gt, x_lq * mask)
        model.test(
            sde,
            save_states=True,
            save_dir=dataset_dir,
            GT=y_gt,
            mask=mask,
            S_sde=s_sde,
            S_GT=x_gt,
            S_LQ=noisy_states,
            dis=model.dis,
        )
        visuals = model.get_current_visuals(need_GT=True)
        _save_visuals_for_sample(visuals, dataset_dir, img_name, suffix=opt["suffix"])


def _run_enhanced_test(test_loader, opt, logger, model, sde, s_sde):
    dataset_opt = test_loader.dataset.opt
    test_set_name = dataset_opt["name"]
    need_gt = bool(dataset_opt.get("dataroot_GT"))

    for _, sample in enumerate(test_loader):
        img_name = sample["stem"][0]
        logger.info("\nTesting [%s] sample [%s]...", test_set_name, img_name)
        sample_dir = os.path.join(opt["path"]["results_root"], test_set_name, img_name)
        util.mkdir(sample_dir)

        degraded = sample["degraded"]
        mask_known = sample["mask_known"]
        mask_hole = sample["mask_hole"]
        gt = sample["GT"] if "GT" in sample else None

        condition = degraded * mask_known
        model.feed_data(
            state=None,
            LQ=condition,
            GT=gt if gt is not None else degraded,
            mask=mask_known,
            S_sde=s_sde,
            S_GT=None,
            S_LQ=None,
            color_prior=sample["color_prior"] if "color_prior" in sample else None,
            confidence=sample["confidence"] if "confidence" in sample else None,
            original_degraded=degraded,
            mask_hole=mask_hole,
            sample_name=img_name,
        )
        model.test(
            sde,
            save_states=bool(opt.get("inference", {}).get("save_states", False)),
            save_dir=sample_dir,
            GT=gt,
            mask=mask_known,
            S_sde=s_sde,
            S_GT=None,
            S_LQ=None,
            dis=model.dis,
        )
        visuals = model.get_current_visuals(need_GT=need_gt)
        _save_visuals_for_sample(
            visuals,
            sample_dir,
            img_name,
            suffix=None,
            save_raw=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        default="./test/texture/config/inpainting/options/test/ir-sde.yml",
        type=str,
        help="Path to option YAML file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        nargs="+",
        default=[],
        help="Override YAML options, e.g. --set texture_core.enabled=false",
    )
    args = parser.parse_args()

    overrides = _flatten_overrides(args.overrides)
    opt = option.parse(args.opt, is_train=False, overrides=overrides)
    opt = option.dict_to_nonedict(opt)

    util.mkdirs(
        path
        for key, path in opt["path"].items()
        if key != "experiments_root" and "pretrain_model" not in key and "resume" not in key
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

    dataset_items = sorted(opt["datasets"].items())
    CONFIG_MISSING_DATASETS = '配置中没有 datasets。'
    if not dataset_items:
        raise RuntimeError(CONFIG_MISSING_DATASETS)

    test_loaders = []
    for _, dataset_opt_raw in dataset_items:
        dataset_opt = _resolve_runtime_dataset_opt(dataset_opt_raw, opt)
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        logger.info("Number of test images in [%s]: %d", dataset_opt["name"], len(test_set))
        if dataset_opt.get("mode") == "mural_inference":
            logger.info(
                "Resolved mural roots: degraded=%s, mask=%s",
                dataset_opt.get("dataroot_degraded"),
                dataset_opt.get("dataroot_mask"),
            )

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

    s_sde = str_util.IRSDE(
        max_sigma=opt["sde"]["max_sigma"],
        T=opt["sde"]["T"],
        schedule=opt["sde"]["schedule"],
        eps=opt["sde"]["eps"],
        device=device,
    )
    s_sde.set_model(model.models)

    for test_loader in test_loaders:
        dataset_mode = test_loader.dataset.opt["mode"]
        start_time = time.time()
        if dataset_mode == "mural_inference":
            _run_enhanced_test(test_loader, opt, logger, model, sde, s_sde)
        else:
            _run_legacy_test(test_loader, opt, logger, model, sde, s_sde)
        logger.info(
            "Finished dataset [%s] in %.2f seconds.",
            test_loader.dataset.opt["name"],
            time.time() - start_time,
        )


if __name__ == "__main__":
    main()
