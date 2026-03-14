"""Configuration loading and backward-compatible normalization."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict


def _resolve_path(base_dir: str, value: str) -> str:
    if not value:
        return value
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    return normalize_config(cfg, config_path=path)


def normalize_config(cfg: Dict[str, Any], config_path: str = "") -> Dict[str, Any]:
    cfg = deepcopy(cfg)
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    project_root = os.path.abspath(os.path.join(base_dir, "..")) if os.path.basename(base_dir) == "configs" else os.getcwd()

    data_cfg = cfg.setdefault("data", {})
    for key in ("train_npz", "val_npz", "test_npz", "sample_index", "train_index", "val_index", "test_index"):
        if key in data_cfg and isinstance(data_cfg[key], str):
            data_cfg[key] = _resolve_path(project_root, data_cfg[key])

    pretrained_cfg = cfg.setdefault("pretrained", {})
    for key in ("alignment_ckpt", "raman_encoder_ckpt", "prototype_bank_path"):
        if key in pretrained_cfg and isinstance(pretrained_cfg[key], str):
            pretrained_cfg[key] = _resolve_path(project_root, pretrained_cfg[key])

    train_cfg = cfg.setdefault("train", {})
    if "save_dir" in train_cfg and isinstance(train_cfg["save_dir"], str):
        train_cfg["save_dir"] = _resolve_path(project_root, train_cfg["save_dir"])

    diff_cfg = cfg.setdefault("diffusion", {})
    if "beta_start" in diff_cfg and "beta_0" not in diff_cfg:
        diff_cfg["beta_0"] = diff_cfg["beta_start"]
    if "beta_end" in diff_cfg and "beta_T" not in diff_cfg:
        diff_cfg["beta_T"] = diff_cfg["beta_end"]

    bridge_cfg = cfg.setdefault("bridge", {})
    proto_cfg = bridge_cfg.setdefault("prototype_bank", {})
    proto_cfg.setdefault("path", pretrained_cfg.get("prototype_bank_path", ""))
    if proto_cfg.get("path"):
        proto_cfg["path"] = _resolve_path(project_root, proto_cfg["path"])
    proto_cfg.setdefault("normalize", True)

    bridge_cfg.setdefault("enable", False)
    bridge_cfg.setdefault("mode", "pred")
    bridge_cfg.setdefault("use_gate", False)
    bridge_cfg.setdefault("use_distill", False)
    bridge_cfg.setdefault("use_group_sampler", False)
    bridge_cfg.setdefault("posterior_temp", 0.07)
    bridge_cfg.setdefault("distill_weight", 0.1)
    bridge_cfg.setdefault("prototype_top_k", 0)
    bridge_cfg.setdefault("refresh_each_epoch", False)
    bridge_cfg.setdefault("teacher_temp", 0.07)

    missing_cfg = cfg.setdefault("missing_modality", {})
    missing_cfg.setdefault("enable", False)
    missing_cfg.setdefault("drop_prob", 0.3)
    missing_cfg.setdefault("lambda_pred", 0.1)

    physics_cfg = cfg.setdefault("physics", {})
    physics_cfg.setdefault("enable", False)
    physics_cfg.setdefault("use_cycle_model", "auto")
    physics_cfg.setdefault("use_spec_color_consistency", False)
    physics_cfg.setdefault("use_parent_consistency", False)
    physics_cfg.setdefault("use_aug_consistency", False)
    physics_cfg.setdefault("use_damage_constraint", False)
    physics_cfg.setdefault("lambda_cycle", 0.2)
    physics_cfg.setdefault("lambda_spec_color", 0.1)
    physics_cfg.setdefault("lambda_spec_pred_consistency", 0.0)
    physics_cfg.setdefault("lambda_parent_consistency", 0.05)
    physics_cfg.setdefault("lambda_aug_consistency", 0.05)
    physics_cfg.setdefault("lambda_damage_mono", 0.01)
    physics_cfg.setdefault("lambda_damage_smooth", 0.01)
    physics_cfg.setdefault("damage_requires_order", True)
    physics_cfg.setdefault("parent_consistency_level", "auto")
    physics_cfg.setdefault("side_consistency_scale", 0.25)
    physics_cfg.setdefault("color_consistency_space", "lab")
    physics_cfg.setdefault("low_confidence_skip_physics", True)
    physics_cfg.setdefault("warmup_steps", 2000)
    physics_cfg.setdefault("t_max", 30)
    physics_cfg.setdefault("exclude_t0", True)
    physics_cfg.setdefault("cond_dependent", True)
    physics_cfg.setdefault("cond_hidden", 128)
    physics_cfg.setdefault("per_channel_k", True)
    physics_cfg.setdefault("learn_c_inf", True)
    physics_cfg.setdefault("init_k", 1.0)

    return cfg
