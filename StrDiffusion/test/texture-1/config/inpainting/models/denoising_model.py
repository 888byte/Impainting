# -*- coding: utf-8 -*-
"""官方骨架兼容的推理模型。

用法:
    由 ``test.py`` 创建，并继续沿用官方调用方式:
        sde.set_model(model.model)
        S_sde.set_model(model.models)
        model.feed_data(...)
        model.test(sde, ..., S_sde=S_sde, ..., dis=model.dis)

Mask 语义:
    - self.mask: mask_known, 1 表示已知区域
    - self.mask_hole: 1 表示待修复区域
    - BrushNet / MGLC 只接收 self.mask_hole
"""

import logging
import os
from collections import OrderedDict
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

import models.networks as networks
import utils as util
from color_prior_generator import ColorPriorGenerator
from lut_processor import LUTProcessor
from .base_model import BaseModel

try:
    from models.mu_denoiser import MuDenoiser, MuDenoiserTrainer

    HAS_MU_DENOISER = True
except ImportError:
    HAS_MU_DENOISER = False

logger = logging.getLogger("base")


def _ensure_batch_time_like(time, device):
    if isinstance(time, (int, float)):
        return torch.tensor([time], device=device)
    return time.to(device)


def _validate_mask_pair(mask_known: Optional[torch.Tensor], mask_hole: Optional[torch.Tensor], where: str):
    """Validate that mask_known and mask_hole are shape-aligned complements."""
    if mask_known is None or mask_hole is None:
        raise ValueError(f"{where}: mask_known and mask_hole must both be provided.")
    if mask_known.shape != mask_hole.shape:
        raise ValueError(
            f"{where}: mask shape mismatch, mask_known={tuple(mask_known.shape)}, "
            f"mask_hole={tuple(mask_hole.shape)}"
        )
    deviation = torch.max(torch.abs(mask_known + mask_hole - 1.0)).item()
    if deviation > 1e-4:
        raise ValueError(
            f"{where}: mask_known and mask_hole must be complementary; max deviation={deviation:.6f}"
        )


class DenoisingModel(BaseModel):
    """Inference-only denoising model that keeps the official G/Gs/Dis layout."""

    def __init__(self, opt):
        super().__init__(opt)
        self.model, self.models, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.models = self.models.to(self.device)
        self.dis = self.dis.to(self.device)

        gpu_ids = opt.get("gpu_ids", [])
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids)
            self.models = DataParallel(self.models, device_ids=gpu_ids)
            self.dis = DataParallel(self.dis, device_ids=gpu_ids)

        self.dataset_opt = next(iter(opt["datasets"].values()))
        self.inference_opt = opt.get("inference", {})
        self.restore_s_guidance = bool(opt.get("restore_S_guidance", False))
        self.discriminator_guidance = bool(
            self.inference_opt.get("discriminator_guidance", {}).get("enabled", False)
        )
        self.save_intermediates = bool(self.inference_opt.get("save_intermediates", False))
        # The model is trained with reverse_sde_step_mean in optimize_parameters.
        # Keep enhanced mural inference deterministic by default; this preserves
        # the SDE formulas while avoiding stochastic reverse noise saturating
        # unconstrained hole pixels to white.
        self.deterministic_reverse = bool(self.inference_opt.get("deterministic_reverse", True))
        # Optional inference-only projection: keep known pixels on target-domain
        # condition_mu during reverse sampling; holes remain predicted by the model.
        self.known_area_projection = bool(self.inference_opt.get("known_area_projection", True))
        self.gt_mode = self.dataset_opt.get("gt_mode", "partial")
        self.lut_delta_gain = max(0.0, float(self.dataset_opt.get("lut_delta_gain", 1.0) or 1.0))
        self.prior_method = str(self.dataset_opt.get("prior_method", "quality")).lower()
        self.inference_mode = self.inference_opt.get("mode", "auto")
        self.expected_train_sde_mu_hole_mode = str(
            self.inference_opt.get(
                "expected_train_sde_mu_hole_mode",
                opt.get("train", {}).get("sde_mu_hole_mode", ""),
            )
        ).lower()
        self.force_legacy_reverse = bool(self.inference_opt.get("force_legacy_reverse", False)) or (
            str(self.inference_mode).lower() == "legacy_reverse"
        )
        self.condition_known_source = str(
            self.inference_opt.get("condition_known_source", "lut")
        ).lower()
        self.structure_source = str(
            self.inference_opt.get("structure_source", "lut")
        ).lower()
        self.safe_prior_min_reliability = float(
            self.inference_opt.get("safe_prior_min_reliability", 0.0) or 0.0
        )
        self.safe_prior_confidence_power = float(
            self.inference_opt.get("safe_prior_confidence_power", 1.0) or 1.0
        )
        if self.safe_prior_confidence_power <= 0:
            logger.warning(
                "Invalid inference.safe_prior_confidence_power=%s; falling back to 1.0",
                self.safe_prior_confidence_power,
            )
            self.safe_prior_confidence_power = 1.0
        self.confidence_debug_threshold = float(
            self.inference_opt.get("confidence_debug_threshold", 0.4) or 0.4
        )

        self.lut_processor = None
        lut_path = self.dataset_opt.get("lut_path")
        if lut_path:
            self.lut_processor = LUTProcessor(lut_path)

        self.color_prior_generator = None
        if lut_path:
            self.color_prior_generator = ColorPriorGenerator(
                lut_path=lut_path,
                alpha=self.dataset_opt.get("lut_alpha", 0.7),
                beta=self.dataset_opt.get("lut_beta", 0.3),
                inpaint_method=self.dataset_opt.get("lut_inpaint_method", "telea"),
                inpaint_mask_dilate=self.dataset_opt.get(
                    "prior_inpaint_mask_dilate",
                    self.dataset_opt.get("inpaint_mask_dilate", 3),
                ),
                lut_delta_gain=self.lut_delta_gain,
            )

        mu_opt = opt.get("mu_denoiser", {})
        self.use_mu_denoiser = bool(mu_opt.get("enabled", False) and HAS_MU_DENOISER)
        self.mu_denoiser_has_weights = False
        self.mu_denoiser = None
        self.mu_denoiser_trainer = None
        if self.use_mu_denoiser:
            self.mu_denoiser = MuDenoiser(
                in_nc=mu_opt.get("in_nc", 5),
                dim=mu_opt.get("dim", 32),
                num_blocks=mu_opt.get("num_blocks", 2),
                num_heads=mu_opt.get("num_heads", 4),
                predict_residual=mu_opt.get("predict_residual", True),
            ).to(self.device)
            self.mu_denoiser_trainer = MuDenoiserTrainer(
                self.mu_denoiser, blind_ratio=mu_opt.get("blind_ratio", 0.1)
            )

        self.load()
        self._log_route_config()
        self.output = None
        self.raw_output = None
        self.debug_outputs: Dict[str, torch.Tensor] = {}

    def _unwrap_model(self, network):
        return network.module if isinstance(network, DataParallel) else network

    def _log_route_config(self):
        """Log the effective inference route before running samples.

        The no-retrain ablation keeps the wrapper class for checkpoint
        compatibility, but disables BrushNet/MGLC/Mu-Denoiser through config.
        This log line is the quickest way to verify that the current run is
        actually on the intended original StrDiffusion trunk.
        """
        module = self._unwrap_model(self.model)
        brushnet_opt = self.opt.get("brushnet", {})
        texture_core_opt = self.opt.get("texture_core", {})
        mu_opt = self.opt.get("mu_denoiser", {})
        inference_opt = self.inference_opt
        network_name = self.opt.get("network_G", {}).get("which_model_G")
        brushnet_runtime = bool(getattr(module, "brushnet_enabled", False))
        texture_core_runtime = bool(getattr(module, "texture_core_enabled", False))
        no_extra_route = (
            network_name == "ConditionalUNetWithBrushNet"
            and not brushnet_runtime
            and not texture_core_runtime
            and not self.use_mu_denoiser
        )
        pure_no_extra_route = (
            no_extra_route
            and not self.discriminator_guidance
            and self.deterministic_reverse
        )

        logger.info(
            "[RouteCheck] network_G=%s model_class=%s no_extra_route=%s pure_no_extra_route=%s",
            network_name,
            module.__class__.__name__,
            no_extra_route,
            pure_no_extra_route,
        )
        logger.info(
            "[RouteCheck] brushnet.enabled(config/runtime)=%s/%s "
            "brushnet.feature_scale(runtime)=%s brushnet.use_spatial_gate(runtime)=%s "
            "texture_core.enabled(config/runtime)=%s/%s "
            "mu_denoiser.enabled(config/available/runtime/has_weights)=%s/%s/%s/%s "
            "restore_S_guidance=%s inference.mode=%s sde_mu_hole_mode=%s "
            "expected_train_sde_mu_hole_mode=%s "
            "save_states=%s save_intermediates=%s discriminator_guidance=%s "
            "deterministic_reverse=%s known_area_projection=%s "
            "force_legacy_reverse=%s condition_known_source=%s structure_source=%s "
            "safe_prior_min_reliability=%.3f safe_prior_confidence_power=%.3f "
            "lut_delta_gain=%.3f confidence_debug_threshold=%.3f",
            bool(brushnet_opt.get("enabled", False)),
            brushnet_runtime,
            getattr(module, "brushnet_feature_scale", None),
            getattr(module, "brushnet_use_spatial_gate", None),
            bool(texture_core_opt.get("enabled", False)),
            texture_core_runtime,
            bool(mu_opt.get("enabled", False)),
            HAS_MU_DENOISER,
            self.use_mu_denoiser,
            getattr(self, "mu_denoiser_has_weights", False),
            self.restore_s_guidance,
            self.inference_mode,
            inference_opt.get("sde_mu_hole_mode", "known_only"),
            self.expected_train_sde_mu_hole_mode or "unset",
            bool(inference_opt.get("save_states", False)),
            self.save_intermediates,
            self.discriminator_guidance,
            self.deterministic_reverse,
            self.known_area_projection,
            self.force_legacy_reverse,
            self.condition_known_source,
            self.structure_source,
            self.safe_prior_min_reliability,
            self.safe_prior_confidence_power,
            self.lut_delta_gain,
            self.confidence_debug_threshold,
        )
        if (
            self.expected_train_sde_mu_hole_mode
            and self.expected_train_sde_mu_hole_mode
            != str(inference_opt.get("sde_mu_hole_mode", "known_only")).lower()
        ):
            logger.warning(
                "[RouteCheck] train/inference sde_mu_hole_mode mismatch: "
                "expected_train=%s inference=%s. "
                "This is a real distribution shift for checkpoints trained with a non-black hole mu.",
                self.expected_train_sde_mu_hole_mode,
                inference_opt.get("sde_mu_hole_mode", "known_only"),
            )
        logger.info(
            "[RouteCheck] pretrain_model_G=%s pretrain_model_Gs=%s pretrain_model_D=%s strict_load=%s",
            self.opt["path"].get("pretrain_model_G"),
            self.opt["path"].get("pretrain_model_Gs"),
            self.opt["path"].get("pretrain_model_D"),
            self.opt["path"].get("strict_load", True),
        )
        if no_extra_route:
            logger.info(
                "[NoExtraRoute] BrushNet/MGLC/Mu-Denoiser are bypassed; "
                "the wrapper is kept only for current-checkpoint key compatibility. "
                "restore_S_guidance stays enabled when configured, matching the original StrDiffusion structure path."
            )
            if inference_opt.get("sde_mu_hole_mode", "known_only") != "known_only":
                logger.warning(
                    "[NoExtraRoute] inference.sde_mu_hole_mode=%s keeps non-zero content in the hole. "
                    "For the original StrDiffusion inpainting start, use known_only so the clean x_init hole is black "
                    "before noise_state adds small Gaussian noise.",
                    inference_opt.get("sde_mu_hole_mode", "known_only"),
                )
            if self.discriminator_guidance or not self.deterministic_reverse:
                logger.warning(
                    "[NoExtraRoute] sampler is still not isolated: discriminator_guidance=%s, "
                    "deterministic_reverse=%s. For the next white-hole diagnosis, set "
                    "inference.discriminator_guidance.enabled=false and inference.deterministic_reverse=true.",
                    self.discriminator_guidance,
                    self.deterministic_reverse,
                )

    def feed_data(
        self,
        state,
        LQ,
        GT,
        mask,
        S_sde,
        S_GT,
        S_LQ,
        color_prior=None,
        confidence=None,
        conf_lut=None,
        original_degraded=None,
        mask_hole=None,
        sample_name=None,
    ):
        """Load one sample while keeping the original feed_data signature."""
        self.state = state.to(self.device) if state is not None else None
        self.condition = LQ.to(self.device) if LQ is not None else None
        self.state_0 = GT.to(self.device) if GT is not None else None
        self.mask = mask.to(self.device).float() if mask is not None else None
        self.mask_hole = (
            mask_hole.to(self.device).float()
            if mask_hole is not None
            else (1.0 - self.mask if self.mask is not None else None)
        )
        if self.mask is not None:
            self.mask = self.mask.clamp(0.0, 1.0)
        if self.mask_hole is not None:
            self.mask_hole = self.mask_hole.clamp(0.0, 1.0)
        if self.mask is not None and self.mask_hole is not None:
            _validate_mask_pair(self.mask, self.mask_hole, "feed_data")
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device) if torch.is_tensor(S_GT) else S_GT
        self.S_LQ = S_LQ.to(self.device) if torch.is_tensor(S_LQ) else S_LQ
        self.original_degraded = (
            original_degraded.to(self.device)
            if original_degraded is not None
            else self.condition
        )
        self.color_prior = color_prior.to(self.device) if color_prior is not None else None
        self.confidence = confidence.to(self.device) if confidence is not None else None
        self.conf_lut = conf_lut.to(self.device) if conf_lut is not None else None
        self.sample_name = sample_name or "sample"

    def _denoise_image(self, image: torch.Tensor, mask_known: Optional[torch.Tensor] = None) -> torch.Tensor:
        """A lightweight edge-preserving smoothing used before LUT processing.

        When mask_known is provided (1=known, 0=hole), normalized convolution
        excludes white/black hole pixels from the smoothing support and fills
        hole pixels from known-region neighborhoods for structure guidance.
        """
        sigma_spatial = 2.0
        kernel_size = 5
        padding = kernel_size // 2
        coords = torch.arange(kernel_size, dtype=image.dtype, device=image.device)
        coords = coords - padding
        gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma_spatial ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)

        if mask_known is not None:
            mask_known = mask_known.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
            if mask_known.shape[1] != 1:
                mask_known = mask_known[:, :1]
            denom = F.conv2d(mask_known, gauss_2d, padding=padding).clamp_min(1e-6)
            known_count = mask_known.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
            known_mean = (image * mask_known).sum(dim=(2, 3), keepdim=True) / known_count
            smoothed_channels = []
            for channel_idx in range(image.shape[1]):
                channel = image[:, channel_idx : channel_idx + 1]
                smoothed_c = F.conv2d(channel * mask_known, gauss_2d, padding=padding) / denom
                smoothed_c = torch.where(
                    denom > 1e-5, smoothed_c, known_mean[:, channel_idx : channel_idx + 1]
                )
                smoothed_channels.append(smoothed_c)
            smoothed = torch.cat(smoothed_channels, dim=1)
            edge_input = image * mask_known + smoothed * (1 - mask_known)
        else:
            smoothed_channels = []
            for channel_idx in range(image.shape[1]):
                channel = image[:, channel_idx : channel_idx + 1]
                smoothed_channels.append(F.conv2d(channel, gauss_2d, padding=padding))
            smoothed = torch.cat(smoothed_channels, dim=1)
            edge_input = image

        gray = 0.299 * edge_input[:, 0:1] + 0.587 * edge_input[:, 1:2] + 0.114 * edge_input[:, 2:3]
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=image.dtype,
            device=image.device,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        edge_weight = 1 - torch.exp(-grad_mag / 0.1)
        denoised = edge_weight * edge_input + (1 - edge_weight) * smoothed
        return torch.clamp(denoised, 0.0, 1.0)

    def _guided_smooth(self, image: torch.Tensor, guide: torch.Tensor, radius: int = 5):
        """Apply bilateral smoothing channel-by-channel."""
        results = []
        for batch_idx in range(image.shape[0]):
            img_np = image[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
            guide_np = guide[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            guide_uint8 = (np.clip(guide_np, 0, 1) * 255).astype(np.uint8)
            _ = guide_uint8  # guide is kept for future extension and documentation parity.

            smoothed = np.zeros_like(img_uint8, dtype=np.float32)
            diameter = radius * 2 + 1
            for channel_idx in range(3):
                filtered = cv2.bilateralFilter(
                    img_uint8[:, :, channel_idx],
                    d=diameter,
                    sigmaColor=50,
                    sigmaSpace=radius,
                )
                smoothed[:, :, channel_idx] = filtered.astype(np.float32) / 255.0
            results.append(
                torch.from_numpy(smoothed).permute(2, 0, 1).to(image.device)
            )
        return torch.stack(results, dim=0)

    def _build_structure_targets(self, image: torch.Tensor):
        """Create grayscale and edge maps for official Gs/S_sde compatibility."""
        gray_list = []
        edge_list = []
        for batch_idx in range(image.shape[0]):
            rgb = image[batch_idx].permute(1, 2, 0).detach().cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, 50, 150)
            gray_list.append(torch.from_numpy(gray.astype(np.float32) / 255.0)[None, ...])
            edge_list.append(torch.from_numpy(edge.astype(np.float32) / 255.0)[None, ...])

        structure_gray = torch.stack(gray_list, dim=0).to(self.device)
        structure_edge = torch.stack(edge_list, dim=0).to(self.device)
        return structure_gray, structure_edge

    def _build_safe_brushnet_prior(self, color_prior, confidence, condition_lut, mask_known, mask_hole):
        """Gate BrushNet prior by confidence only.

        The original gap-based reliability gate was too aggressive: it replaced
        the cv2.inpaint color_prior with lut_transformed whenever the two
        differed by more than gap_scale (~0.15), which is a normal and expected
        difference for inpainted texture regions.  This destroyed the inpaint
        content that BrushNet needs for color guidance.

        Now we only use confidence as the gate weight in hole pixels.
        Known pixels always use lut_transformed (target domain).
        """
        color_prior = color_prior.to(dtype=condition_lut.dtype, device=condition_lut.device)
        mask_known = mask_known.to(dtype=condition_lut.dtype, device=condition_lut.device)
        mask_hole = mask_hole.to(dtype=condition_lut.dtype, device=condition_lut.device)
        if confidence is None:
            prior_reliability = torch.ones_like(mask_known)
        else:
            prior_reliability = confidence.to(dtype=condition_lut.dtype, device=condition_lut.device).clamp(0.0, 1.0)
        if self.safe_prior_confidence_power != 1.0:
            prior_reliability = prior_reliability.pow(self.safe_prior_confidence_power)
        if self.safe_prior_min_reliability > 0.0:
            floor = torch.full_like(
                prior_reliability,
                max(0.0, min(1.0, self.safe_prior_min_reliability)),
            )
            prior_reliability = torch.maximum(prior_reliability, floor)
        prior_reliability = prior_reliability.clamp(0.0, 1.0)

        # In hole pixels: blend color_prior (inpaint result) by confidence.
        # Low-confidence holes get more lut_transformed, high-confidence get more color_prior.
        safe_hole_prior = prior_reliability * color_prior + (1 - prior_reliability) * condition_lut
        safe_prior = condition_lut * mask_known + safe_hole_prior * mask_hole
        safe_confidence = torch.ones_like(mask_known) * mask_known + prior_reliability * mask_hole
        return safe_prior, safe_confidence, prior_reliability

    def _prepare_brushnet_inputs(self):
        """Mirror the training-side color path as closely as possible."""
        mask_known = self.mask
        mask_hole = self.mask_hole
        _validate_mask_pair(mask_known, mask_hole, "_prepare_brushnet_inputs")
        degraded = self.original_degraded

        denoised_original = self._denoise_image(degraded, mask_known=mask_known)

        color_prior = self.color_prior
        confidence = self.confidence
        prior_debug = {}
        if color_prior is None or confidence is None:
            if self.color_prior_generator is None:
                raise RuntimeError("缺少 ColorPriorGenerator，无法自动生成 color_prior/confidence。")
            generated = self.color_prior_generator.generate_tensor(
                degraded,
                mask_hole,
                device=self.device,
                method=self.prior_method,
                debug=self.save_intermediates,
            )
            if self.save_intermediates:
                generated_prior, generated_confidence, prior_debug = generated
            else:
                generated_prior, generated_confidence = generated
            color_prior = generated_prior if color_prior is None else color_prior
            confidence = generated_confidence if confidence is None else confidence

        lut_transformed = denoised_original
        if self.lut_processor is not None:
            lut_transformed, lut_confidence = self.lut_processor.apply_to_tensor(
                denoised_original
            )
            if self.dataset_opt.get("lut_smooth_radius", 0) > 0:
                lut_transformed = self._guided_smooth(
                    lut_transformed,
                    guide=denoised_original,
                    radius=self.dataset_opt.get("lut_smooth_radius", 5),
                )
            # Interpret lut_strength as max global blend strength. Do not let
            # values >1 saturate the whole image to a full LUT jump.
            strength = float(max(0.0, min(1.0, self.dataset_opt.get("lut_strength", 1.0))))
            effective_weight = torch.clamp(lut_confidence, 0.0, 1.0) * strength
            lut_delta = lut_transformed - denoised_original
            lut_transformed = torch.clamp(
                denoised_original + lut_delta * effective_weight * self.lut_delta_gain,
                0.0,
                1.0,
            )

        # Align known pixels to CondLUT, then softly gate the hole prior by
        # target-domain consistency.  The raw inpaint prior can be very bright
        # on large white-filled holes; if fed directly to BrushNet it pushes the
        # score toward white even though SDE mu is already correct.
        raw_color_prior = lut_transformed * mask_known + color_prior * mask_hole
        raw_confidence = torch.ones_like(mask_known) * mask_known + confidence * mask_hole
        color_prior, confidence, prior_reliability = self._build_safe_brushnet_prior(
            raw_color_prior, raw_confidence, lut_transformed, mask_known, mask_hole
        )

        if self.save_intermediates:
            mask_hole_3c = mask_hole.expand(-1, color_prior.shape[1], -1, -1)
            hole_denom = mask_hole_3c.sum().clamp_min(1.0)
            raw_cp_hole_mean = float((raw_color_prior * mask_hole_3c).sum().item() / hole_denom.item())
            cp_hole_mean = float((color_prior * mask_hole_3c).sum().item() / hole_denom.item())
            cp_hole_std = float((((color_prior - cp_hole_mean) ** 2) * mask_hole_3c).sum().div(hole_denom).sqrt().item())
            white_ratio = float((((color_prior > 0.95).all(dim=1, keepdim=True).float() * mask_hole).sum() / mask_hole.sum().clamp_min(1.0)).item())
            conf_hole_mean = float((confidence * mask_hole).sum().div(mask_hole.sum().clamp_min(1.0)).item())
            reliability_hole_mean = float((prior_reliability * mask_hole).sum().div(mask_hole.sum().clamp_min(1.0)).item())
            reliability_stats = self._masked_scalar_stats(prior_reliability, mask_hole)
            low_conf = (prior_reliability < self.confidence_debug_threshold).float() * mask_hole
            low_conf_ratio = float(low_conf.sum().div(mask_hole.sum().clamp_min(1.0)).item())
            logger.info(
                "[ColorPrior Debug] raw_hole_mean=%.6f hole_mean=%.6f hole_std=%.6f hole_white_ratio=%.6f confidence_hole_mean=%.6f reliability_hole_mean=%.6f",
                raw_cp_hole_mean,
                cp_hole_mean,
                cp_hole_std,
                white_ratio,
                conf_hole_mean,
                reliability_hole_mean,
            )
            logger.info(
                "[Confidence Debug] reliability(min=%.4f,p10=%.4f,p50=%.4f,p90=%.4f,max=%.4f) "
                "low_ratio_lt%.2f=%.4f safe_prior_min=%.3f conf_power=%.3f",
                reliability_stats.get("min", 0.0),
                reliability_stats.get("p10", 0.0),
                reliability_stats.get("p50", 0.0),
                reliability_stats.get("p90", 0.0),
                reliability_stats.get("max", 0.0),
                self.confidence_debug_threshold,
                low_conf_ratio,
                self.safe_prior_min_reliability,
                self.safe_prior_confidence_power,
            )

        if self.use_mu_denoiser and getattr(self, "mu_denoiser_has_weights", False):
            mu_clean_lut = self.mu_denoiser_trainer.inference(
                lut_transformed,
                mask_known,
                confidence,
            ).clamp(0.0, 1.0)
        else:
            mu_clean_lut = lut_transformed

        known_source = mu_clean_lut
        if self.condition_known_source in {"gt", "gt_if_available"}:
            if self.state_0 is not None:
                known_source = self.state_0
            elif self.condition_known_source == "gt":
                raise RuntimeError("inference.condition_known_source=gt but this sample has no GT tensor.")
        elif self.condition_known_source == "degraded":
            known_source = degraded
        elif self.condition_known_source in {"lut", "condition_lut", "target_lut"}:
            known_source = mu_clean_lut
        else:
            raise ValueError(
                f"Unsupported inference.condition_known_source={self.condition_known_source!r}; "
                "expected lut|degraded|gt|gt_if_available"
            )

        # SDE mu construction must match training.  Do not use raw degraded input.
        # known_only preserves the original inpainting semantics; condition_lut anchors
        # holes with the target-domain LUT estimate; safe_prior uses the confidence-gated
        # BrushNet prior and should be treated as an ablation.
        mu_hole_mode = self.inference_opt.get("sde_mu_hole_mode", "known_only")
        if mu_hole_mode == "known_only":
            condition_mu = known_source * mask_known
        elif mu_hole_mode == "condition_lut":
            condition_mu = known_source * mask_known + lut_transformed * mask_hole
        elif mu_hole_mode == "safe_prior":
            condition_mu = known_source * mask_known + color_prior * mask_hole
        else:
            raise ValueError(
                f"Unsupported inference.sde_mu_hole_mode={mu_hole_mode!r}; "
                "expected known_only|condition_lut|safe_prior"
            )

        return {
            "denoised_original": denoised_original,
            "lut_transformed": lut_transformed,
            "condition_lut": lut_transformed,
            "condition_mu": condition_mu,
            "color_prior": color_prior,
            "color_prior_raw": raw_color_prior,
            "confidence": confidence,
            "prior_reliability": prior_reliability,
            "mu_clean_lut": mu_clean_lut,
            "known_source": known_source,
            "mu_clean": condition_mu,
            "prior_debug": prior_debug,
        }

    def _build_training_target_like(self) -> Optional[torch.Tensor]:
        """Build a visualization target that follows the training GT construction rule."""
        if self.color_prior_generator is None or self.state_0 is None or self.mask_hole is None:
            return None

        reference = self.state_0[0].detach().float().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
        reference = (reference * 255.0).round().astype(np.uint8)
        mask_hole = self.mask_hole[0, 0].detach().float().cpu().clamp(0.0, 1.0).numpy()
        mask_hole = (mask_hole * 255.0).round().astype(np.uint8)
        target = self.color_prior_generator.build_target(
            reference,
            mask_hole,
            mode=self.gt_mode,
            feather_radius=7,
        )
        target = torch.from_numpy(target.astype(np.float32) / 255.0)
        target = target.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return target

    def _save_intermediate_outputs(self, save_dir: str):
        """Save debugging tensors to image files when enabled."""
        if not self.save_intermediates:
            return

        os.makedirs(save_dir, exist_ok=True)
        for name, tensor in self.debug_outputs.items():
            if tensor is None:
                continue
            if tensor.dim() == 4:
                tensor = tensor[0]
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                image = util.tensor2img(tensor)
            else:
                if tensor.dim() == 3:
                    tensor = tensor[0]
                gray = tensor.detach().float().cpu().clamp(0, 1).numpy()
                image = (gray * 255.0).round().astype(np.uint8)
            util.save_img(image, os.path.join(save_dir, f"{name}.png"))

    def test(
        self,
        sde=None,
        save_states=False,
        save_dir="save_dir",
        GT=None,
        mask=None,
        S_sde=None,
        S_GT=None,
        S_LQ=None,
        dis=None,
    ):
        """Run one inference pass while keeping the official call shape."""
        self.model.eval()
        self.models.eval()
        self.dis.eval()

        with torch.no_grad():
            enhanced_mode = (
                self.inference_mode == "brushnet_final"
                or self.opt["network_G"]["which_model_G"] == "ConditionalUNetWithBrushNet"
            )
            if enhanced_mode:
                _validate_mask_pair(self.mask, self.mask_hole, "test/enhanced")
                prepared = self._prepare_brushnet_inputs()
                self.color_prior = prepared["color_prior"]
                self.confidence = prepared["confidence"]
                mu_clean = prepared["mu_clean"]
                prior_debug = prepared.get("prior_debug", {})
                target_like = self._build_training_target_like()
                prepared["training_target_like"] = target_like
                self.condition = prepared["condition_mu"]
                sde.set_mu(self.condition)
                x_init = self.condition
                self.state = sde.noise_state(x_init)
                x_init_hole = self._masked_rgb_stats(x_init, self.mask_hole)
                noisy_start_hole = self._masked_rgb_stats(self.state, self.mask_hole)
                logger.info(
                    "[RouteCheck] sample=%s x_init=condition_mu sde_mu_hole_mode=%s "
                    "restore_S_guidance=%s brushnet_runtime=%s texture_core_runtime=%s "
                    "mu_denoiser_runtime=%s "
                    "x_init_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
                    "noisy_start_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f)",
                    self.sample_name,
                    self.inference_opt.get("sde_mu_hole_mode", "known_only"),
                    self.restore_s_guidance,
                    bool(getattr(self._unwrap_model(self.model), "brushnet_enabled", False)),
                    bool(getattr(self._unwrap_model(self.model), "texture_core_enabled", False)),
                    self.use_mu_denoiser,
                    x_init_hole.get("mean", 0.0),
                    x_init_hole.get("min", 0.0),
                    x_init_hole.get("max", 0.0),
                    x_init_hole.get("white_ratio", 0.0),
                    noisy_start_hole.get("mean", 0.0),
                    noisy_start_hole.get("min", 0.0),
                    noisy_start_hole.get("max", 0.0),
                    noisy_start_hole.get("white_ratio", 0.0),
                )

                # Default mural inference uses the target-domain LUT estimate for
                # structure.  For baseline parity diagnostics we can intentionally
                # switch to GT-known semantics when GT is available, matching the
                # original StrDiffusion test flow more closely.
                if self.structure_source in {"gt", "gt_if_available"}:
                    if self.state_0 is not None:
                        structure_input = self.state_0
                        resolved_structure_source = "gt"
                    elif self.structure_source == "gt":
                        raise RuntimeError("inference.structure_source=gt but this sample has no GT tensor.")
                    else:
                        structure_input = prepared["lut_transformed"]
                        resolved_structure_source = "lut_fallback"
                elif self.structure_source == "degraded":
                    structure_input = self.original_degraded
                    resolved_structure_source = "degraded"
                elif self.structure_source in {"condition_mu", "mu"}:
                    structure_input = self.condition
                    resolved_structure_source = "condition_mu"
                elif self.structure_source in {"lut", "condition_lut", "target_lut"}:
                    structure_input = prepared["lut_transformed"]
                    resolved_structure_source = "lut"
                else:
                    raise ValueError(
                        f"Unsupported inference.structure_source={self.structure_source!r}; "
                        "expected lut|degraded|condition_mu|gt|gt_if_available"
                    )
                logger.info(
                    "[StructureRoute] sample=%s structure_source=%s resolved=%s has_gt=%s",
                    self.sample_name,
                    self.structure_source,
                    resolved_structure_source,
                    self.state_0 is not None,
                )
                structure_gray, structure_edge = self._build_structure_targets(
                    structure_input
                )
                structure_state = None
                if S_sde is not None:
                    S_sde.set_mu(structure_edge * self.mask)
                    structure_state = S_sde.noise_state(structure_edge * self.mask)

                if self.force_legacy_reverse:
                    legacy_known = (
                        GT.to(self.device)
                        if torch.is_tensor(GT)
                        else prepared["known_source"]
                    )
                    logger.info(
                        "[LegacyReverseRoute] sample=%s using original reverse_sde loop "
                        "(enhanced_inference=false) with wrapper=%s; "
                        "BrushNet/MGLC/Mu runtime flags remain as logged above.",
                        self.sample_name,
                        self.opt["network_G"]["which_model_G"],
                    )
                    pred_full = sde.reverse_sde(
                        self.state,
                        save_states=save_states,
                        save_dir=save_dir,
                        GT=legacy_known,
                        mask=self.mask,
                        S_sde=S_sde,
                        S_GT=structure_gray,
                        S_LQ=structure_state,
                        dis=dis,
                        S_LQs=structure_edge,
                    )
                else:
                    pred_full = sde.reverse_sde(
                        self.state,
                        save_states=save_states,
                        save_dir=save_dir,
                        GT=GT,
                        mask=self.mask,
                        S_sde=S_sde,
                        S_GT=structure_gray,
                        S_LQ=structure_state,
                        dis=dis,
                        S_LQs=structure_edge,
                        enhanced_inference=True,
                        gt_mode=self.gt_mode,
                        mask_hole=self.mask_hole,
                        color_prior=self.color_prior,
                        confidence=self.confidence,
                        restore_S_guidance=self.restore_s_guidance,
                        discriminator_guidance=self.discriminator_guidance,
                        deterministic_reverse=self.deterministic_reverse,
                        known_area_projection=self.known_area_projection,
                    )
                self.raw_output = pred_full
                if self.gt_mode == "partial":
                    known_source = self.original_degraded
                else:
                    # full mode still protects known pixels, but the protected
                    # value is target-domain CondLUT, not raw degraded input.
                    known_source = prepared["lut_transformed"]
                compose_alpha = self._build_composite_alpha(
                    self.mask_hole,
                    source=self.original_degraded,
                )
                if self.force_legacy_reverse:
                    # The original reverse_sde branch already returns
                    # known_source * mask_known + predicted_hole * mask_hole.
                    # Do not apply the mural feather/white-guard composite here;
                    # otherwise this parity run would no longer be the original
                    # StrDiffusion sampler.
                    self.output = pred_full
                else:
                    self.output = known_source * (1 - compose_alpha) + pred_full * compose_alpha

                self._log_inference_debug_stats(pred_full, self.output, prepared)

                self.debug_outputs = {
                    "original_degraded": self.original_degraded,
                    "mask_hole": self.mask_hole,
                    "mask_known": self.mask,
                    "color_prior": self.color_prior,
                    "color_prior_raw": prepared.get("color_prior_raw"),
                    "prior_reliability": prepared.get("prior_reliability"),
                    "color_prior_lut": prior_debug.get("color_prior_lut"),
                    "color_prior_inpainted": prior_debug.get("color_prior_inpainted"),
                    "image_for_lut": prior_debug.get("image_for_lut"),
                    "conf_lut_prior": prior_debug.get("conf_lut"),
                    "conf_inpaint_prior": prior_debug.get("conf_inpaint"),
                    "prior_inpaint_mask": prior_debug.get("inpaint_mask"),
                    "training_target_like": target_like,
                    "confidence": self.confidence,
                    "denoised_original": prepared["denoised_original"],
                    "condition_lut": prepared["condition_lut"],
                    "lut_transformed": prepared["lut_transformed"],
                    "condition_mu": self.condition,
                    "mu_clean_lut": prepared["mu_clean_lut"],
                    "known_source": prepared.get("known_source"),
                    "mu_clean": mu_clean,
                    "x_init": x_init,
                    "x_start_noisy": self.state,
                    "structure_source_image": structure_input,
                    "structure_gray": structure_gray,
                    "structure_edge": structure_edge,
                    "compose_alpha": compose_alpha,
                    "raw_pred": self.raw_output,
                    "final": self.output,
                }
                self._save_intermediate_outputs(save_dir)
            else:
                sde.set_mu(self.condition)
                if S_sde is not None:
                    S_sde.set_mu(self.S_LQ)
                self.output = sde.reverse_sde(
                    self.state,
                    save_states=save_states,
                    save_dir=save_dir,
                    GT=GT,
                    mask=mask,
                    S_sde=S_sde,
                    S_GT=S_GT,
                    S_LQ=S_LQ,
                    dis=dis,
                    S_LQs=self.S_LQ,
                )
                self.raw_output = self.output

        self.model.train()
        self.models.train()
        self.dis.train()

    def get_current_log(self):
        return OrderedDict()

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        out_dict["RawOutput"] = self.raw_output.detach()[0].float().cpu()
        if need_GT and self.state_0 is not None:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def load_network_from_state(self, state_dict, network, strict=True):
        if isinstance(network, DataParallel):
            network = network.module
        incompatible = network.load_state_dict(state_dict, strict=strict)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        total = len(network.state_dict())
        loaded = max(0, total - len(missing))
        logger.info(
            "[LoadCheck] loaded %d/%d tensors into %s, missing=%d, unexpected=%d",
            loaded,
            total,
            network.__class__.__name__,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("[LoadCheck] missing keys sample: %s", missing[:20])
        if unexpected:
            logger.warning("[LoadCheck] unexpected keys sample: %s", unexpected[:20])

    def _build_composite_alpha(
        self,
        mask_hole: torch.Tensor,
        source: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference-only soft compose mask to remove white/gray mask fringes.

        mask_hole semantics are unchanged (1=hole). We optionally dilate the
        hole before final compositing and feather the boundary, so placeholder
        pixels just outside the annotated mask are not preserved as hard white
        rims. This does not change the SDE trajectory or network structure.
        """
        alpha = mask_hole.float()
        if bool(self.inference_opt.get("compose_white_guard", False)):
            # Some inference masks are slightly tighter than the actually
            # white-filled damaged pixels.  Those pixels are then treated as
            # "known" and survive the final composite as white rims.  Build a
            # conservative auxiliary alpha from near-white source pixels only in
            # a narrow neighborhood of the hole mask.
            guard_dilate = int(self.inference_opt.get("compose_white_guard_dilate", 8) or 0)
            threshold = float(self.inference_opt.get("compose_white_threshold", 0.965))
            source_tensor = source if source is not None else self.original_degraded
            if guard_dilate > 0 and source_tensor is not None:
                k = 2 * guard_dilate + 1
                near_hole = F.max_pool2d(alpha, kernel_size=k, stride=1, padding=guard_dilate)
                white_like = (source_tensor.detach().float() > threshold).all(dim=1, keepdim=True).float()
                alpha = torch.maximum(alpha, white_like * near_hole)
        dilate = int(self.inference_opt.get("compose_mask_dilate", 0) or 0)
        feather = int(self.inference_opt.get("compose_feather", 0) or 0)
        if dilate > 0:
            k = 2 * dilate + 1
            alpha = F.max_pool2d(alpha, kernel_size=k, stride=1, padding=dilate)
        if feather > 0:
            k = 2 * feather + 1
            alpha = F.avg_pool2d(alpha, kernel_size=k, stride=1, padding=feather)
        return alpha.clamp(0.0, 1.0)

    def _masked_rgb_stats(self, tensor, mask):
        if tensor is None or mask is None:
            return {}
        rgb = tensor.detach().float()
        mask = mask.detach().float()
        if mask.shape[1] != rgb.shape[1]:
            mask = mask.expand(-1, rgb.shape[1], -1, -1)
        denom = mask.sum().clamp_min(1.0)
        masked = rgb * mask
        mean = masked.sum() / denom
        valid = mask > 0.5
        masked_values = rgb[valid]
        if masked_values.numel() > 0:
            min_val = masked_values.min()
            max_val = masked_values.max()
        else:
            min_val = rgb.new_tensor(0.0)
            max_val = rgb.new_tensor(0.0)
        white = ((rgb > 0.95).all(dim=1, keepdim=True).float() * mask[:, :1]).sum()
        white_denom = mask[:, :1].sum().clamp_min(1.0)
        return {
            "mean": float(mean.item()),
            "min": float(min_val.item()),
            "max": float(max_val.item()),
            "white_ratio": float((white / white_denom).item()),
        }

    def _masked_l1(self, lhs, rhs, mask):
        if lhs is None or rhs is None or mask is None:
            return None
        lhs = lhs.detach().float()
        rhs = rhs.detach().float().to(device=lhs.device)
        mask = mask.detach().float().to(device=lhs.device)
        if mask.shape[1] != lhs.shape[1]:
            mask = mask.expand(-1, lhs.shape[1], -1, -1)
        denom = mask.sum().clamp_min(1.0)
        return float(((lhs - rhs).abs() * mask).sum().div(denom).item())

    def _masked_scalar_stats(self, tensor, mask):
        if tensor is None or mask is None:
            return {}
        values = tensor.detach().float()
        mask = mask.detach().float().to(device=values.device)
        if mask.shape[1] != values.shape[1]:
            mask = mask.expand(-1, values.shape[1], -1, -1)
        selected = values[mask > 0.5]
        if selected.numel() == 0:
            return {}
        quantiles = torch.quantile(
            selected,
            torch.tensor([0.1, 0.5, 0.9], device=selected.device),
        )
        return {
            "mean": float(selected.mean().item()),
            "min": float(selected.min().item()),
            "p10": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p90": float(quantiles[2].item()),
            "max": float(selected.max().item()),
        }

    def _log_inference_debug_stats(self, pred_full, final, prepared=None):
        if self.mask_hole is None:
            return
        raw_hole = self._masked_rgb_stats(pred_full, self.mask_hole)
        final_hole = self._masked_rgb_stats(final, self.mask_hole)
        cond_known = self._masked_rgb_stats(self.condition, self.mask)
        cond_hole = self._masked_rgb_stats(self.condition, self.mask_hole)
        prior_hole = self._masked_rgb_stats(self.color_prior, self.mask_hole)
        
        lut_hole = {}
        if prepared is not None and "lut_transformed" in prepared:
            lut_hole = self._masked_rgb_stats(prepared["lut_transformed"], self.mask_hole)

        logger.info(
            "[Inference Debug] gt_mode=%s deterministic_reverse=%s "
            "raw_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "final_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "cond_known(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "cond_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "prior_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "lut_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f)",
            self.gt_mode,
            self.deterministic_reverse,
            raw_hole.get("mean", 0.0),
            raw_hole.get("min", 0.0),
            raw_hole.get("max", 0.0),
            raw_hole.get("white_ratio", 0.0),
            final_hole.get("mean", 0.0),
            final_hole.get("min", 0.0),
            final_hole.get("max", 0.0),
            final_hole.get("white_ratio", 0.0),
            cond_known.get("mean", 0.0),
            cond_known.get("min", 0.0),
            cond_known.get("max", 0.0),
            cond_known.get("white_ratio", 0.0),
            cond_hole.get("mean", 0.0),
            cond_hole.get("min", 0.0),
            cond_hole.get("max", 0.0),
            cond_hole.get("white_ratio", 0.0),
            prior_hole.get("mean", 0.0),
            prior_hole.get("min", 0.0),
            prior_hole.get("max", 0.0),
            prior_hole.get("white_ratio", 0.0),
            lut_hole.get("mean", 0.0),
            lut_hole.get("min", 0.0),
            lut_hole.get("max", 0.0),
            lut_hole.get("white_ratio", 0.0),
        )
        if prepared is not None:
            lut = prepared.get("lut_transformed")
            denoised = prepared.get("denoised_original")
            raw_prior = prepared.get("color_prior_raw")
            prior_debug = prepared.get("prior_debug") or {}
            if lut is not None and denoised is not None:
                logger.info(
                    "[LUTDelta Debug] denoised_to_lut_known=%.6f denoised_to_lut_hole=%.6f "
                    "condmu_to_lut_known=%.6f condmu_to_lut_hole=%.6f rawprior_to_safeprior_hole=%.6f",
                    self._masked_l1(denoised, lut, self.mask) or 0.0,
                    self._masked_l1(denoised, lut, self.mask_hole) or 0.0,
                    self._masked_l1(self.condition, lut, self.mask) or 0.0,
                    self._masked_l1(self.condition, lut, self.mask_hole) or 0.0,
                    self._masked_l1(raw_prior, self.color_prior, self.mask_hole) or 0.0,
                )

            image_for_lut = prior_debug.get("image_for_lut")
            color_prior_lut = prior_debug.get("color_prior_lut")
            if image_for_lut is not None and color_prior_lut is not None:
                logger.info(
                    "[ColorTransform Debug] degraded_to_prefill_known=%.6f degraded_to_prefill_hole=%.6f "
                    "prefill_to_lut_known=%.6f prefill_to_lut_hole=%.6f",
                    self._masked_l1(self.original_degraded, image_for_lut, self.mask) or 0.0,
                    self._masked_l1(self.original_degraded, image_for_lut, self.mask_hole) or 0.0,
                    self._masked_l1(image_for_lut, color_prior_lut, self.mask) or 0.0,
                    self._masked_l1(image_for_lut, color_prior_lut, self.mask_hole) or 0.0,
                )
        if self.state_0 is not None:
            gt = self.state_0.to(device=final.device)
            lut = prepared.get("lut_transformed") if prepared is not None else None
            target_like = prepared.get("training_target_like") if prepared is not None else None
            final_prior_l1 = self._masked_l1(final, self.color_prior, self.mask_hole) or 0.0
            final_lut_l1 = self._masked_l1(final, lut, self.mask_hole) or 0.0
            final_white_ratio_hole = final_hole.get("white_ratio", 0.0)
            training_target_to_lut = self._masked_l1(gt, lut, self.mask_hole) or 0.0
            target_like_to_gt = self._masked_l1(target_like, gt, self.mask_hole) if target_like is not None else float("nan")
            logger.info(
                "[Target Debug] final_gt_l1=%.6f raw_gt_l1=%.6f prior_gt_l1=%.6f "
                "lut_gt_l1=%.6f training_target_to_lut=%.6f final_prior_l1=%.6f "
                "final_lut_l1=%.6f final_white_ratio_hole=%.6f target_like_gt_l1=%.6f",
                self._masked_l1(final, gt, self.mask_hole) or 0.0,
                self._masked_l1(pred_full, gt, self.mask_hole) or 0.0,
                self._masked_l1(self.color_prior, gt, self.mask_hole) or 0.0,
                training_target_to_lut,
                training_target_to_lut,
                final_prior_l1,
                final_lut_l1,
                final_white_ratio_hole,
                target_like_to_gt,
                )
            if final_white_ratio_hole > 0.05 or final_hole.get("mean", 0.0) > 0.85:
                logger.warning(
                    "[WhiteMask Alert] sample=%s final_hole_mean=%.4f final_white_ratio_hole=%.4f "
                    "sde_mu_hole_mode=%s. This is the old white-mask failure; do not treat this run as fixed.",
                    self.sample_name,
                    final_hole.get("mean", 0.0),
                    final_white_ratio_hole,
                    self.inference_opt.get("sde_mu_hole_mode", "known_only"),
                )
            mu_hole_mode = str(self.inference_opt.get("sde_mu_hole_mode", "known_only")).lower()
            if mu_hole_mode != "known_only":
                logger.info(
                    "[MuAnchor Debug] mode=%s final_lut_l1=%.6f final_prior_l1=%.6f "
                    "note=%s",
                    mu_hole_mode,
                    final_lut_l1,
                    final_prior_l1,
                    (
                        "pass-through: final is almost the hole mu anchor, so the sampler is not adding texture"
                        if final_lut_l1 < 0.01 or final_prior_l1 < 0.01
                        else "not a pure anchor copy"
                    ),
                )
        if prepared is not None and prepared.get("prior_reliability") is not None:
            reliability = prepared["prior_reliability"].to(device=final.device).detach().float()
            threshold = self.confidence_debug_threshold
            low_mask = self.mask_hole * (reliability < threshold).float()
            high_mask = self.mask_hole * (reliability >= threshold).float()
            hole_pixels = self.mask_hole.sum().clamp_min(1.0)
            low_ratio = float(low_mask.sum().div(hole_pixels).item())
            high_ratio = float(high_mask.sum().div(hole_pixels).item())
            final_low = self._masked_rgb_stats(final, low_mask)
            final_high = self._masked_rgb_stats(final, high_mask)
            logger.info(
                "[ConfidenceSlice Debug] threshold=%.3f low_ratio=%.4f high_ratio=%.4f "
                "final_low(mean=%.4f,white=%.4f) final_high(mean=%.4f,white=%.4f)",
                threshold,
                low_ratio,
                high_ratio,
                final_low.get("mean", 0.0),
                final_low.get("white_ratio", 0.0),
                final_high.get("mean", 0.0),
                final_high.get("white_ratio", 0.0),
            )
            if self.state_0 is not None:
                gt = self.state_0.to(device=final.device)
                lut = prepared.get("lut_transformed") if prepared is not None else None
                low_has_pixels = float(low_mask.sum().item()) > 0.0
                high_has_pixels = float(high_mask.sum().item()) > 0.0
                logger.info(
                    "[TargetByConfidence Debug] threshold=%.3f "
                    "final_gt_low=%.6f prior_gt_low=%.6f lut_gt_low=%.6f "
                    "final_gt_high=%.6f prior_gt_high=%.6f lut_gt_high=%.6f",
                    threshold,
                    self._masked_l1(final, gt, low_mask) if low_has_pixels else float("nan"),
                    self._masked_l1(self.color_prior, gt, low_mask) if low_has_pixels else float("nan"),
                    self._masked_l1(lut, gt, low_mask) if low_has_pixels else float("nan"),
                    self._masked_l1(final, gt, high_mask) if high_has_pixels else float("nan"),
                    self._masked_l1(self.color_prior, gt, high_mask) if high_has_pixels else float("nan"),
                    self._masked_l1(lut, gt, high_mask) if high_has_pixels else float("nan"),
                )

    def load(self):
        """Load texture G, structure Gs, and discriminator D in official order."""
        strict_load = self.opt["path"].get("strict_load", True)

        load_path_g = self.opt["path"].get("pretrain_model_G")
        if load_path_g:
            logger.info("Loading model for G [%s] ...", load_path_g)
            checkpoint = torch.load(load_path_g, map_location=self.device)
            model_state = {}
            mu_denoiser_state = {}
            for key, value in checkpoint.items():
                if key.startswith("mu_denoiser."):
                    mu_denoiser_state[key[len("mu_denoiser.") :]] = value
                elif key.startswith("module.mu_denoiser."):
                    mu_denoiser_state[key[len("module.mu_denoiser.") :]] = value
                elif key.startswith("module."):
                    model_state[key[7:]] = value
                else:
                    model_state[key] = value
            self.load_network_from_state(model_state, self.model, strict=strict_load)

            if self.use_mu_denoiser:
                if not mu_denoiser_state:
                    logger.warning(
                        "[Model] Mu-Denoiser is enabled but checkpoint has no mu_denoiser.* weights; "
                        "falling back to LUT condition_mu."
                    )
                else:
                    self.mu_denoiser.load_state_dict(mu_denoiser_state, strict=False)
                    self.mu_denoiser_has_weights = True
            elif mu_denoiser_state:
                logger.info(
                    "[LoadCheck] ignored %d mu_denoiser tensors because mu_denoiser.enabled=false",
                    len(mu_denoiser_state),
                )


        load_path_gs = self.opt["path"].get("pretrain_model_Gs")
        if load_path_gs:
            logger.info("Loading model for Gs [%s] ...", load_path_gs)
            checkpoint_gs = torch.load(load_path_gs, map_location=self.device)
            gs_state = {}
            for key, value in checkpoint_gs.items():
                gs_state[key[7:] if key.startswith("module.") else key] = value
            self.load_network_from_state(gs_state, self.models, strict=strict_load)

        load_path_d = self.opt["path"].get("pretrain_model_D")
        if load_path_d:
            logger.info("Loading model for D [%s] ...", load_path_d)
            checkpoint_d = torch.load(load_path_d, map_location=self.device)
            d_state = {}
            for key, value in checkpoint_d.items():
                d_state[key[7:] if key.startswith("module.") else key] = value
            self.load_network_from_state(d_state, self.dis, strict=strict_load)

    def save(self, iter_label):
        raise NotImplementedError("The texture-1 tree is used for inference only.")
