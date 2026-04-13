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
        self.prior_method = str(self.dataset_opt.get("prior_method", "quality")).lower()
        self.inference_mode = self.inference_opt.get("mode", "auto")

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
        self.output = None
        self.raw_output = None
        self.debug_outputs: Dict[str, torch.Tensor] = {}

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
            lut_transformed = (
                denoised_original * (1 - effective_weight)
                + lut_transformed * effective_weight
            )

        # ?????: known ????? CondLUT?hole ??? prior generator?
        # ?? confidence ???? CondLUT??????/?? hole ???/?? prior ???
        # Align known pixels to the same target-domain CondLUT used by condition_mu,
        # but keep ColorPriorGenerator's inpainted prior in holes.  Confidence is
        # passed as reliability guidance, not used to overwrite hole colors.
        color_prior = lut_transformed * mask_known + color_prior * mask_hole
        confidence = torch.ones_like(mask_known) * mask_known + confidence * mask_hole

        if self.save_intermediates:
            mask_hole_3c = mask_hole.expand(-1, color_prior.shape[1], -1, -1)
            hole_denom = mask_hole_3c.sum().clamp_min(1.0)
            cp_hole_mean = float((color_prior * mask_hole_3c).sum().item() / hole_denom.item())
            cp_hole_std = float((((color_prior - cp_hole_mean) ** 2) * mask_hole_3c).sum().div(hole_denom).sqrt().item())
            white_ratio = float((((color_prior > 0.95).all(dim=1, keepdim=True).float() * mask_hole).sum() / mask_hole.sum().clamp_min(1.0)).item())
            conf_hole_mean = float((confidence * mask_hole).sum().div(mask_hole.sum().clamp_min(1.0)).item())
            logger.info(
                "[ColorPrior Debug] hole_mean=%.6f hole_std=%.6f hole_white_ratio=%.6f confidence_hole_mean=%.6f",
                cp_hole_mean,
                cp_hole_std,
                white_ratio,
                conf_hole_mean,
            )

        if self.use_mu_denoiser and getattr(self, "mu_denoiser_has_weights", False):
            mu_clean_lut = self.mu_denoiser_trainer.inference(
                lut_transformed,
                mask_known,
                confidence,
            ).clamp(0.0, 1.0)
        else:
            mu_clean_lut = lut_transformed
        condition_mu = mu_clean_lut * mask_known

        return {
            "denoised_original": denoised_original,
            "lut_transformed": lut_transformed,
            "condition_lut": lut_transformed,
            "condition_mu": condition_mu,
            "color_prior": color_prior,
            "confidence": confidence,
            "mu_clean_lut": mu_clean_lut,
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
                self.condition = prepared["condition_mu"]
                sde.set_mu(self.condition)
                x_init = self.condition
                self.state = sde.noise_state(x_init)

                # Structure guidance stays auxiliary. It should come from the
                # denoised observed input, not from mu_clean.
                structure_gray, structure_edge = self._build_structure_targets(
                    prepared["denoised_original"]
                )
                structure_state = None
                if S_sde is not None:
                    S_sde.set_mu(structure_edge * self.mask)
                    structure_state = S_sde.noise_state(structure_edge * self.mask)

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
                self.output = known_source * self.mask + pred_full * self.mask_hole

                self._log_inference_debug_stats(pred_full, self.output)

                self.debug_outputs = {
                    "original_degraded": self.original_degraded,
                    "mask_hole": self.mask_hole,
                    "mask_known": self.mask,
                    "color_prior": self.color_prior,
                    "color_prior_lut": prior_debug.get("color_prior_lut"),
                    "color_prior_inpainted": prior_debug.get("color_prior_inpainted"),
                    "conf_lut_prior": prior_debug.get("conf_lut"),
                    "conf_inpaint_prior": prior_debug.get("conf_inpaint"),
                    "training_target_like": target_like,
                    "confidence": self.confidence,
                    "denoised_original": prepared["denoised_original"],
                    "condition_lut": prepared["condition_lut"],
                    "lut_transformed": prepared["lut_transformed"],
                    "condition_mu": self.condition,
                    "mu_clean_lut": prepared["mu_clean_lut"],
                    "mu_clean": mu_clean,
                    "x_init": x_init,
                    "structure_gray": structure_gray,
                    "structure_edge": structure_edge,
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
        white = ((rgb > 0.95).all(dim=1, keepdim=True).float() * mask[:, :1]).sum()
        white_denom = mask[:, :1].sum().clamp_min(1.0)
        return {
            "mean": float(mean.item()),
            "min": float(rgb.min().item()),
            "max": float(rgb.max().item()),
            "white_ratio": float((white / white_denom).item()),
        }

    def _log_inference_debug_stats(self, pred_full, final):
        if self.mask_hole is None:
            return
        raw_hole = self._masked_rgb_stats(pred_full, self.mask_hole)
        final_hole = self._masked_rgb_stats(final, self.mask_hole)
        cond_known = self._masked_rgb_stats(self.condition, self.mask)
        logger.info(
            "[Inference Debug] gt_mode=%s deterministic_reverse=%s "
            "raw_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "final_hole(mean=%.4f,min=%.4f,max=%.4f,white=%.4f) "
            "cond_known(mean=%.4f,min=%.4f,max=%.4f,white=%.4f)",
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
