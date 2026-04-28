import logging
from collections import OrderedDict
import os
import numpy as np

import math
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torchvision.utils as tvutils
from tqdm import tqdm
from ema_pytorch import EMA

import torch.nn.functional as F
import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

# LUT处理器（用于对去噪图像应用颜色变换）
from lut_processor import LUTProcessor

# ============ Self-Supervised Mu-Denoiser ============
# 用于在 SDE 训练前清理条件均值 mu
try:
    from models.mu_denoiser import MuDenoiser, MuDenoiserTrainer
    HAS_MU_DENOISER = True
except ImportError:
    HAS_MU_DENOISER = False
    print("[Warning] MuDenoiser 未找到，将使用原始 mu")

logger = logging.getLogger("base")


import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():

    return transforms.ToPILImage()


def image_to_tensor():

    return transforms.ToTensor()


def gray_to_edge(image, sigma):

    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

    return edge



class AdversarialLoss(nn.Module):
  r"""
  Adversarial loss
  https://arxiv.org/abs/1711.10337
  """

  def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
    r"""
    type = nsgan | lsgan | hinge
    """
    super(AdversarialLoss, self).__init__()
    self.type = type
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))

    if type == 'nsgan':
      self.criterion = nn.BCELoss()
    elif type == 'lsgan':
      self.criterion = nn.MSELoss()
    elif type == 'hinge':
      self.criterion = nn.ReLU()

  def patchgan(self, outputs, is_real=None, is_disc=None):
    if self.type == 'hinge':
      if is_disc:
        if is_real:
          outputs = -outputs
        return self.criterion(1 + outputs).mean()
      else:
        return (-outputs).mean()
    else:
      labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
      loss = self.criterion(outputs, labels)
      return loss

  def __call__(self, outputs, is_real=None, is_disc=None):
    return self.patchgan(outputs, is_real, is_disc)


class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        #if opt["dist"]:
            #self.rank = torch.distributed.get_rank()
        #else:
            #self.rank = -1  # non dist training
        train_opt = opt["train"]
        self.train_opt = train_opt
        self.loaded_model_param_names = set()
        self.freeze_pretrained_until_iter = int(
            train_opt.get("freeze_pretrained_until_iter", 0)
        )
        self.freeze_loaded_pretrained_only = bool(
            train_opt.get("freeze_loaded_pretrained_only", True)
        )
        self.enable_pretrained_freeze = (
            self.freeze_pretrained_until_iter > 0
            and not bool(opt.get("path", {}).get("resume_state"))
        )
        self._frozen_pretrained_param_names = set()
        self._pretrained_trunk_frozen = False
        self._pretrained_trunk_unfrozen = False
        

        # define network and load pretrained models
        self.model, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.dis = self.dis.to(self.device)
        
        #必改
        gpu_ids = opt.get('gpu_ids', None)
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids, output_device=gpu_ids[0])
            self.dis   = DataParallel(self.dis,   device_ids=gpu_ids, output_device=gpu_ids[0])

        # ============ 加载 LUT 处理器 ============
        # 用于对去噪后的图像应用颜色变换
        train_dataset_opt = opt.get('datasets', {}).get('train', {})
        lut_path = train_dataset_opt.get('lut_path', None)
        if lut_path is not None and os.path.exists(lut_path):
            self.lut_processor = LUTProcessor(lut_path)
            logger.info(f"[Model] 已加载 LUT 处理器: {lut_path}")
        else:
            self.lut_processor = None
            if lut_path:
                logger.warning(f"[Model] LUT 文件不存在: {lut_path}")
        
        # LUT / target-domain configuration.
        self.gt_mode = train_dataset_opt.get('gt_mode', 'full')
        self.lut_strength = float(train_dataset_opt.get('lut_strength', 1.0))
        self.lut_fade_boost = float(train_dataset_opt.get('lut_fade_boost', 3.0))
        self.lut_smooth_radius = int(train_dataset_opt.get('lut_smooth_radius', 0))
        logger.info(
            f"[Model] GT mode={self.gt_mode}, LUT strength={self.lut_strength}, "
            f"LUT fade boost={self.lut_fade_boost}, smooth radius={self.lut_smooth_radius}"
        )
        logger.info(
            "[Model] train.sde_mu_hole_mode=%s infer_x0_loss_weight=%s infer_x0_grad=%s",
            opt.get('train', {}).get('sde_mu_hole_mode', 'known_only'),
            opt.get('train', {}).get('infer_x0_loss_weight', 0.0),
            opt.get('train', {}).get('infer_x0_grad', False),
        )
        
        # ============ 初始化 Self-Supervised Mu-Denoiser ============
        mu_denoiser_opt = opt.get('mu_denoiser', {})
        self.mu_denoiser_opt = mu_denoiser_opt
        self.use_mu_denoiser = mu_denoiser_opt.get('enabled', False) and HAS_MU_DENOISER
        self.use_mu_denoiser_for_condition_mu = bool(
            mu_denoiser_opt.get("use_for_condition_mu", False)
        )
        self.mu_denoiser_has_weights = False
        self.mu_denoiser_loaded_weights = False
        
        if self.use_mu_denoiser:
            self.mu_denoiser = MuDenoiser(
                in_nc=mu_denoiser_opt.get('in_nc', 5),
                dim=mu_denoiser_opt.get('dim', 32),
                num_blocks=mu_denoiser_opt.get('num_blocks', 2),
                num_heads=mu_denoiser_opt.get('num_heads', 4),
                predict_residual=mu_denoiser_opt.get('predict_residual', True),
            ).to(self.device)
            
            # 训练器封装自监督训练逻辑
            self.mu_denoiser_trainer = MuDenoiserTrainer(
                self.mu_denoiser,
                blind_ratio=mu_denoiser_opt.get('blind_ratio', 0.1)
            )
            
            # 超参数
            self.lambda_ss = mu_denoiser_opt.get('lambda_ss', 1.0)
            self.lambda_tv = mu_denoiser_opt.get('lambda_tv', 0.01)
            
            logger.info(f"[Model] Mu-Denoiser 已启用: dim={mu_denoiser_opt.get('dim', 32)}, "
                        f"blocks={mu_denoiser_opt.get('num_blocks', 2)}, "
                        f"blind_ratio={mu_denoiser_opt.get('blind_ratio', 0.1)}")
        else:
            self.mu_denoiser = None
            self.mu_denoiser_trainer = None
            self.optimizer_mu = None
            self.scheduler_mu = None
            self._optimizer_mu_init_lr = None
            if mu_denoiser_opt.get('enabled', False) and not HAS_MU_DENOISER:
                logger.warning("[Model] Mu-Denoiser 配置已启用但模块未找到")
        
        # Load checkpoints after Mu-Denoiser is constructed so optional
        # mu_denoiser.* weights can be restored when present.
        self.load()
        
        if self.is_train:
            self.model.train()
            self.dis.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss_tri = nn.TripletMarginLoss().to(self.device)
            self.adversarial_loss = AdversarialLoss(type = 'hinge').to(self.device)
            self.weight = opt['train']['weight']
            # Optional x0 reconstruction auxiliary loss.
            #
            # The original one-step SDE supervision optimizes x_t -> x_{t-1}.
            # In mural inpainting the inference failure can still happen at the
            # final x0 hole region: the one-step loss is low, but the accumulated
            # reverse trajectory saturates the hole to white.  This auxiliary
            # term does not change the SDE formula or network structure; it only
            # decodes the current model-predicted noise back to an estimated x0
            # using the existing forward relation and supervises it against the
            # target-domain training_target.
            self.x0_recon_loss_weight = float(train_opt.get("x0_recon_loss_weight", 0.0))
            self.x0_recon_loss_start_iter = int(train_opt.get("x0_recon_loss_start_iter", 0))
            self.x0_recon_clamp_b_min = float(train_opt.get("x0_recon_clamp_b_min", 1e-3))
            # Timestep-aware decay for x0 auxiliary loss.
            # At high-noise timesteps B(t) is small, so x0_hat = mu + (xt-mu-sigma*noise)/B
            # amplifies noise prediction errors.  We decay the x0 loss weight as t increases
            # so the high-t curriculum does not inadvertently amplify x0 gradients.
            # weight_at_t = x0_recon_loss_weight * exp(-x0_high_t_decay * t/T)
            # x0_high_t_decay=0 disables the decay (original behavior).
            self.x0_high_t_decay = float(train_opt.get("x0_high_t_decay", 0.0))
            if self.x0_recon_loss_weight > 0:
                logger.info(
                    "[Model] x0 reconstruction auxiliary loss enabled: "
                    f"weight={self.x0_recon_loss_weight}, "
                    f"start_iter={self.x0_recon_loss_start_iter}, "
                    f"clamp_b_min={self.x0_recon_clamp_b_min}, "
                    f"high_t_decay={self.x0_high_t_decay}"
                )
            # Extra mural inpainting bootstrap loss.
            #
            # The normal one-step target samples xt from the forward process of
            # training_target, so hole pixels still contain B(t) * target
            # information.  In inference, however, the hole part of x_init is
            # condition_mu plus noise (known-only mode uses 0 in holes).  If the reverse trajectory
            # drifts off-manifold, late low-noise steps see blank/white holes
            # that the model was not trained to correct.  This auxiliary branch
            # keeps the SDE formula unchanged but trains the same network call on
            # an inference-like state: known area follows the normal forward
            # state, hole area is blank/noisy from condition_mu.  It directly
            # supervises the implied x0 against the target-domain GT.
            self.infer_x0_loss_weight = float(train_opt.get("infer_x0_loss_weight", 0.0))
            self.infer_x0_loss_start_iter = int(train_opt.get("infer_x0_loss_start_iter", 0))
            self.infer_x0_t_min_ratio = float(train_opt.get("infer_x0_t_min_ratio", 0.10))
            self.infer_x0_t_max_ratio = float(train_opt.get("infer_x0_t_max_ratio", 0.70))
            self.infer_x0_grad = bool(train_opt.get("infer_x0_grad", False))
            self.infer_x0_loss_interval = max(1, int(train_opt.get("infer_x0_loss_interval", 1)))
            self.infer_x0_microbatch = max(0, int(train_opt.get("infer_x0_microbatch", 0)))
            self.require_infer_x0_grad_for_known_only = bool(
                train_opt.get("require_infer_x0_grad_for_known_only", False)
            )
            _mu_mode_for_guard = str(train_opt.get("sde_mu_hole_mode", "known_only")).lower()
            _infer_grad_ok = self.infer_x0_loss_weight > 0 and self.infer_x0_grad
            if _mu_mode_for_guard == "known_only" and not _infer_grad_ok:
                _msg = (
                    "[X8Guard] known_only removes target/color content from hole during inference; "
                    "training must enable a real inference-like blank-hole loss. "
                    f"Got infer_x0_loss_weight={self.infer_x0_loss_weight}, "
                    f"infer_x0_grad={self.infer_x0_grad}."
                )
                if self.require_infer_x0_grad_for_known_only:
                    raise ValueError(_msg)
                logger.warning(_msg)
            if self.infer_x0_loss_weight > 0:
                logger.info(
                    "[Model] inference-like blank-hole x0 loss enabled: "
                    f"weight={self.infer_x0_loss_weight}, "
                    f"start_iter={self.infer_x0_loss_start_iter}, "
                    f"t_range=[{self.infer_x0_t_min_ratio}, {self.infer_x0_t_max_ratio}], "
                    f"grad={self.infer_x0_grad}, "
                    f"interval={self.infer_x0_loss_interval}, "
                    f"microbatch={self.infer_x0_microbatch}"
                )

            # x12: keep the main diffusion target on the stable raw domain,
            # and use the LUT domain only as a weak hole-only color auxiliary.
            self.color_aux_loss_weight = float(train_opt.get("color_aux_loss_weight", 0.0))
            self.color_aux_loss_start_iter = int(train_opt.get("color_aux_loss_start_iter", 0))
            self.color_aux_blur_kernel = int(train_opt.get("color_aux_blur_kernel", 7))
            self.color_aux_clamp_b_min = float(train_opt.get("color_aux_clamp_b_min", 1e-3))
            if self.color_aux_blur_kernel > 1 and self.color_aux_blur_kernel % 2 == 0:
                self.color_aux_blur_kernel += 1
            if self.color_aux_loss_weight > 0:
                logger.info(
                    "[Model] color auxiliary loss enabled: "
                    f"weight={self.color_aux_loss_weight}, "
                    f"start_iter={self.color_aux_loss_start_iter}, "
                    f"blur_kernel={self.color_aux_blur_kernel}, "
                    f"clamp_b_min={self.color_aux_clamp_b_min}"
                )

            # optimizers
            self.optimizer_d = torch.optim.Adam(self.dis.parameters(), lr = 1e-4, betas = (0.5, 0.99))#1e-4
            
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            # Split parameters into two groups:
            # - pretrained (backbone UNet): use the configured lr_G (low, e.g. 5e-6)
            # - new modules (BrushNet, MGLC, mu_denoiser backbone): use lr_new
            #   which defaults to 10x lr_G so randomly-initialised weights converge
            #   faster without destabilising the pretrained backbone.
            lr_new = float(train_opt.get("lr_new", train_opt["lr_G"] * 10))
            fallback_new_module_prefixes = (
                "brushnet.",
                "mglc_mid.",
                "mglc_dec.",
                "main_guidance_proj.",
            )
            pretrained_param_names = self._resolve_pretrained_param_names(
                fallback_new_module_prefixes
            )
            pretrained_params = []
            new_params = []
            for k, v in self.model.named_parameters():
                if not v.requires_grad:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))
                    continue
                is_new = k not in pretrained_param_names
                if is_new:
                    new_params.append(v)
                else:
                    pretrained_params.append(v)
            optim_params = [
                {"params": pretrained_params, "lr": train_opt["lr_G"]},
                {"params": new_params, "lr": lr_new},
            ]
            logger.info(
                "[Model] Param groups: pretrained=%d (lr=%.2e), new=%d (lr=%.2e)",
                len(pretrained_params), train_opt["lr_G"], len(new_params), lr_new,
            )


            if train_opt['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'AdamW':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            elif train_opt['optimizer'] == 'Lion':
                self.optimizer = Lion(
                    optim_params, 
                    lr=train_opt["lr_G"],
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
            else:
                print('Not implemented optimizer, default using Adam!')

            self.optimizers.append(self.optimizer)
            
            # Mu-Denoiser 优化器（独立于主网络）
            if self.use_mu_denoiser:
                self.optimizer_mu = torch.optim.Adam(
                    self.mu_denoiser.parameters(),
                    lr=mu_denoiser_opt.get('lr', 1e-4),
                    betas=(0.9, 0.999),
                )
                self._optimizer_mu_init_lr = mu_denoiser_opt.get('lr', 1e-4)
                self.scheduler_mu = self._build_mu_scheduler(train_opt)
                logger.info(f"[Model] Mu-Denoiser 优化器已创建: lr={mu_denoiser_opt.get('lr', 1e-4)}")

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(
                            optimizer,
                            train_opt["lr_steps"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                            gamma=train_opt["lr_gamma"],
                            clear_state=train_opt["clear_state"],
                        )
                    )
            elif train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, 
                            T_max=train_opt["niter"],
                            eta_min=train_opt["eta_min"])
                    ) 
            else:
                raise NotImplementedError("MultiStepLR learning rate scheme is enough.")

            self._apply_pretrained_trunk_freeze()
            self.ema = EMA(self.model, beta=0.995, update_every=10).to(self.device)
            self.log_dict = OrderedDict()

    def _build_mu_scheduler(self, train_opt):
        if not self.use_mu_denoiser or self.optimizer_mu is None:
            return None

        if train_opt["lr_scheme"] == "MultiStepLR":
            return lr_scheduler.MultiStepLR_Restart(
                self.optimizer_mu,
                train_opt["lr_steps"],
                restarts=train_opt["restarts"],
                weights=train_opt["restart_weights"],
                gamma=train_opt["lr_gamma"],
                clear_state=train_opt["clear_state"],
            )
        if train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_mu,
            T_max=train_opt["niter"],
            eta_min=train_opt["eta_min"],
        )
        raise NotImplementedError("Unsupported lr scheme for Mu-Denoiser.")

    def _resolve_pretrained_param_names(self, fallback_new_module_prefixes):
        loaded_names = set(getattr(self, "loaded_model_param_names", set()) or set())
        if loaded_names and self.freeze_loaded_pretrained_only:
            return {
                name
                for name, _ in self.model.named_parameters()
                if name in loaded_names
            }
        return {
            name
            for name, _ in self.model.named_parameters()
            if not any(name.startswith(prefix) for prefix in fallback_new_module_prefixes)
        }

    def _apply_pretrained_trunk_freeze(self):
        if not self.enable_pretrained_freeze or self._pretrained_trunk_frozen:
            return
        if self.freeze_pretrained_until_iter <= 0:
            return

        frozen_names = self._resolve_pretrained_param_names(
            ("brushnet.", "mglc_mid.", "mglc_dec.", "main_guidance_proj.")
        )
        for name, param in self.model.named_parameters():
            if name in frozen_names:
                param.requires_grad_(False)

        self._frozen_pretrained_param_names = frozen_names
        self._pretrained_trunk_frozen = True
        self._pretrained_trunk_unfrozen = False
        logger.info(
            "[Freeze] frozen %d pretrained trunk params until iter %d",
            len(frozen_names),
            self.freeze_pretrained_until_iter,
        )

    def _maybe_unfreeze_pretrained_trunk(self, step):
        if not self.enable_pretrained_freeze:
            return
        if not self._pretrained_trunk_frozen or self._pretrained_trunk_unfrozen:
            return
        if int(step) < self.freeze_pretrained_until_iter:
            return

        for name, param in self.model.named_parameters():
            if name in self._frozen_pretrained_param_names:
                param.requires_grad_(True)

        self._pretrained_trunk_unfrozen = True
        logger.info(
            "[Freeze] unfroze pretrained trunk at iter %d (%d params)",
            int(step),
            len(self._frozen_pretrained_param_names),
        )

    def _set_optimizer_mu_lr(self, lr_value):
        if self.optimizer_mu is None:
            return
        for param_group in self.optimizer_mu.param_groups:
            param_group["lr"] = lr_value

    def _sync_mu_lr_to_iter(self, cur_iter, warmup_iter=-1):
        if self.optimizer_mu is None or self._optimizer_mu_init_lr is None:
            return

        if warmup_iter is not None and warmup_iter > 0 and cur_iter < warmup_iter:
            self._set_optimizer_mu_lr(self._optimizer_mu_init_lr / warmup_iter * cur_iter)
            return

        train_opt = self.train_opt
        if train_opt["lr_scheme"] == "MultiStepLR":
            gamma = train_opt["lr_gamma"]
            step_count = sum(1 for step in train_opt["lr_steps"] if cur_iter >= step)
            self._set_optimizer_mu_lr(self._optimizer_mu_init_lr * (gamma ** step_count))
            return

        if train_opt["lr_scheme"] == "TrueCosineAnnealingLR":
            eta_min = train_opt["eta_min"]
            t_max = max(1, train_opt["niter"])
            cosine = math.cos(math.pi * min(cur_iter, t_max) / t_max)
            lr_value = eta_min + (self._optimizer_mu_init_lr - eta_min) * (1 + cosine) / 2
            self._set_optimizer_mu_lr(lr_value)
            return

        raise NotImplementedError("Unsupported lr scheme for Mu-Denoiser.")

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        super(DenoisingModel, self).update_learning_rate(cur_iter, warmup_iter)
        if self.scheduler_mu is not None:
            self.scheduler_mu.step()
            if warmup_iter is not None and warmup_iter > 0 and cur_iter < warmup_iter:
                self._set_optimizer_mu_lr(self._optimizer_mu_init_lr / warmup_iter * cur_iter)

    def save_training_state(self, epoch, iter_step, label=None, extra_state=None):
        state = {"epoch": epoch, "iter": iter_step, "schedulers": [], "optimizers": []}
        for scheduler in self.schedulers:
            state["schedulers"].append(scheduler.state_dict())
        for optimizer in self.optimizers:
            state["optimizers"].append(optimizer.state_dict())

        if self.use_mu_denoiser and self.optimizer_mu is not None:
            state["optimizer_mu"] = self.optimizer_mu.state_dict()
        if self.use_mu_denoiser and self.scheduler_mu is not None:
            state["scheduler_mu"] = self.scheduler_mu.state_dict()

        if extra_state:
            state.update(extra_state)

        save_label = label if label is not None else str(iter_step)
        save_filename = "{}.state".format(save_label)
        save_path = os.path.join(self.opt["path"]["training_state"], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        super(DenoisingModel, self).resume_training(resume_state)

        if not self.use_mu_denoiser or self.optimizer_mu is None:
            return

        if "optimizer_mu" in resume_state:
            self.optimizer_mu.load_state_dict(resume_state["optimizer_mu"])
        else:
            logger.warning("[Model] resume_state 中缺少 optimizer_mu，将按当前迭代同步 Mu-Denoiser 学习率。")

        if self.scheduler_mu is not None:
            if "scheduler_mu" in resume_state:
                self.scheduler_mu.load_state_dict(resume_state["scheduler_mu"])
            else:
                logger.warning("[Model] resume_state 中缺少 scheduler_mu，将按当前迭代推断 Mu-Denoiser 学习率。")
                self._sync_mu_lr_to_iter(
                    int(float(resume_state["iter"])),
                    self.train_opt.get("warmup_iter", -1),
                )


    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ, 
                  color_prior=None, confidence=None, conf_lut=None,
                  original_degraded=None, reference_degraded=None,
                  condition_lut=None, mu_clean_lut=None,
                  training_target_lut=None,
                  denoised_observed_mask_aware=None):
        """
        加载训练数据
        
        Args:
            state: 噪声状态
            LQ: 低质量输入（条件，可能带 mask 涂黑）
            GT: Ground Truth（LUT 变换后的目标）
            mask: 掩码
            S_sde: 结构SDE
            S_GT: 结构GT
            S_LQ: 结构LQ
            color_prior: [可选] 颜色先验图，用于BrushNet
            confidence: [可选] 置信度图，用于BrushNet
            conf_lut: [可选] LUT置信度图
            original_degraded: [可选] 当前观测输入（真实缺损外观），用于条件链
            reference_degraded: [可选] 完整褪色参考图，仅用于训练目标生成
        """
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)   # LQ（可能带 mask 涂黑）
        self.state_0 = GT.to(self.device)     # GT
        self.mask = mask.to(self.device)      # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)
        
        # 当前观测输入：模拟真实推理时已经存在缺损的输入图像
        if original_degraded is not None:
            self.original_degraded = original_degraded.to(self.device)
        else:
            self.original_degraded = self.condition

        # 完整参考图：只用于构造训练目标，不参与推理时的条件生成
        if reference_degraded is not None:
            self.reference_degraded = reference_degraded.to(self.device)
        else:
            self.reference_degraded = self.original_degraded
        
        # BrushNet条件
        if color_prior is not None:
            self.color_prior = color_prior.to(self.device)
        else:
            self.color_prior = None
            
        if confidence is not None:
            self.confidence = confidence.to(self.device)
        else:
            self.confidence = None
        
        # LUT置信度
        if conf_lut is not None:
            self.conf_lut = conf_lut.to(self.device)
        else:
            self.conf_lut = None

        # Target-domain condition chain used by mural training/debug.
        self.condition_lut = (
            condition_lut.to(self.device) if condition_lut is not None else self.condition
        )
        self.mu_clean_lut = (
            mu_clean_lut.to(self.device) if mu_clean_lut is not None else self.condition_lut
        )
        self.training_target_lut = (
            training_target_lut.to(self.device) if training_target_lut is not None else self.state_0
        )
        self.denoised_observed_mask_aware = (
            denoised_observed_mask_aware.to(self.device)
            if denoised_observed_mask_aware is not None
            else None
        )


    def _blur_for_color_aux(self, tensor, kernel_size):
        if tensor is None or kernel_size <= 1:
            return tensor
        pad = kernel_size // 2
        padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    def _estimate_x0_from_noise(self, sde, xt, noise, timesteps):
        batch = xt.shape[0]
        t = timesteps.reshape(batch, 1, 1, 1).long().to(xt.device)
        sigma = sde.sigma_bar(t).to(dtype=xt.dtype, device=xt.device)
        b = torch.exp(-sde.thetas_cumsum[t] * sde.dt).to(dtype=xt.dtype, device=xt.device)
        mu = self.condition.to(dtype=xt.dtype, device=xt.device)
        x0_hat = mu + (xt - mu - sigma * noise) / b.clamp_min(self.color_aux_clamp_b_min)
        return x0_hat.clamp(0.0, 1.0)

    def compute_mu_clean_no_grad(self, condition_lut, mask_known, confidence=None, step=None):
        """
        Return target-domain MuCleanr output for the current condition_lut.

        condition_lut is already LUT(denoised(observed_degraded)); MuCleanr must
        never receive raw degraded-domain input for SDE mu construction.
        If MuDenoiser is training from scratch, keep SDE mu on plain CondLUT for
        a warmup period so random early MuCleanr weights cannot tint condition_mu.
        """
        if (
            not self.use_mu_denoiser
            or self.mu_denoiser is None
            or not getattr(self, "mu_denoiser_has_weights", False)
        ):
            return condition_lut

        if not getattr(self, "mu_denoiser_loaded_weights", False):
            warmup_iter = int(self.mu_denoiser_opt.get("sde_warmup_iter", 1000))
            if step is None or step < warmup_iter:
                return condition_lut

        with torch.no_grad():
            mu_clean_lut = self.mu_denoiser_trainer.inference(
                condition_lut, mask_known, confidence
            )

        return mu_clean_lut.clamp(0.0, 1.0)

    def optimize_parameters(self, step, timesteps, sde=None):
        from collections import OrderedDict
        import torch.nn as nn
        self.log_dict = OrderedDict()
        self._maybe_unfreeze_pretrained_trunk(step)

        # train.py has already built the single mural target domain and passed it
        # through feed_data(GT=...).  Do not recompute denoised/LUT targets here.
        training_target = self.state_0
        training_target_lut = getattr(self, "training_target_lut", training_target)
        condition_lut_for_mu = getattr(self, "condition_lut", self.condition)
        mu_clean_lut = getattr(self, "mu_clean_lut", condition_lut_for_mu)
        mu_losses = {}

        # ============ Self-Supervised Mu-Denoiser training ============
        mu_denoiser_loss = None
        if self.use_mu_denoiser and self.is_train:
            y_hat, loss_mu, mu_losses = self.mu_denoiser_trainer.train_step(
                y_degraded=condition_lut_for_mu.detach(),
                mask_known=self.mask,
                confidence=self.confidence,
                lambda_ss=self.lambda_ss,
                lambda_tv=self.lambda_tv,
            )
            mu_clean_lut = y_hat.detach().clamp(0.0, 1.0)
            mu_denoiser_loss = loss_mu
            for key, val in mu_losses.items():
                self.log_dict[key] = val

        # Texture SDE mu is exactly the feed_data condition_mu built in train.py:
        # target-domain condition_lut * mask, or MuCleanr(condition_lut) * mask.
        sde.set_mu(self.condition)
        
        # 使用 GT 计算最优逆步骤
        yt_1_optimum = sde.reverse_optimum_step(self.state, training_target, timesteps)
        timesteps = timesteps.to(self.device)
        
        # Get noise and score
        S_timestep, S_optimum = self.S_sde.generate_random_states_texture(x0=self.S_GT, mu=self.S_LQ * self.mask, timesteps = timesteps)
        S_optimum = self.S_sde.reverse_optimum_step(S_optimum, self.S_GT, timesteps)
        
        # ============ 传递BrushNet条件 ============
        brushnet_kwargs = {}
        if self.color_prior is not None:
            brushnet_kwargs['color_prior'] = self.color_prior
        if self.confidence is not None:
            brushnet_kwargs['confidence'] = self.confidence
        if hasattr(self, 'mask') and self.mask is not None:
            # mask约定: self.mask=1表示已知, BrushNet需要1=需要修复
            brushnet_kwargs['mask'] = 1 - self.mask
        if getattr(self, 'original_degraded', None) is not None:
            brushnet_kwargs['observed_degraded'] = self.original_degraded

        model_output = sde.noise_fn(self.state, timesteps.squeeze(), S_optimum, **brushnet_kwargs)
        if isinstance(model_output, (tuple, list)):
            noise = model_output[0]
            maybe_gate = model_output[1] if len(model_output) > 1 else None
            enable_g_score_aux = self.train_opt.get("enable_g_score_aux", False) is True
            is_real_gate = (
                maybe_gate is not None
                and torch.is_tensor(maybe_gate)
                and maybe_gate is not noise
                and maybe_gate.dim() == noise.dim()
                and maybe_gate.shape[0] == noise.shape[0]
                and maybe_gate.shape[-2:] == noise.shape[-2:]
                and maybe_gate.shape[1] == 1
            )
            g_score = maybe_gate if (enable_g_score_aux and is_real_gate) else None
        else:
            noise = model_output
            g_score = None
        # ============ 传递BrushNet条件完成 ============
        
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        
        # ============ 优化器更新 ============
        self.optimizer.zero_grad()
        if self.use_mu_denoiser:
            self.optimizer_mu.zero_grad()
        
        # 主损失：模型输出 vs GT
        loss_components = self.loss_fn.compute_components(
            yt_1_expection, yt_1_optimum, self.mask
        )
        loss = loss_components["loss_total"]

        # ============ 可选 g_score 辅助损失 ============
        g_score_loss_val = 0.0
        if g_score is not None:
            mask_hole = 1 - self.mask  # 1=hole, 0=known
            ones_like_gs = torch.ones_like(g_score)
            _l1 = nn.L1Loss(reduction='mean')
            _l2 = nn.MSELoss()
            g_score_hole_loss = 0.1 * (
                _l1(ones_like_gs * mask_hole, g_score * mask_hole)
                + _l2(ones_like_gs * mask_hole, g_score * mask_hole)
            )
            g_score_blend_loss = _l1(
                yt_1_expection * g_score + (1 - g_score) * yt_1_optimum,
                yt_1_optimum,
            )
            g_score_total = g_score_hole_loss + g_score_blend_loss
            loss = loss + g_score_total
            g_score_loss_val = float(g_score_total.item())

        # 总损失 = 扩散损失 + Mu-Denoiser损失（如果有）
        total_loss = loss
        if mu_denoiser_loss is not None:
            total_loss = total_loss + mu_denoiser_loss
        
        total_loss.backward()
        
        # 梯度裁剪（主网络）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Mu-Denoiser 优化器更新
        if self.use_mu_denoiser:
            torch.nn.utils.clip_grad_norm_(self.mu_denoiser.parameters(), max_norm=1.0)
            self.optimizer_mu.step()
            self.mu_denoiser_has_weights = True

        # set log
        self.log_dict["loss"] = loss.item()
        self.log_dict["loss_main"] = loss.item()
        self.log_dict["loss_total"] = total_loss.item()
        self.log_dict["loss_known"] = loss_components["loss_known"].item()
        self.log_dict["loss_hole"] = loss_components["loss_hole"].item()
        self.log_dict["loss_hole_weighted"] = loss_components["loss_hole_weighted"].item()
        self.log_dict["loss_g_score"] = g_score_loss_val
        self.log_dict["stats_g_score_aux_enabled"] = 1.0 if g_score is not None else 0.0

        self.log_dict["loss_mu_total"] = float(mu_denoiser_loss.item()) if mu_denoiser_loss is not None else 0.0
        self.log_dict["loss_mu_ss"] = float(mu_losses.get("l_ss", 0.0)) if self.use_mu_denoiser and self.is_train else 0.0
        self.log_dict["loss_mu_tv"] = float(mu_losses.get("l_tv", 0.0)) if self.use_mu_denoiser and self.is_train else 0.0
        self.log_dict["lr_main"] = float(self.optimizer.param_groups[0]["lr"])
        if len(self.optimizer.param_groups) > 1:
            self.log_dict["lr_new"] = float(self.optimizer.param_groups[1]["lr"])
        if self.use_mu_denoiser:
            self.log_dict["lr_mu"] = float(self.optimizer_mu.param_groups[0]["lr"])
        self.log_dict["mask_hole_ratio"] = float((1 - self.mask).mean().item())
        timesteps_float = timesteps.detach().float()

        # ============ 额外诊断指标 ============
        if g_score is not None:
            with torch.no_grad():
                _mask_hole_g = (1 - self.mask).to(dtype=g_score.dtype)
                _mask_known_g = self.mask.to(dtype=g_score.dtype)
                if g_score.shape[1] != _mask_hole_g.shape[1]:
                    _mhg = _mask_hole_g.expand_as(g_score)
                    _mkg = _mask_known_g.expand_as(g_score)
                else:
                    _mhg = _mask_hole_g
                    _mkg = _mask_known_g
                self.log_dict["stats_g_score_hole_mean"] = float((g_score * _mhg).sum().item() / _mhg.sum().clamp_min(1.0).item())
                self.log_dict["stats_g_score_known_mean"] = float((g_score * _mkg).sum().item() / _mkg.sum().clamp_min(1.0).item())
                self.log_dict["stats_g_score_global_mean"] = float(g_score.mean().item())
        else:
            self.log_dict["stats_g_score_hole_mean"] = 0.0
            self.log_dict["stats_g_score_known_mean"] = 0.0
            self.log_dict["stats_g_score_global_mean"] = 0.0

        with torch.no_grad():
            self.log_dict["stats_noise_mean"] = float(noise.detach().mean().item())
            self.log_dict["stats_noise_std"] = float(noise.detach().std().item())
            self.log_dict["stats_noise_abs_max"] = float(noise.detach().abs().max().item())
            _sb = sde.sigma_bar(timesteps).to(noise.device, noise.dtype)
            _score_mag = (noise.detach().abs() / _sb.clamp_min(1e-8)).mean()
            self.log_dict["stats_score_magnitude"] = float(_score_mag.item())
            _pred_opt_diff = (yt_1_expection - yt_1_optimum).detach().abs()
            _mask_h3 = (1 - self.mask).expand_as(_pred_opt_diff)
            _mask_k3 = self.mask.expand_as(_pred_opt_diff)
            self.log_dict["stats_pred_opt_diff_hole"] = float((_pred_opt_diff * _mask_h3).sum().item() / _mask_h3.sum().clamp_min(1.0).item())
            self.log_dict["stats_pred_opt_diff_known"] = float((_pred_opt_diff * _mask_k3).sum().item() / _mask_k3.sum().clamp_min(1.0).item())

        self.log_dict["stats_timestep_mean"] = float(timesteps_float.mean().item())
        high_t_min_ratio = float(getattr(sde, "high_t_min_ratio", 0.65))
        self.log_dict["stats_timestep_high_ratio"] = float((timesteps_float >= (high_t_min_ratio * float(getattr(sde, "T", 400)))).float().mean().item())

        texture_condition_gap = ((self.condition - self.original_degraded) * self.mask).abs().sum() / self.mask.expand_as(self.condition).sum().clamp_min(1.0)
        self.log_dict["texture_condition_gap"] = float(texture_condition_gap.item())
        condition_target_gap = (self.condition - training_target * self.mask).abs().mean()
        self.log_dict["condition_target_gap"] = float(condition_target_gap.item())
        degraded_target_gap = (self.original_degraded * self.mask - training_target * self.mask).abs().mean()
        self.log_dict["degraded_target_gap"] = float(degraded_target_gap.item())

        mask_3c = self.mask.expand(-1, condition_lut_for_mu.shape[1], -1, -1)
        known_denom = mask_3c.sum().clamp_min(1.0)
        state_hole_mask = (1 - self.mask).expand_as(training_target)
        state_hole_denom = state_hole_mask.sum().clamp_min(1.0)
        state_hole = self.state.detach() * state_hole_mask
        target_hole = training_target.detach() * state_hole_mask
        cond_hole = self.condition.detach() * state_hole_mask
        state_hole_mean = state_hole.sum() / state_hole_denom
        target_hole_mean = target_hole.sum() / state_hole_denom
        cond_hole_mean = cond_hole.sum() / state_hole_denom
        state_hole_white_map = (self.state.detach().clamp(0.0, 1.0) > 0.95).all(dim=1, keepdim=True).float()
        target_hole_white_map = (training_target.detach().clamp(0.0, 1.0) > 0.95).all(dim=1, keepdim=True).float()
        hole_mask_1c = 1 - self.mask
        self.log_dict["stats_train_state_hole_mean"] = float(state_hole_mean.item())
        self.log_dict["stats_train_target_hole_mean"] = float(target_hole_mean.item())
        self.log_dict["stats_train_condition_hole_mean"] = float(cond_hole_mean.item())
        self.log_dict["stats_train_state_hole_white_ratio"] = float((state_hole_white_map * hole_mask_1c).sum().div(hole_mask_1c.sum().clamp_min(1.0)).item())
        self.log_dict["stats_train_target_hole_white_ratio"] = float((target_hole_white_map * hole_mask_1c).sum().div(hole_mask_1c.sum().clamp_min(1.0)).item())
        self.log_dict["stats_train_state_to_target_hole"] = float(((self.state.detach() - training_target.detach()).abs() * state_hole_mask).sum().div(state_hole_denom).item())
        self.log_dict["stats_train_state_to_condition_hole"] = float(((self.state.detach() - self.condition.detach()).abs() * state_hole_mask).sum().div(state_hole_denom).item())
        
        condition_lut_delta_known = ((condition_lut_for_mu - self.original_degraded).abs() * mask_3c).sum() / known_denom
        mask_hole_for_stats = 1 - self.mask
        mask_hole_3c_for_stats = mask_hole_for_stats.expand(-1, condition_lut_for_mu.shape[1], -1, -1)
        hole_denom_for_stats = mask_hole_3c_for_stats.sum().clamp_min(1.0)
        condition_lut_delta_hole = ((condition_lut_for_mu - self.original_degraded).abs() * mask_hole_3c_for_stats).sum() / hole_denom_for_stats
        training_target_delta = (training_target - self.reference_degraded).abs().mean()
        training_target_to_lut = (training_target - condition_lut_for_mu).abs().mean()
        self.log_dict["stats_condition_lut_delta_known"] = float(condition_lut_delta_known.item())
        self.log_dict["stats_condition_lut_delta_hole"] = float(condition_lut_delta_hole.item())
        self.log_dict["stats_prefill_to_lut_known"] = float(condition_lut_delta_known.item())
        self.log_dict["stats_prefill_to_lut_hole"] = float(condition_lut_delta_hole.item())
        self.log_dict["stats_training_target_delta"] = float(training_target_delta.item())
        self.log_dict["stats_training_target_to_lut"] = float(training_target_to_lut.item())

        self.log_dict.update(self._compute_condition_stats(
            color_prior=self.color_prior,
            condition_lut=condition_lut_for_mu,
            mu_clean_lut=mu_clean_lut,
            confidence=self.confidence,
            mask_known=self.mask,
        ))

        denoised_observed = getattr(self, "denoised_observed_mask_aware", None)
        self._debug_refiner_info = {
            'original_degraded': self.original_degraded.detach(),
            'reference_degraded': self.reference_degraded.detach(),
            'denoised_observed_mask_aware': denoised_observed.detach() if denoised_observed is not None else None,
            'condition_lut': condition_lut_for_mu.detach(),
            'condition_mu': self.condition.detach(),
            'mu_clean_lut': mu_clean_lut.detach(),
            'training_target': training_target.detach(),
            'training_target_lut': training_target_lut.detach(),
            'color_prior': self.color_prior.detach() if self.color_prior is not None else None,
            'confidence': self.confidence.detach() if self.confidence is not None else None,
            'mask_known': self.mask.detach(),
            'mask_hole': (1 - self.mask).detach(),
            'structure_gray_from_target': self.S_GT.detach(),
            'structure_edge_from_target': self.S_LQ.detach(),
        }
    def _masked_mean_std(self, tensor, mask):
        if tensor is None or mask is None:
            return 0.0, 0.0

        tensor = tensor.detach()
        mask = mask.detach()
        if tensor.shape[1] != mask.shape[1]:
            mask = mask.expand(-1, tensor.shape[1], -1, -1)

        denom = mask.sum().clamp_min(1.0)
        mean = (tensor * mask).sum() / denom
        var = ((tensor - mean) ** 2 * mask).sum() / denom
        std = torch.sqrt(var.clamp_min(0.0))
        return float(mean.item()), float(std.item())

    def _compute_condition_stats(self, color_prior, condition_lut, mu_clean_lut, confidence, mask_known):
        mask_hole = 1 - mask_known
        color_prior_mean, color_prior_std = self._masked_mean_std(color_prior, mask_hole)
        condition_known_mean, condition_known_std = self._masked_mean_std(condition_lut, mask_known)
        mu_known_mean, mu_known_std = self._masked_mean_std(mu_clean_lut, mask_known)
        confidence_hole_mean, _ = self._masked_mean_std(confidence, mask_hole)

        cond_lut_hole_mean, cond_lut_hole_std = self._masked_mean_std(condition_lut, mask_hole)
        cond_mu_known_mean, _ = self._masked_mean_std(self.condition, mask_known)
        cond_mu_hole_mean, _ = self._masked_mean_std(self.condition, mask_hole)
        if color_prior is not None and condition_lut is not None:
            prior_lut_gap = (color_prior - condition_lut).abs().mean(dim=1, keepdim=True)
            prior_lut_gap_hole, _ = self._masked_mean_std(prior_lut_gap, mask_hole)
        else:
            prior_lut_gap_hole = 0.0

        color_prior_white_ratio = 0.0
        if color_prior is not None:
            cp = color_prior.detach()
            white_pixels = (cp > 0.95).all(dim=1, keepdim=True).float()
            denom = mask_hole.detach().sum().clamp_min(1.0)
            color_prior_white_ratio = float((white_pixels * mask_hole.detach()).sum().item() / denom.item())

        return OrderedDict(
            [
                ("stats_color_prior_hole_mean", color_prior_mean),
                ("stats_color_prior_hole_std", color_prior_std),
                ("stats_color_prior_hole_white_ratio", color_prior_white_ratio),
                ("stats_color_prior_lut_gap_hole", prior_lut_gap_hole),
                ("stats_confidence_hole_mean", confidence_hole_mean),
                ("stats_condition_known_mean", condition_known_mean),
                ("stats_condition_known_std", condition_known_std),
                ("stats_mu_known_mean", mu_known_mean),
                ("stats_mu_known_std", mu_known_std),
                ("stats_cond_lut_hole_mean", cond_lut_hole_mean),
                ("stats_cond_lut_hole_std", cond_lut_hole_std),
                ("stats_sde_mu_known_mean", cond_mu_known_mean),
                ("stats_sde_mu_hole_mean", cond_mu_hole_mean),
            ]
        )


    def _build_lut_transformed(self, denoised_image):
        """
        自适应褪色程度的 LUT 色彩变换（训练/推理共享）。

        褪色严重的区域（低饱和度+高亮度）获得更强的 LUT 复原，
        保存较好的区域（高饱和度）只做轻微校正。

        Args:
            denoised_image: [B, 3, H, W]，已完成预去噪的输入

        Returns:
            lut_transformed: [B, 3, H, W]
            lut_confidence: [B, 1, H, W]
        """
        if self.lut_processor is None:
            return denoised_image, torch.ones_like(denoised_image[:, :1])

        lut_raw, lut_confidence = self.lut_processor.apply_to_tensor(denoised_image)

        if getattr(self, "lut_smooth_radius", 0) > 0:
            lut_raw = self._guided_smooth(
                lut_raw,
                guide=denoised_image,
                radius=self.lut_smooth_radius
            )

        lut_delta = lut_raw - denoised_image

        # 计算每像素褪色程度 fade_degree in [0, 1]
        # 高 = 低饱和度+高亮度（严重褪色）; 低 = 高饱和度（保存较好）
        with torch.no_grad():
            r = denoised_image[:, 0:1]
            g = denoised_image[:, 1:2]
            b = denoised_image[:, 2:3]
            max_rgb = torch.max(torch.max(r, g), b)
            min_rgb = torch.min(torch.min(r, g), b)
            chroma = max_rgb - min_rgb
            brightness = max_rgb
            saturation = chroma / brightness.clamp(min=0.01)
            fade_degree = ((1.0 - saturation) * brightness).clamp(0.0, 1.0)

        # 自适应强度：褪色区域获得更强的 LUT 复原
        adaptive_strength = getattr(self, "lut_strength", 1.0) * (
            1.0 + fade_degree * (getattr(self, "lut_fade_boost", 3.0) - 1.0)
        )
        effective_weight = torch.clamp(lut_confidence, 0.0, 1.0) * adaptive_strength

        lut_transformed = torch.clamp(
            denoised_image + lut_delta * effective_weight,
            0.0,
            1.0,
        )
        return lut_transformed, lut_confidence
    def _denoise_image(self, image, mask_known=None):
        """
        Lightweight edge-preserving smoothing before LUT.

        Args:
            image: [B, 3, H, W] RGB in [0, 1].
            mask_known: optional [B, 1, H, W], 1=known and 0=hole.  When
                provided, normalized convolution prevents white/black hole
                pixels from contributing to known-region smoothing.
        """
        sigma_spatial = 2.0
        kernel_size = 5
        padding = kernel_size // 2

        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - padding
        gauss_1d = torch.exp(-x**2 / (2 * sigma_spatial**2))
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
            for c in range(image.shape[1]):
                channel = image[:, c:c + 1]
                smoothed_c = F.conv2d(channel * mask_known, gauss_2d, padding=padding) / denom
                smoothed_c = torch.where(denom > 1e-5, smoothed_c, known_mean[:, c:c + 1])
                smoothed_channels.append(smoothed_c)
            smoothed = torch.cat(smoothed_channels, dim=1)
            edge_input = image * mask_known + smoothed * (1 - mask_known)
        else:
            smoothed_channels = []
            for c in range(image.shape[1]):
                channel = image[:, c:c + 1]
                smoothed_channels.append(F.conv2d(channel, gauss_2d, padding=padding))
            smoothed = torch.cat(smoothed_channels, dim=1)
            edge_input = image

        gray = 0.299 * edge_input[:, 0:1] + 0.587 * edge_input[:, 1:2] + 0.114 * edge_input[:, 2:3]
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=image.dtype, device=image.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=image.dtype, device=image.device
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        edge_weight = 1 - torch.exp(-grad_mag / 0.1)
        denoised = edge_weight * edge_input + (1 - edge_weight) * smoothed
        return torch.clamp(denoised, 0.0, 1.0)

    def _guided_smooth(self, image, guide, radius=5):
        """
        使用联合双边滤波平滑图像，以 guide 为引导保持边缘
        
        这可以减少 LUT 变换后的颜色割裂，让相似颜色区域获得更一致的变换结果
        
        Args:
            image: [B, 3, H, W] 需要平滑的图像（LUT 变换结果）
            guide: [B, 3, H, W] 引导图像（去噪后的原图）
            radius: 滤波半径
        
        Returns:
            smoothed: [B, 3, H, W] 平滑后的图像
        """
        import cv2
        
        B, C, H, W = image.shape
        results = []
        
        for b in range(B):
            # 转换为 numpy [H, W, 3]
            img_np = image[b].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            guide_np = guide[b].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            
            # 转换为 uint8
            img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            guide_uint8 = (np.clip(guide_np, 0, 1) * 255).astype(np.uint8)
            
            # 联合双边滤波
            # sigma_color: 颜色空间滤波的 sigma，较大值意味着更多颜色混合
            # sigma_space: 坐标空间滤波的 sigma
            d = radius * 2 + 1
            sigma_color = 50  # 颜色相似度阈值
            sigma_space = radius  # 空间距离 sigma
            
            # 使用 guide 作为参考进行联合双边滤波
            # OpenCV 没有直接的 joint bilateral filter，我们分通道处理
            smoothed = np.zeros_like(img_uint8, dtype=np.float32)
            for c in range(3):
                # 使用 guide 的灰度作为边缘参考
                guide_gray = cv2.cvtColor(guide_uint8, cv2.COLOR_RGB2GRAY)
                
                # 双边滤波
                filtered = cv2.bilateralFilter(
                    img_uint8[:, :, c], 
                    d=d, 
                    sigmaColor=sigma_color, 
                    sigmaSpace=sigma_space
                )
                smoothed[:, :, c] = filtered.astype(np.float32) / 255.0
            
            # 转回 tensor
            smoothed_tensor = torch.from_numpy(smoothed).permute(2, 0, 1).to(image.device)
            results.append(smoothed_tensor)
        
        return torch.stack(results, dim=0)

    
    
    def test(self, sde=None, save_states=False, save_dir='save_dir', GT = None, mask = None, S_sde = None, S_GT = None, S_LQ = None, structure_guide = None):
        sde.set_mu(self.condition)
        S_sde.set_mu(self.S_LQ)
        self.model.eval()
        self.models.eval()
        with torch.no_grad():
            self.output = sde.reverse_sde(self.state, save_states=save_states, save_dir=save_dir, GT = GT, mask = mask, S_sde = S_sde, S_GT = S_GT, S_LQ = S_LQ, structure_guide = None)

        self.model.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_training_debug(self):
        return getattr(self, "_debug_refiner_info", None)

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict["Input"] = self.condition.detach()[0].float().cpu()
        out_dict["Output"] = self.output.detach()[0].float().cpu()
        if need_GT:
            out_dict["GT"] = self.state_0.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.model)
        if isinstance(self.model, nn.DataParallel) or isinstance(
            self.model, DistributedDataParallel
        ):
            net_struc_str = "{} - {}".format(
                self.model.__class__.__name__, self.model.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.model.__class__.__name__)
        if self.rank <= 0:
            logger.info(
                "Network G structure: {}, with parameters: {:,d}".format(
                    net_struc_str, n
                )
            )
            logger.info(s)

    def load(self):
        load_path_G = self.opt["path"]["pretrain_model_G"]
        if load_path_G is not None:
            print('load-------------------------------')
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            
            # 加载完整 checkpoint（可能包含 mu_denoiser）
            checkpoint = torch.load(load_path_G, map_location=self.device)
            
            # 分离主模型和 D_mu 的权重
            model_state = {}
            mu_denoiser_state = {}
            
            for k, v in checkpoint.items():
                if k.startswith('mu_denoiser.'):
                    # D_mu 权重（去掉前缀）
                    mu_denoiser_state[k[len('mu_denoiser.'):]] = v
                elif k.startswith('module.mu_denoiser.'):
                    mu_denoiser_state[k[len('module.mu_denoiser.'):]] = v
                elif k.startswith('module.'):
                    model_state[k[7:]] = v
                else:
                    model_state[k] = v
            
            # 加载主模型
            self.load_network_from_state(model_state, self.model, self.opt["path"]["strict_load"])
            self._maybe_bootstrap_brushnet_from_trunk()
            
            # 加载 D_mu（如果有）
            use_mu_denoiser = getattr(self, 'use_mu_denoiser', False)
            mu_denoiser = getattr(self, 'mu_denoiser', None)
            if use_mu_denoiser and mu_denoiser is not None and len(mu_denoiser_state) > 0:
                try:
                    mu_denoiser.load_state_dict(mu_denoiser_state, strict=False)
                    self.mu_denoiser_has_weights = True
                    self.mu_denoiser_loaded_weights = True
                    logger.info(f"[Model] Mu-Denoiser 权重已加载 ({len(mu_denoiser_state)} 个参数)")
                except Exception as e:
                    logger.warning(f"[Model] Mu-Denoiser 权重加载失败: {e}")
            elif use_mu_denoiser and len(mu_denoiser_state) == 0:
                logger.info("[Model] Checkpoint 中未找到 Mu-Denoiser 权重，将从头训练")
    
    def load_network_from_state(self, state_dict, network, strict=True):
        """从 state_dict 加载网络（辅助方法）"""
        from torch.nn.parallel import DataParallel, DistributedDataParallel
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        param_names = {name for name, _ in network.named_parameters()}
        incompatible = network.load_state_dict(state_dict, strict=strict)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        missing_set = set(missing)
        loaded_param_names = {
            name for name in param_names if name in state_dict and name not in missing_set
        }
        model_ref = self.model.module if isinstance(self.model, (DataParallel, DistributedDataParallel)) else self.model
        if network is model_ref:
            self.loaded_model_param_names = loaded_param_names
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

    def _maybe_bootstrap_brushnet_from_trunk(self):
        brushnet_opt = self.opt.get("brushnet", {})
        if not bool(brushnet_opt.get("bootstrap_from_pretrained_trunk", False)):
            return

        model_ref = self.model.module if isinstance(self.model, (DataParallel, DistributedDataParallel)) else self.model
        bootstrap_fn = getattr(model_ref, "bootstrap_brushnet_from_trunk", None)
        if bootstrap_fn is None:
            return

        loaded_names = set(getattr(self, "loaded_model_param_names", set()) or set())
        brushnet_core_prefixes = (
            "brushnet.init_conv.",
            "brushnet.time_mlp.",
            "brushnet.downs.",
            "brushnet.mid_block1.",
            "brushnet.mid_attn.",
            "brushnet.mid_block2.",
        )
        has_brushnet_weights = any(
            any(name.startswith(prefix) for prefix in brushnet_core_prefixes)
            for name in loaded_names
        )
        if has_brushnet_weights:
            logger.info(
                "[BrushNetInit] skip trunk bootstrap because checkpoint already contains BrushNet encoder weights"
            )
            return

        bootstrap_fn(
            reset_zero_convs=bool(brushnet_opt.get("bootstrap_reset_zero_convs", True))
        )
        logger.info(
            "[BrushNetInit] initialized BrushNet encoder from pretrained trunk (official BrushNet-style from_unet)"
        )

    def save(self, iter_label):
        """
        保存模型权重（包含主模型 + D_mu）
        
        权重文件结构:
        {
            'conv1.weight': ...,       # 主模型参数
            'conv1.bias': ...,
            ...
            'mu_denoiser.stem.weight': ...,  # D_mu 参数（带前缀）
            'mu_denoiser.stem.bias': ...,
            ...
        }
        """
        import os
        from torch.nn.parallel import DataParallel, DistributedDataParallel
        
        save_filename = "{}_{}.pth".format(iter_label, "G")
        save_path = os.path.join(self.opt["path"]["models"], save_filename)
        
        # 获取主模型 state_dict
        model = self.model
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model = model.module
        combined_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # 添加 D_mu state_dict（带前缀，静默添加不打印）
        if self.use_mu_denoiser and self.mu_denoiser is not None:
            for k, v in self.mu_denoiser.state_dict().items():
                combined_state[f'mu_denoiser.{k}'] = v.cpu()
            #logger.info(f"[Model] 保存权重包含 Mu-Denoiser ({sum(1 for k in combined_state if k.startswith('mu_denoiser.'))} 个参数)")
        
        torch.save(combined_state, save_path)
        #logger.info(f"[Model] 模型已保存到 {save_path}")
