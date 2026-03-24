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
        

        # define network and load pretrained models
        self.model, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.dis = self.dis.to(self.device)
        
        #必改
        gpu_ids = opt.get('gpu_ids', None)
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids, output_device=gpu_ids[0])
            self.dis   = DataParallel(self.dis,   device_ids=gpu_ids, output_device=gpu_ids[0])

        self.load()
        
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
        
        # 获取 LUT 相关配置
        self.gt_mode = train_dataset_opt.get('gt_mode', 'full')
        self.lut_strength = train_dataset_opt.get('lut_strength', 1.0)  # LUT 强度
        self.lut_smooth_radius = train_dataset_opt.get('lut_smooth_radius', 0)  # 平滑半径
        logger.info(f"[Model] GT 模式: {self.gt_mode}, LUT 强度: {self.lut_strength}, 平滑半径: {self.lut_smooth_radius}")
        
        # ============ 初始化 Self-Supervised Mu-Denoiser ============
        mu_denoiser_opt = opt.get('mu_denoiser', {})
        self.mu_denoiser_opt = mu_denoiser_opt
        self.use_mu_denoiser = mu_denoiser_opt.get('enabled', False) and HAS_MU_DENOISER
        
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
        
        if self.is_train:
            self.model.train()
            self.dis.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss_tri = nn.TripletMarginLoss().to(self.device)
            self.adversarial_loss = AdversarialLoss(type = 'hinge').to(self.device)
            self.weight = opt['train']['weight']

            # optimizers
            self.optimizer_d = torch.optim.Adam(self.dis.parameters(), lr = 1e-4, betas = (0.5, 0.99))#1e-4
            
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0
            optim_params = []
            for (k,v,) in self.model.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning("Params [{:s}] will not optimize.".format(k))


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
                  original_degraded=None, reference_degraded=None):
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

    def compute_mu_clean_no_grad(self, y_degraded, mask_known, confidence=None):
        """
        保留该辅助接口供调试/对比使用，但不再驱动 texture SDE 主干。

        Texture 主干的 mu 已恢复为原版 StrDiffusion 语义：
        ``observed_degraded * mask_known``。
        """
        if not self.use_mu_denoiser or self.mu_denoiser is None:
            return y_degraded * mask_known

        with torch.no_grad():
            mu_clean = self.mu_denoiser_trainer.inference(
                y_degraded, mask_known, confidence
            )
            mu_clean = mu_clean * mask_known

        return mu_clean

    def optimize_parameters(self, step, timesteps, sde=None):
        self.log_dict = OrderedDict()

        # ============ 第一阶段：分离“条件链输入”和“训练目标输入” ============
        # original_degraded  : 当前观测输入（真实缺损外观）
        # reference_degraded : 完整褪色参考图（只用于生成训练目标）
        with torch.no_grad():
            denoised_original = self._denoise_image(self.original_degraded)
            if self.reference_degraded.data_ptr() == self.original_degraded.data_ptr():
                denoised_reference = denoised_original
            else:
                denoised_reference = self._denoise_image(self.reference_degraded)

            # 条件链：严格模拟推理时的真实缺损输入
            lut_transformed, lut_confidence = self._build_lut_transformed(denoised_original)

            # 训练目标：始终来自完整参考图，避免把真实缺损输入错误写进监督信号
            target_lut_transformed, _ = self._build_lut_transformed(denoised_reference)
            if self.gt_mode == 'full':
                color_changed = target_lut_transformed
            else:
                color_changed = denoised_reference * self.mask + target_lut_transformed * (1 - self.mask)
        
        # 使用颜色变换后的图像作为训练目标（第二阶段的GT）
        training_target = color_changed
        
        # ============ 重要：同步更新 color_prior 的非 mask 区域 ============
        # 使用“真实缺损输入”对应的 LUT 结果更新已知区域，使训练和推理一致
        if self.color_prior is not None:
            self.color_prior = (
                lut_transformed * self.mask +
                self.color_prior * (1 - self.mask)
            )
        
        # ============ Self-Supervised Mu-Denoiser 训练 ============
        # 在 SDE 训练前，对 mu 进行自监督去噪
        mu_denoiser_loss = None
        if self.use_mu_denoiser and self.is_train:
            # 使用真实缺损输入进行去噪训练，和推理保持一致
            # 注意：self.mask 已经是 SDE 语义 (1=known, 0=hole)
            y_hat, loss_mu, mu_losses = self.mu_denoiser_trainer.train_step(
                y_degraded=self.original_degraded,
                mask_known=self.mask,
                confidence=self.confidence,
                lambda_ss=self.lambda_ss,
                lambda_tv=self.lambda_tv,
            )
            
            # CRITICAL: detach 防止扩散梯度影响 D_mu
            # mu_clean 只保留已知区域（和原始 mu = Y_degraded * mask 语义一致）
            mu_clean = (y_hat.detach()) * self.mask
            mu_denoiser_loss = loss_mu
            
            # 记录 D_mu 损失
            for key, val in mu_losses.items():
                self.log_dict[key] = val
        else:
            # 回退到原始行为
            mu_clean = self.condition
        
        # Texture 主干恢复到原版 StrDiffusion 语义：
        # cond/mu 必须始终等于 masked observed input，而不是辅助分支的 mu_clean。
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

        noise, _ = sde.noise_fn(self.state, timesteps.squeeze(), S_optimum, **brushnet_kwargs)
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
        
        # 总损失 = 扩散损失 + Mu-Denoiser 损失（如果有）
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

        # set log
        self.log_dict["loss"] = loss.item()
        self.log_dict["loss_main"] = loss.item()
        self.log_dict["loss_total"] = total_loss.item()
        self.log_dict["loss_known"] = loss_components["loss_known"].item()
        self.log_dict["loss_hole"] = loss_components["loss_hole"].item()
        self.log_dict["loss_hole_weighted"] = loss_components["loss_hole_weighted"].item()
        self.log_dict["loss_mu_total"] = float(mu_denoiser_loss.item()) if mu_denoiser_loss is not None else 0.0
        self.log_dict["loss_mu_ss"] = float(mu_losses.get("l_ss", 0.0)) if self.use_mu_denoiser and self.is_train else 0.0
        self.log_dict["loss_mu_tv"] = float(mu_losses.get("l_tv", 0.0)) if self.use_mu_denoiser and self.is_train else 0.0
        self.log_dict["lr_main"] = float(self.optimizer.param_groups[0]["lr"])
        if self.use_mu_denoiser:
            self.log_dict["lr_mu"] = float(self.optimizer_mu.param_groups[0]["lr"])
        self.log_dict["mask_hole_ratio"] = float((1 - self.mask).mean().item())
        texture_condition_gap = (self.condition - self.original_degraded * self.mask).abs().mean()
        self.log_dict["texture_condition_gap"] = float(texture_condition_gap.item())
        self.log_dict.update(self._compute_condition_stats(
            color_prior=self.color_prior,
            lut_transformed=lut_transformed,
            mu_clean=mu_clean,
            mask_known=self.mask,
        ))
        
        # ============ 保存调试信息 ============
        # 调试顺序：
        # 1. Input: 原始褪色图（未变色，未去噪）
        # 2. Denoised: 去噪后的原图
        # 3. ColorChanged: LUT颜色变换后的图像（训练目标）
        # 4. Prior: 颜色先验
        # 5. Original+Mask: 原图 + mask 涂黑
        # 6. Mask
        self._debug_refiner_info = {
            'original_degraded': self.original_degraded.detach(),   # 当前观测输入（真实缺损外观）
            'reference_degraded': self.reference_degraded.detach(), # 完整参考图
            'denoised_original': denoised_original.detach(),        # 条件链去噪结果
            'mu_clean': mu_clean.detach() if mu_clean is not self.condition else None,
            'lut_transformed': lut_transformed.detach(),            # 条件链 LUT 结果
            'color_changed': color_changed.detach(),                # 训练目标
            'color_prior': self.color_prior.detach() if self.color_prior is not None else None,
            'original_with_mask': self.condition.detach(),
            'mask': self.mask.detach(),
            'mask_known': self.mask.detach(),
            'mask_hole': (1 - self.mask).detach(),
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

    def _compute_condition_stats(self, color_prior, lut_transformed, mu_clean, mask_known):
        mask_hole = 1 - mask_known
        color_prior_mean, color_prior_std = self._masked_mean_std(color_prior, mask_hole)
        lut_hole_mean, _ = self._masked_mean_std(lut_transformed, mask_hole)
        mu_known_mean, mu_known_std = self._masked_mean_std(mu_clean, mask_known)
        return OrderedDict(
            [
                ("stats_color_prior_hole_mean", color_prior_mean),
                ("stats_color_prior_hole_std", color_prior_std),
                ("stats_lut_hole_mean", lut_hole_mean),
                ("stats_mu_known_mean", mu_known_mean),
                ("stats_mu_known_std", mu_known_std),
            ]
        )

    def _build_lut_transformed(self, denoised_image):
        """
        对输入图像执行训练/推理共享的 LUT 后处理。

        Args:
            denoised_image: [B, 3, H, W]，已经完成预去噪的输入

        Returns:
            lut_transformed: [B, 3, H, W]
            lut_confidence: [B, 1, H, W]
        """
        if self.lut_processor is None:
            return denoised_image, torch.ones_like(denoised_image[:, :1])

        lut_transformed, lut_confidence = self.lut_processor.apply_to_tensor(denoised_image)

        if self.lut_smooth_radius > 0:
            lut_transformed = self._guided_smooth(
                lut_transformed,
                guide=denoised_image,
                radius=self.lut_smooth_radius
            )

        effective_weight = self.lut_strength * lut_confidence
        lut_transformed = (
            denoised_image * (1 - effective_weight) +
            lut_transformed * effective_weight
        )
        return lut_transformed, lut_confidence
    
    def _denoise_image(self, image):
        """
        使用双边滤波对图像进行边缘保持去噪
        
        这是一个固定算法（不需要训练），在平坦区域平滑颜色，在边缘保持清晰
        
        Args:
            image: [B, 3, H, W] RGB 图像 in [0, 1]
        
        Returns:
            denoised: [B, 3, H, W] 去噪后的 RGB 图像 in [0, 1]
        """
        # 使用简单的高斯模糊作为去噪（可调整参数）
        # 这是一个近似的边缘保持滤波
        sigma_spatial = 2.0  # 空间平滑程度
        kernel_size = 5
        
        # 创建高斯核
        x = torch.arange(kernel_size, dtype=image.dtype, device=image.device) - kernel_size // 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma_spatial**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d.view(1, 1, kernel_size, kernel_size)
        
        # 对每个通道分别进行高斯平滑
        padding = kernel_size // 2
        smoothed = []
        for c in range(3):
            channel = image[:, c:c+1, :, :]
            channel_smoothed = F.conv2d(channel, gauss_2d, padding=padding)
            smoothed.append(channel_smoothed)
        smoothed = torch.cat(smoothed, dim=1)
        
        # 计算边缘权重（高梯度区域保持原值）
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        # Sobel 梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 边缘权重：梯度大的地方保持原值
        sigma_edge = 0.1
        edge_weight = 1 - torch.exp(-grad_mag / sigma_edge)  # 边缘=1, 平坦=0
        
        # 混合：边缘区域保持原值，平坦区域使用平滑值
        denoised = edge_weight * image + (1 - edge_weight) * smoothed
        
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
            
            # 加载 D_mu（如果有）
            use_mu_denoiser = getattr(self, 'use_mu_denoiser', False)
            mu_denoiser = getattr(self, 'mu_denoiser', None)
            if use_mu_denoiser and mu_denoiser is not None and len(mu_denoiser_state) > 0:
                try:
                    mu_denoiser.load_state_dict(mu_denoiser_state, strict=False)
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
        network.load_state_dict(state_dict, strict=strict)

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
