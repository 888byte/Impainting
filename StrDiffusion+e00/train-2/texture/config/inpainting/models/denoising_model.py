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
                # 不加入 self.optimizers，因为我们单独管理它
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


    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ, 
                  color_prior=None, confidence=None, conf_lut=None, original_degraded=None):
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
            original_degraded: [可选] 原始褪色图（无 mask 涂黑），用于去噪
        """
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)   # LQ（可能带 mask 涂黑）
        self.state_0 = GT.to(self.device)     # GT
        self.mask = mask.to(self.device)      # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)
        
        # 原始褪色图（用于第一阶段去噪）
        if original_degraded is not None:
            self.original_degraded = original_degraded.to(self.device)
        else:
            # 如果没有提供，回退到 condition
            self.original_degraded = self.condition
        
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
        计算干净的 mu（无梯度版本），用于 train.py 的 generate_random_states
        
        CRITICAL: 确保 generate_random_states 和 optimize_parameters 用同一个 mu_clean
        
        Args:
            y_degraded: [B, 3, H, W] 原始褪色图
            mask_known: [B, 1, H, W] SDE mask (1=known, 0=hole)
            confidence: [B, 1, H, W] 可选置信度
        
        Returns:
            mu_clean: [B, 3, H, W] 去噪后的 mu (已乘 mask)
        """
        if not self.use_mu_denoiser or self.mu_denoiser is None:
            # 回退到原始行为
            return y_degraded * mask_known
        
        with torch.no_grad():
            mu_clean = self.mu_denoiser_trainer.inference(
                y_degraded, mask_known, confidence
            )
            # mu 只保留已知区域（和原始 mu 语义一致）
            mu_clean = mu_clean * mask_known
        
        return mu_clean

    def optimize_parameters(self, step, timesteps, sde=None):
        # ============ 第一阶段：原始褪色图去噪 ============
        # 对原始褪色图（original_degraded）进行边缘保持去噪
        # 去噪是所有后续操作的基础
        with torch.no_grad():
            denoised_original = self._denoise_image(self.original_degraded)
        
        # ============ 第二阶段：对去噪后的图像应用LUT颜色变换 ============
        # 根据配置的 gt_mode 决定变换方式：
        # - full: 全图LUT变换
        # - partial: 仅mask区域LUT变换，非mask区域保持去噪后的原色
        with torch.no_grad():
            if self.lut_processor is not None:
                # 应用 LUT 变换，获取颜色映射和置信度
                lut_transformed, lut_confidence = self.lut_processor.apply_to_tensor(denoised_original)
                # lut_confidence: [B, 1, H, W] 每个像素的LUT置信度 (0-1)
                
                # ============ 平滑处理（导向滤波）============
                # 以去噪后的原图为引导，平滑 LUT 结果，减少颜色割裂
                if self.lut_smooth_radius > 0:
                    lut_transformed = self._guided_smooth(
                        lut_transformed, 
                        guide=denoised_original, 
                        radius=self.lut_smooth_radius
                    )
                
                # ============ 置信度加权 LUT 混合 ============
                # 使用 LUT 置信度进行逐像素加权混合：
                # - 高置信度像素：使用更多 LUT 颜色
                # - 低置信度像素：保持更多原色
                # 最终权重 = lut_strength * lut_confidence
                effective_weight = self.lut_strength * lut_confidence  # [B, 1, H, W]
                
                # 逐像素混合：原色 * (1 - weight) + LUT * weight
                lut_transformed = (
                    denoised_original * (1 - effective_weight) + 
                    lut_transformed * effective_weight
                )
                
                if self.gt_mode == 'full':
                    # 全图 LUT 变换
                    color_changed = lut_transformed
                else:  # partial 或其他模式
                    # 仅 mask 区域 LUT 变换
                    # mask: 1=已知区域, 0=缺失区域
                    # 缺失区域（mask=0）使用 LUT 变换，已知区域保持去噪原色
                    color_changed = denoised_original * self.mask + lut_transformed * (1 - self.mask)
            else:
                # 没有 LUT 处理器，直接使用去噪后的图像
                color_changed = denoised_original
                lut_transformed = denoised_original
                lut_confidence = torch.ones_like(denoised_original[:, :1])
        
        # 使用颜色变换后的图像作为训练目标（第二阶段的GT）
        training_target = color_changed
        
        # ============ Self-Supervised Mu-Denoiser 训练 ============
        # 在 SDE 训练前，对 mu 进行自监督去噪
        mu_denoiser_loss = None
        if self.use_mu_denoiser and self.is_train:
            # 使用 original_degraded（未涂黑的原始图）进行去噪
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
        
        # 更新 SDE 的条件（用 mu_clean 替换原始的 self.condition）
        sde.set_mu(mu_clean)
        
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
        loss = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask)
        
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
        
        # ============ 保存调试信息 ============
        # 调试顺序：
        # 1. Input: 原始褪色图（未变色，未去噪）
        # 2. Denoised: 去噪后的原图
        # 3. ColorChanged: LUT颜色变换后的图像（训练目标）
        # 4. Prior: 颜色先验
        # 5. Original+Mask: 原图 + mask 涂黑
        # 6. Mask
        self._debug_refiner_info = {
            'original_degraded': self.original_degraded.detach(),  # 原始褪色图
            'denoised_original': denoised_original.detach(),       # 双边滤波去噪后的原图
            'mu_clean': mu_clean.detach() if mu_clean is not self.condition else None,  # D_mu 去噪的 mu
            'color_changed': color_changed.detach(),               # LUT变换后（训练目标）
            'color_prior': self.color_prior.detach() if self.color_prior is not None else None,
            'original_with_mask': (self.original_degraded * self.mask).detach(),  # 原图+mask涂黑
            'mask': self.mask.detach(),
        }
    
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
            self.load_network(load_path_G, self.model, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        
