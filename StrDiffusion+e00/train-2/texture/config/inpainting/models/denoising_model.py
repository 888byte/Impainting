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

# ============ ChromaRefiner 支持 ============
try:
    from .chroma_refiner import create_chroma_refiner
    HAS_CHROMA_REFINER = True
except ImportError:
    HAS_CHROMA_REFINER = False
    print("[Warning] ChromaRefiner 未找到，色度精炼功能不可用")
# ============ ChromaRefiner 支持 ============

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
        
        # ============ ChromaRefiner 初始化 ============
        self.chroma_refiner = None
        self.refiner_opt = opt.get('chroma_refiner', {})
        if HAS_CHROMA_REFINER and self.refiner_opt.get('enabled', False):
            self.chroma_refiner = create_chroma_refiner(opt)
            if self.chroma_refiner is not None:
                self.chroma_refiner = self.chroma_refiner.to(self.device)
                logger.info(f"[ChromaRefiner] 已初始化: in_ch={self.refiner_opt.get('in_channels', 6)}, "
                           f"hidden_ch={self.refiner_opt.get('hidden_channels', 32)}, "
                           f"num_blocks={self.refiner_opt.get('num_blocks', 1)}")
        # ============ ChromaRefiner 初始化完成 ============
        
        if self.is_train:
            self.model.train()
            self.dis.train()
            if self.chroma_refiner is not None:
                self.chroma_refiner.train()

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
            
            # ============ ChromaRefiner 优化器 ============
            if self.chroma_refiner is not None:
                refiner_lr = self.refiner_opt.get('lr', train_opt["lr_G"])
                self.optimizer_refiner = torch.optim.Adam(
                    self.chroma_refiner.parameters(),
                    lr=refiner_lr,
                    weight_decay=wd_G,
                    betas=(train_opt["beta1"], train_opt["beta2"]),
                )
                self.optimizers.append(self.optimizer_refiner)
                # Refiner loss 权重配置
                self.lambda_ref = self.refiner_opt.get('lambda_ref', 0.1)
                self.lambda_ab = self.refiner_opt.get('lambda_ab', 1.0)
                self.lambda_keep = self.refiner_opt.get('lambda_keep', 0.1)
                self.lambda_tv = self.refiner_opt.get('lambda_tv', 0.01)
                self.refiner_gamma = self.refiner_opt.get('gamma', 1.0)
                logger.info(f"[ChromaRefiner] 优化器已配置: lr={refiner_lr}, lambda_ref={self.lambda_ref}")
            # ============ ChromaRefiner 优化器完成 ============

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


    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ, color_prior=None, confidence=None, conf_lut=None):
        """
        加载训练数据
        
        Args:
            state: 噪声状态
            LQ: 低质量输入（条件）
            GT: Ground Truth
            mask: 掩码
            S_sde: 结构SDE
            S_GT: 结构GT
            S_LQ: 结构LQ
            color_prior: [可选] 颜色先验图，用于BrushNet
            confidence: [可选] 置信度图，用于BrushNet
            conf_lut: [可选] LUT置信度图，用于ChromaRefiner
        """
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        #if GT is not None: 
        self.state_0 = GT.to(self.device)  # GT
        self.mask = mask.to(self.device) # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)
        
        # BrushNet条件（新增）
        if color_prior is not None:
            self.color_prior = color_prior.to(self.device)
        else:
            self.color_prior = None
            
        if confidence is not None:
            self.confidence = confidence.to(self.device)
        else:
            self.confidence = None
        
        # ChromaRefiner 额外输入：LUT置信度
        if conf_lut is not None:
            self.conf_lut = conf_lut.to(self.device)
        else:
            self.conf_lut = None



    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        
        # ============ ChromaRefiner 精炼 color_prior ============
        refined_color_prior = self.color_prior
        loss_refiner = None
        
        if (self.chroma_refiner is not None and 
            self.color_prior is not None and 
            self.confidence is not None):
            
            # 计算 mask_hole (1=hole, BrushNet 约定)
            mask_hole = 1 - self.mask  # self.mask 是 SDE 约定 (1=known)
            
            # 精炼 color_prior
            refined_color_prior, delta_update, gate, delta_ab_norm, ab_img_norm = self.refine_prior(
                img=self.condition,  # Y_degraded * mask_for_sde
                prior_base=self.color_prior,
                conf=self.confidence,
                mask_hole=mask_hole,
                conf_lut=self.conf_lut if hasattr(self, 'conf_lut') else None
            )
            
            # 计算 Refiner loss (独立监督)
            loss_refiner = self._compute_refiner_loss(
                delta_update=delta_update,
                gate=gate,
                delta_ab_norm=delta_ab_norm,
                ab_img_norm=ab_img_norm,
                y_gt=self.state_0,
                conf=self.confidence
            )
            
            # 保存调试信息（用于验证 refined_color_prior 是否正确传递）
            self._debug_refiner_info = {
                'original_prior': self.color_prior.detach(),
                'refined_prior': refined_color_prior.detach(),
                'delta_update': delta_update.detach(),
                'gate': gate.detach(),
            }
        # ============ ChromaRefiner 完成 ============
        
        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        timesteps = timesteps.to(self.device)
        # Get noise and score
        S_timestep, S_optimum = self.S_sde.generate_random_states_texture(x0=self.S_GT, mu=self.S_LQ * self.mask, timesteps = timesteps)
        S_optimum = self.S_sde.reverse_optimum_step(S_optimum, self.S_GT, timesteps)
        
        # ============ 传递BrushNet条件（使用精炼后的 prior）============
        # 将color_prior, confidence, mask传递给网络
        # 
        # 注意：mask约定问题！
        # - self.mask 是 SDE 约定: 1=已知, 0=缺失
        # - BrushNet 期望约定: 1=需要修复, 0=已知
        # 因此需要取反！
        #
        brushnet_kwargs = {}
        if refined_color_prior is not None:
            brushnet_kwargs['color_prior'] = refined_color_prior
        if self.confidence is not None:
            brushnet_kwargs['confidence'] = self.confidence
        if hasattr(self, 'mask') and self.mask is not None:
            # 关键修复：取反mask给BrushNet（1=需要修复）
            brushnet_kwargs['mask'] = 1 - self.mask
        
        noise, _ = sde.noise_fn(self.state, timesteps.squeeze(), S_optimum, **brushnet_kwargs)
        # ============ 传递BrushNet条件完成 ============
        
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        
        # ============ 优化器更新 ============
        self.optimizer.zero_grad()
        if self.chroma_refiner is not None and hasattr(self, 'optimizer_refiner'):
            self.optimizer_refiner.zero_grad()
        
        # 主损失
        loss_main = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask)
        
        # 总损失 = 主损失 + Refiner损失
        loss = loss_main
        if loss_refiner is not None:
            loss = loss + self.lambda_ref * loss_refiner
        
        loss.backward()
        self.optimizer.step()
        if self.chroma_refiner is not None and hasattr(self, 'optimizer_refiner'):
            self.optimizer_refiner.step()
        

        # set log
        self.log_dict["loss"] = loss.item()
        self.log_dict["loss_main"] = loss_main.item()
        if loss_refiner is not None:
            self.log_dict["loss_refiner"] = loss_refiner.item()
    
    def refine_prior(self, img, prior_base, conf, mask_hole, conf_lut=None):
        """
        在线精炼 color_prior (Lab 空间)
        
        Args:
            img: Y_degraded [B,3,H,W] in [0,1]
            prior_base: color_prior [B,3,H,W] in [0,1]
            conf: confidence [B,1,H,W] in [0,1]
            mask_hole: [B,1,H,W] (1=hole)
            conf_lut: [可选] [B,1,H,W] LUT置信度 in [0,1]
        
        Returns:
            prior_refined: [B,3,H,W] in [0,1]
            delta_update_norm: [B,2,H,W] (用于 loss 计算)
            gate: [B,1,H,W] 门控系数
            delta_ab_norm: [B,2,H,W] 原始色度差
            ab_img_norm: [B,2,H,W] 原图ab归一化
        """
        # Step 1: RGB → Lab (手动实现，避免依赖 kornia)
        lab_img = self._rgb_to_lab(img)      # L:0~100, ab:-128~127
        lab_prior = self._rgb_to_lab(prior_base)
        
        # Step 2: 归一化
        L_norm = lab_img[:, 0:1, :, :] / 100.0
        ab_img_norm = lab_img[:, 1:3, :, :] / 128.0
        ab_prior_norm = lab_prior[:, 1:3, :, :] / 128.0
        delta_ab_norm = ab_prior_norm - ab_img_norm
        
        # Step 3: 构造 Refiner 输入 (6 通道)
        # delta_ab(2) + L(1) + conf(1) + mask(1) + conf_lut(1)
        if conf_lut is not None:
            ref_in = torch.cat([delta_ab_norm, L_norm, conf, mask_hole, conf_lut], dim=1)
        else:
            # 如果没有 conf_lut，用 conf 填充
            ref_in = torch.cat([delta_ab_norm, L_norm, conf, mask_hole, conf], dim=1)
        
        # Step 4: Refiner 前向
        delta_update_norm = self.chroma_refiner(ref_in)
        
        # Step 5: 门控与限幅
        gamma = self.refiner_gamma if hasattr(self, 'refiner_gamma') else 1.0
        gate = ((1.0 - conf) ** gamma) * mask_hole
        delta_final_norm = delta_ab_norm + gate * delta_update_norm
        
        # Step 6: 合成 refined Lab
        ab_refined_norm = ab_img_norm + delta_final_norm
        L_refined = lab_img[:, 0:1, :, :]  # 保持原亮度
        lab_refined = torch.cat([L_refined, ab_refined_norm * 128.0], dim=1)
        
        # Step 7: Lab → RGB
        prior_refined = self._lab_to_rgb(lab_refined)
        prior_refined = torch.clamp(prior_refined, 0.0, 1.0)
        
        return prior_refined, delta_update_norm, gate, delta_ab_norm, ab_img_norm
    
    def _compute_refiner_loss(self, delta_update, gate, delta_ab_norm, ab_img_norm, y_gt, conf):
        """
        计算 Refiner 的独立监督 loss
        
        L_ref = λ1*L_ab + λ2*L_keep + λ3*L_tv
        """
        # L_ab: 颜色监督 (只监督 ab)
        lab_gt = self._rgb_to_lab(y_gt)
        ab_gt_norm = lab_gt[:, 1:3, :, :] / 128.0
        
        # refined ab = img ab + delta_ab + gate * update
        ab_refined_norm = ab_img_norm + delta_ab_norm + gate * delta_update
        
        # 加权颜色损失 (gate 区域重点监督)
        L_ab = (torch.abs(ab_refined_norm - ab_gt_norm) * gate).sum() / (gate.sum() + 1e-8)
        
        # L_keep: 高置信区域保守项 (2通道分别计算)
        L_keep = (torch.abs(delta_update) * conf).mean()
        
        # L_tv: TV 平滑项
        L_tv = self._total_variation(delta_update)
        
        # 加权求和
        lambda_ab = self.lambda_ab if hasattr(self, 'lambda_ab') else 1.0
        lambda_keep = self.lambda_keep if hasattr(self, 'lambda_keep') else 0.1
        lambda_tv = self.lambda_tv if hasattr(self, 'lambda_tv') else 0.01
        
        return lambda_ab * L_ab + lambda_keep * L_keep + lambda_tv * L_tv
    
    def _total_variation(self, x):
        """计算 Total Variation loss"""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return diff_h + diff_w
    
    def _rgb_to_lab(self, rgb):
        """
        RGB [0,1] → Lab (L:0~100, ab:-128~127)
        使用 D65 白点
        """
        # RGB → XYZ
        mask = rgb > 0.04045
        rgb_linear = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        
        # sRGB to XYZ matrix
        r, g, b = rgb_linear[:, 0:1], rgb_linear[:, 1:2], rgb_linear[:, 2:3]
        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
        
        # Normalize by D65 white point
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        # XYZ → Lab
        epsilon = 0.008856
        kappa = 903.3
        
        fx = torch.where(x > epsilon, x ** (1/3), (kappa * x + 16) / 116)
        fy = torch.where(y > epsilon, y ** (1/3), (kappa * y + 16) / 116)
        fz = torch.where(z > epsilon, z ** (1/3), (kappa * z + 16) / 116)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b_ch = 200 * (fy - fz)
        
        return torch.cat([L, a, b_ch], dim=1)
    
    def _lab_to_rgb(self, lab):
        """
        Lab (L:0~100, ab:-128~127) → RGB [0,1]
        使用 D65 白点
        """
        L, a, b_ch = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        
        # Lab → XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b_ch / 200
        
        epsilon = 0.008856
        kappa = 903.3
        
        x = torch.where(fx ** 3 > epsilon, fx ** 3, (116 * fx - 16) / kappa)
        y = torch.where(L > kappa * epsilon, ((L + 16) / 116) ** 3, L / kappa)
        z = torch.where(fz ** 3 > epsilon, fz ** 3, (116 * fz - 16) / kappa)
        
        # D65 白点
        x = x * 0.95047
        y = y * 1.00000
        z = z * 1.08883
        
        # XYZ → sRGB
        r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
        b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252
        
        rgb_linear = torch.cat([r, g, b], dim=1)
        
        # 线性 RGB → sRGB
        mask = rgb_linear > 0.0031308
        rgb = torch.where(mask, 1.055 * (rgb_linear ** (1/2.4)) - 0.055, 12.92 * rgb_linear)
        
        return torch.clamp(rgb, 0.0, 1.0)

    
    
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
        
