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

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.optimizer import Lion

from models.modules.loss import MatchingLoss

from .base_model import BaseModel

# LUT处理器
try:
    from lut_processor import LUTProcessor
except ImportError:
    LUTProcessor = None

# Mu-Denoiser（可选）
try:
    from models.mu_denoiser import MuDenoiser, MuDenoiserTrainer
    HAS_MU_DENOISER = True
except ImportError:
    HAS_MU_DENOISER = False

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




class DenoisingModel(BaseModel):
    def __init__(self, opt):
        super(DenoisingModel, self).__init__(opt)

        train_opt = opt["train"]
        

        # define network and load pretrained models
        self.model, self.models, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.models = self.models.to(self.device)
        self.dis = self.dis.to(self.device)
        if len(opt["gpu_ids"]) > 1:  
            self.model = DataParallel(self.model)
            self.models = DataParallel(self.models)
            self.dis = DataParallel(self.dis)
        
        # ============ 初始化 Mu-Denoiser（推理用）============
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
            
            # 推理时只需要 trainer 的 inference 方法
            self.mu_denoiser_trainer = MuDenoiserTrainer(
                self.mu_denoiser,
                blind_ratio=mu_denoiser_opt.get('blind_ratio', 0.1)
            )
            logger.info(f"[Model] Mu-Denoiser 已初始化 (推理模式)")
        else:
            self.mu_denoiser = None
            self.mu_denoiser_trainer = None
            if not HAS_MU_DENOISER and mu_denoiser_opt.get('enabled', False):
                logger.warning("[Model] Mu-Denoiser 配置已启用但模块未找到")
        
        # ============ 加载 LUT 处理器 ============
        lut_opt = opt.get('lut', {})
        lut_path = lut_opt.get('path', None)
        if lut_path and LUTProcessor is not None and os.path.exists(lut_path):
            self.lut_processor = LUTProcessor(lut_path)
            logger.info(f"[Model] LUT 处理器已加载: {lut_path}")
        else:
            self.lut_processor = None
        
        # LUT 相关配置
        self.gt_mode = lut_opt.get('gt_mode', 'partial')
        self.lut_strength = lut_opt.get('lut_strength', 0.7)
        self.lut_smooth_radius = lut_opt.get('lut_smooth_radius', 5)
        
        self.load()
        
        if self.is_train:
            self.model.train()
            self.models.train()
            self.dis.train()

            is_weighted = opt['train']['is_weighted']
            loss_type = opt['train']['loss_type']
            self.loss_fn = MatchingLoss(loss_type, is_weighted).to(self.device)
            self.loss1 = nn.L1Loss(reduction='mean') 
            self.loss2 = nn.MSELoss()
            
            self.weight = opt['train']['weight']

            # optimizers
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
                  color_prior=None, confidence=None, original_degraded=None):
        """
        加载数据
        
        Args:
            state: 噪声状态
            LQ: 低质量输入（条件）
            GT: Ground Truth
            mask: 掩码 (1=已知, 0=缺失)
            S_sde: 结构SDE
            S_GT: 结构GT
            S_LQ: 结构LQ
            color_prior: [可选] 颜色先验图，用于BrushNet
            confidence: [可选] 置信度图，用于BrushNet
            original_degraded: [可选] 原始褪色图（无 mask 涂黑）
        """
        self.state = state.to(self.device)    # noisy_state
        self.condition = LQ.to(self.device)  # LQ
        self.state_0 = GT.to(self.device)  # GT
        self.mask = mask.to(self.device) # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)
        
        # 原始褪色图（用于去噪）
        if original_degraded is not None:
            self.original_degraded = original_degraded.to(self.device)
        else:
            self.original_degraded = None
        
        # BrushNet条件
        if color_prior is not None:
            self.color_prior = color_prior.to(self.device)
        else:
            self.color_prior = None
            
        if confidence is not None:
            self.confidence = confidence.to(self.device)
        else:
            self.confidence = None

    def compute_mu_clean(self, y_degraded, mask_known, confidence=None):
        """
        使用 Mu-Denoiser 生成干净的 mu（推理用）
        
        Args:
            y_degraded: [B, 3, H, W] 原始褪色图
            mask_known: [B, 1, H, W] SDE mask (1=known, 0=hole)
            confidence: [B, 1, H, W] 可选置信度
        
        Returns:
            mu_clean: [B, 3, H, W] 去噪后的 mu (已乘 mask)
        """
        if not self.use_mu_denoiser or self.mu_denoiser is None:
            return y_degraded * mask_known
        
        with torch.no_grad():
            self.mu_denoiser.eval()
            mu_clean = self.mu_denoiser_trainer.inference(
                y_degraded, mask_known, confidence
            )
            mu_clean = mu_clean * mask_known
        
        return mu_clean

    def optimize_parameters(self, step, timesteps, sde=None):
        sde.set_mu(self.condition)
        
        self.optimizer.zero_grad()
        
        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)
        yt_1_expection  = torch.zeros_like(yt_1_optimum)
        timesteps = timesteps.to(self.device)
        # Get noise and score
        S_timestep, S_states = self.S_sde.generate_random_states_texture(x0=self.S_GT, mu=self.S_LQ * self.mask, timesteps = timesteps - 1)
        S_optimum = self.S_sde.reverse_optimum_step(S_states, self.S_GT, S_timestep)
        noise,g_score = sde.noise_fn(self.state, timesteps.squeeze(),S_optimum)
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)
        
        loss = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask)
        loss += 0.1 * (self.loss1(torch.ones_like(g_score*(1-self.mask)),g_score*(1-self.mask))+self.loss2(torch.ones_like(g_score)*(1-self.mask),g_score*(1-self.mask)))
        loss += self.loss1(yt_1_expection * g_score + (1 - g_score) * yt_1_optimum, yt_1_optimum)
        loss.backward()
        self.optimizer.step()
        self.ema.update()

        # set log
        self.log_dict["loss"] = loss.item()

    def begin(self,sde,noisy_states,S_sde,X_GT,mask):
        return sde.begin(noisy_states,S_sde,X_GT,mask)
    
    
    def test(self, sde=None, save_states=False, save_dir='save_dir', GT=None, mask=None, 
             S_sde=None, S_GT=None, S_LQ=None, dis=None, 
             color_prior=None, confidence=None):
        """
        测试/推理
        
        Args:
            sde: SDE实例
            save_states: 是否保存中间状态
            save_dir: 保存目录
            GT: Ground Truth
            mask: 掩码 (1=已知, 0=缺失)
            S_sde: 结构SDE
            S_GT: 结构GT
            S_LQ: 结构LQ
            dis: 判别器
            color_prior: [可选] 颜色先验图
            confidence: [可选] 置信度图
        """
        # ============ 使用 Mu-Denoiser 生成干净 mu ============
        if self.use_mu_denoiser and self.original_degraded is not None:
            mu_clean = self.compute_mu_clean(
                self.original_degraded, self.mask, self.confidence
            )
            sde.set_mu(mu_clean)
        else:
            sde.set_mu(self.condition)
        
        S_sde.set_mu(self.S_LQ)
        self.model.eval()
        self.models.eval()
        
        # 使用存储的color_prior和confidence（如果没有传入参数）
        if color_prior is None and hasattr(self, 'color_prior'):
            color_prior = self.color_prior
        if confidence is None and hasattr(self, 'confidence'):
            confidence = self.confidence
        
        # 构建BrushNet参数
        brushnet_kwargs = {}
        if color_prior is not None:
            brushnet_kwargs['color_prior'] = color_prior
        if confidence is not None:
            brushnet_kwargs['confidence'] = confidence
        if mask is not None:
            # BrushNet期望 mask=1表示需要修复，SDE的mask=1表示已知
            brushnet_kwargs['brushnet_mask'] = 1 - mask.to(self.device)
        
        with torch.no_grad():
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
                **brushnet_kwargs
            )

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
        load_path_Gs = self.opt["path"]["pretrain_model_Gs"]
        load_path_D = self.opt["path"]["pretrain_model_D"]
        
        if load_path_G is not None:
            logger.info("Loading model for G [{:s}] ...".format(load_path_G))
            
            # 加载 checkpoint（可能包含 mu_denoiser 权重）
            checkpoint = torch.load(load_path_G, map_location=self.device)
            
            # 分离主模型和 D_mu 的权重
            model_state = {}
            mu_denoiser_state = {}
            
            for k, v in checkpoint.items():
                if k.startswith('mu_denoiser.'):
                    mu_denoiser_state[k[len('mu_denoiser.'):]] = v
                elif k.startswith('module.mu_denoiser.'):
                    mu_denoiser_state[k[len('module.mu_denoiser.'):]] = v
                elif k.startswith('module.'):
                    model_state[k[7:]] = v
                else:
                    model_state[k] = v
            
            # 加载主模型
            self._load_network_from_state(model_state, self.model, self.opt["path"]["strict_load"])
            
            # 加载 Mu-Denoiser（如果有）
            if self.use_mu_denoiser and self.mu_denoiser is not None and len(mu_denoiser_state) > 0:
                try:
                    self.mu_denoiser.load_state_dict(mu_denoiser_state, strict=False)
                    logger.info(f"[Model] Mu-Denoiser 权重已加载 ({len(mu_denoiser_state)} 个参数)")
                except Exception as e:
                    logger.warning(f"[Model] Mu-Denoiser 权重加载失败: {e}")
            elif self.use_mu_denoiser and len(mu_denoiser_state) == 0:
                logger.warning("[Model] Checkpoint 中未找到 Mu-Denoiser 权重")
        
        # 加载 structure model
        if load_path_Gs is not None:
            logger.info("Loading model for Gs [{:s}] ...".format(load_path_Gs))
            self.load_network(load_path_Gs, self.models, self.opt["path"]["strict_load"])
        
        # 加载 discriminator
        if load_path_D is not None:
            logger.info("Loading model for D [{:s}] ...".format(load_path_D))
            self.load_network(load_path_D, self.dis, self.opt["path"]["strict_load"])

    def _load_network_from_state(self, state_dict, network, strict=True):
        """从 state_dict 加载网络"""
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        network.load_state_dict(state_dict, strict=strict)

    def save(self, iter_label):
        self.save_network(self.model, "G", iter_label)
        self.save_network(self.ema.ema_model, "EMA", 'lastest')
        
