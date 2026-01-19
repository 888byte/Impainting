"""
纹理去噪模型（Texture Denoising Model）。
主要职责：构建网络、定义损失/优化器、执行训练与推理。
"""

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

logger = logging.getLogger("base")


import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():
    """tensor -> PIL.Image"""
    return transforms.ToPILImage()


def image_to_tensor():
    """PIL.Image -> tensor"""
    return transforms.ToTensor()


def gray_to_edge(image, sigma):
    """灰度图 -> Canny 边缘图。"""

    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

    return edge


# 额外日志工具：统计学习率与梯度范数
def get_lr(optim):
        if optim is None or len(optim.param_groups) == 0:
            return 0.0
        return float(optim.param_groups[0].get("lr", 0.0))

@torch.no_grad()
def grad_l2_norm(module: torch.nn.Module) -> float:
    """计算梯度的 L2 范数（监控训练稳定性）。"""
    total = 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.pow(2).sum().item())
    return float(total ** 0.5)
#-----------------------------------------------------------



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
        # dist 训练：取真实 rank；非 dist：rank=-1
        self.rank = -1
        if opt.get("dist", False):
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank = dist.get_rank()
        train_opt = opt["train"]
        

        # 构建生成器/判别器并加载预训练模型
        self.model, self.dis = networks.define_G(opt)
        self.model = self.model.to(self.device)
        self.dis = self.dis.to(self.device)
        
        # 多卡训练时使用 DataParallel
        gpu_ids = opt.get('gpu_ids', None)
        if gpu_ids is not None and len(gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=gpu_ids, output_device=gpu_ids[0])
            self.dis   = DataParallel(self.dis,   device_ids=gpu_ids, output_device=gpu_ids[0])

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

            # 优化器与调度器
            self.optimizer_d = torch.optim.Adam(self.dis.parameters(), lr = 1e-4, betas = (0.5, 0.99))#1e-4
            
            trainable = [(n, p.numel()) for n,p in self.model.named_parameters() if p.requires_grad]
            print("Trainable param tensors:", len(trainable))
            print("Trainable total params:", sum(x[1] for x in trainable))
            print("Example trainable names:", [trainable[i][0] for i in range(min(10, len(trainable)))])

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


    def feed_data(self, state, LQ, GT, mask, S_sde, S_GT, S_LQ, color_prior=None, confidence=None):
        """缓存训练数据到 GPU：xt/LQ/GT/mask/结构引导 + 颜色先验/置信度。"""
        self.state = state.to(self.device)      # noisy_state
        self.condition = LQ.to(self.device)     # LQ（mu）
        self.state_0 = GT.to(self.device)       # GT
        self.mask = mask.to(self.device)        # mask
        self.S_sde = S_sde
        self.S_GT = S_GT.to(self.device)
        self.S_LQ = S_LQ.to(self.device)

        # ✅ 新增：BrushNet 需要的两个输入（允许 None）
        self.color_prior = None if color_prior is None else color_prior.to(self.device)
        self.confidence  = None if confidence  is None else confidence.to(self.device)



    def optimize_parameters(self, step, timesteps, sde=None):
        """
        1) 先构造纹理目标 yt_1_optimum
        2) 构造结构引导 S_optimum（不能为 None，否则 SPADE 会崩）
        3) noise_fn 必须传 S_optimum，同时透传 color_prior/confidence 给 PixelBrushNet
        4) 反传只更新 brush 分支（你已冻结主干的话）
        """
        # 让 SDE 知道当前的 mu（条件图）
        sde.set_mu(self.condition)

        # timesteps 放到 device
        timesteps = timesteps.to(self.device)

        # -----------------------------
        # 1) 纹理目标：yt_1_optimum
        # -----------------------------
        yt_1_optimum = sde.reverse_optimum_step(self.state, self.state_0, timesteps)

        # -----------------------------
        # 2) 结构引导：S_optimum（关键！不能传 None）
        #    这块逻辑来自你原来的实现
        # -----------------------------
        # 注意：S_sde 是一个独立的 sde（你 feed_data 里传进来的 self.S_sde）
        # 它需要 mu（结构条件）= self.S_LQ
        if self.S_sde is not None:
            self.S_sde.set_mu(self.S_LQ)

            # 生成结构分支的随机状态 + 最优反推目标
            # x0 = self.S_GT（结构 GT）
            # mu = self.S_LQ * mask（结构条件的已知区域）
            S_timestep, S_state = self.S_sde.generate_random_states_texture(
                x0=self.S_GT,
                mu=self.S_LQ * self.mask,
                timesteps=timesteps
            )
            # 得到结构的最优目标（S_optimum）
            S_optimum = self.S_sde.reverse_optimum_step(S_state, self.S_GT, timesteps)
        else:
            # 如果你确实没有结构 SDE，就不要走 SPADE guide（需要你在 SPADE 里做 segmap=None 兜底）
            # 但你目前网络里有 SPADE guide，所以正常训练这里不应该为 None
            S_optimum = None

        # -----------------------------
        # 3) 预测噪声 noise（必须把 S_optimum 传进去）
        #    同时透传 BrushNet 的两个输入（如果你 feed_data 里存了）
        # -----------------------------
        noise, _ = sde.noise_fn(
            self.state,
            timesteps.squeeze(),
            S_optimum,
            color_prior=getattr(self, "color_prior", None),
            confidence=getattr(self, "confidence", None),
        )

        # score & 期望 yt_1_expection
        score = sde.get_score_from_noise(noise, timesteps)
        yt_1_expection = sde.reverse_sde_step_mean(self.state, score, timesteps)

        # -----------------------------
        # 4) loss & backward
        # -----------------------------
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(yt_1_expection, yt_1_optimum, self.mask)
        loss.backward()
        self.optimizer.step()

        # 记录日志
        self.log_dict["loss"] = float(loss.detach().item())



    
    def test(self, sde=None, save_states=False, save_dir='save_dir', GT = None, mask = None, S_sde = None, S_GT = None, S_LQ = None, structure_guide = None):
        """推理/测试：执行 reverse SDE 生成输出。"""
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
        
