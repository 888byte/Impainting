"""
损失函数集合：主要是 masked inpainting 的 matching loss。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys

class MatchingLoss(nn.Module):
    """对遮挡/未遮挡区域分别计算损失并加权。"""
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, mask, weights=None):
        """mask: 1 表示未遮挡，0 表示遮挡。"""
       
        lossm = self.loss_fn(predict * (1 - mask), target * (1 - mask), reduction='none')
        lossm = einops.reduce(lossm, 'b ... -> b (...)', 'mean')
        
        lossu = self.loss_fn(predict * mask, target * mask, reduction='none')
        lossu = einops.reduce(lossu, 'b ... -> b (...)', 'mean')

        loss = lossu + 10 * lossm
        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()    
      
