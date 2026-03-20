import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import sys

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def compute_components(self, predict, target, mask, weights=None):
        """Return the current loss decomposition without changing the training formula."""
        lossm = self.loss_fn(predict * (1 - mask), target * (1 - mask), reduction='none')
        lossm = einops.reduce(lossm, 'b ... -> b (...)', 'mean')

        lossu = self.loss_fn(predict * mask, target * mask, reduction='none')
        lossu = einops.reduce(lossu, 'b ... -> b (...)', 'mean')

        loss_hole_weighted = 10 * lossm
        loss = lossu + loss_hole_weighted
        if self.is_weighted and weights is not None:
            loss = weights * loss

        return {
            'loss_total': loss.mean(),
            'loss_known': lossu.mean(),
            'loss_hole': lossm.mean(),
            'loss_hole_weighted': loss_hole_weighted.mean(),
        }

    def forward(self, predict, target, mask, weights=None):
        return self.compute_components(predict, target, mask, weights)['loss_total']
      
