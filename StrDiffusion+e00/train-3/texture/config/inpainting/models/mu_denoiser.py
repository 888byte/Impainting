"""
Self-Supervised Mu-Denoiser (D_mu) for Stage2 IR-SDE

This module cleans the target-domain condition mean (mu) before feeding to the SDE.
For mural mode the RGB input is condition_lut = LUT(denoised(observed_degraded)),
not the raw degraded-domain image, so the SDE drift remains in the target color domain.

Training: Uses blind-spot/Noise2Self approach (no clean GT required)
- Sample blind spots in known (mask=1) regions
- Replace blind spot pixels with neighbor values
- Predict full image, compute loss only on blind spots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pureformer_blocks import TransformerBlock


class MuDenoiser(nn.Module):
    """
    Self-supervised mu denoiser used to clean mu in Stage2.

    Input: concat([rgb, mask_known, confidence]) => [B, 5, H, W]
      - rgb: target-domain condition_lut (0~1), e.g. LUT(denoised(observed_degraded))
      - mask_known: mask_for_sde (1=known, 0=hole) - SDE semantics!
      - confidence: optional LUT confidence, if None use ones

    Output: y_hat (denoised RGB) => [B, 3, H, W]
    
    IMPORTANT: mask semantics must match SDE convention:
      - 1 = known/valid pixels
      - 0 = hole/missing pixels (to be inpainted)
    This is the OPPOSITE of dataset mask (1=missing, 0=known)!
    """
    def __init__(
        self,
        in_nc: int = 5,
        dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        ffn_expansion_factor: float = 2.0,
        attn_dilations=(1, 2),
        ffn_dilations=(1,),
        predict_residual: bool = True,   # Safer: less color drifting
        clamp_output: bool = True,
    ):
        super().__init__()
        self.in_nc = in_nc
        self.dim = dim
        self.predict_residual = predict_residual
        self.clamp_output = clamp_output

        self.stem = nn.Conv2d(in_nc, dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                attn_dilations=attn_dilations,
                ffn_dilations=ffn_dilations,
                gate="relu_sigmoid",
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, 5, H, W] = [rgb(3), mask_known(1), confidence(1)]
               mask_known uses SDE semantics: 1=known, 0=hole
        
        Returns:
            y_hat: [B, 3, H, W] denoised RGB image
        """
        # x: [B, 5, H, W] = [rgb(3), mask(1), conf(1)]
        rgb = x[:, 0:3]
        mask_known = x[:, 3:4]  # 1=known, 0=hole (SDE semantics)

        feat = self.stem(x)
        feat = self.blocks(feat)
        out = self.head(feat)

        if self.predict_residual:
            # residual means "remove what should be removed"
            y_hat = rgb - out
        else:
            y_hat = out

        # Keep hole region as input (we don't want to predict holes)
        # This is consistent with mu = y_hat * mask_known used in SDE
        y_hat = y_hat * mask_known + rgb * (1.0 - mask_known)

        if self.clamp_output:
            y_hat = y_hat.clamp(0.0, 1.0)
        return y_hat


class MuDenoiserTrainer:
    """
    Helper class for self-supervised training of MuDenoiser.
    
    Implements blind-spot masking (Noise2Self style) for training
    without clean ground truth.
    
    Usage in optimize_parameters():
        trainer = MuDenoiserTrainer(mu_denoiser, blind_ratio=0.1)
        y_hat, losses = trainer.train_step(
            y_degraded, mask_known, confidence,
            lambda_ss=1.0, lambda_tv=0.01
        )
        mu_clean = y_hat.detach() * mask_known
    """
    
    def __init__(self, model: MuDenoiser, blind_ratio: float = 0.1):
        """
        Args:
            model: MuDenoiser instance
            blind_ratio: Percentage of known pixels to mask as blind spots (0.05-0.15)
        """
        self.model = model
        self.blind_ratio = blind_ratio
    
    def generate_blind_mask(self, mask_known: torch.Tensor) -> torch.Tensor:
        """
        Generate blind-spot mask within known regions only.
        
        CRITICAL: blind spots are ONLY sampled from mask_known=1 regions!
        This is essential for Noise2Self to work correctly.
        
        Args:
            mask_known: [B, 1, H, W] with 1=known, 0=hole (SDE semantics)
        
        Returns:
            m_blind: [B, 1, H, W] with 1=blind spot, 0=visible
                     blind spots are a subset of known regions
        """
        B, _, H, W = mask_known.shape
        device = mask_known.device
        
        # Random mask with blind_ratio probability
        random_mask = torch.rand(B, 1, H, W, device=device) < self.blind_ratio
        
        # CRITICAL: Only sample blind spots from known (mask=1) regions
        m_blind = random_mask.float() * mask_known
        
        return m_blind
    
    def corrupt_blind_spots(
        self, 
        image: torch.Tensor, 
        m_blind: torch.Tensor,
        method: str = "neighbor"
    ) -> torch.Tensor:
        """
        Replace blind spot pixels with corrupted values.
        
        This prevents identity mapping and forces the network to
        learn the underlying signal structure.
        
        Args:
            image: [B, 3, H, W] original image
            m_blind: [B, 1, H, W] blind spot mask (1=blind, 0=visible)
            method: "neighbor" (shift), "noise" (random), or "mean" (local avg)
        
        Returns:
            corrupted: [B, 3, H, W] image with blind spots replaced
        """
        if method == "neighbor":
            # Shift by 1 pixel (simple, effective)
            # Random direction to avoid bias
            direction = torch.randint(0, 4, (1,)).item()
            if direction == 0:
                shifted = F.pad(image[:, :, 1:, :], (0, 0, 0, 1), mode='replicate')
            elif direction == 1:
                shifted = F.pad(image[:, :, :-1, :], (0, 0, 1, 0), mode='replicate')
            elif direction == 2:
                shifted = F.pad(image[:, :, :, 1:], (0, 1, 0, 0), mode='replicate')
            else:
                shifted = F.pad(image[:, :, :, :-1], (1, 0, 0, 0), mode='replicate')
            corrupted = image * (1 - m_blind) + shifted * m_blind
            
        elif method == "noise":
            # Replace with uniform noise
            noise = torch.rand_like(image)
            corrupted = image * (1 - m_blind) + noise * m_blind
            
        elif method == "mean":
            # Replace with local mean (3x3 average)
            kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9.0
            local_mean = []
            for c in range(3):
                lm = F.conv2d(image[:, c:c+1], kernel, padding=1)
                local_mean.append(lm)
            local_mean = torch.cat(local_mean, dim=1)
            corrupted = image * (1 - m_blind) + local_mean * m_blind
        else:
            raise ValueError(f"Unknown corruption method: {method}")
        
        return corrupted
    
    def compute_self_supervised_loss(
        self,
        y_hat: torch.Tensor,
        y_target: torch.Tensor,
        m_blind: torch.Tensor,
        loss_type: str = "charbonnier"
    ) -> torch.Tensor:
        """
        Compute self-supervised loss ONLY on blind spot pixels.
        
        Key insight from Noise2Self: When blind spots are independently
        corrupted, the optimal prediction is the conditional expectation
        of the clean signal.
        
        Args:
            y_hat: [B, 3, H, W] network prediction
            y_target: [B, 3, H, W] original (noisy) target
            m_blind: [B, 1, H, W] blind spot mask
            loss_type: "l1", "l2", or "charbonnier"
        
        Returns:
            loss: scalar loss value
        """
        # Only compute loss on blind spots
        y_hat_blind = y_hat * m_blind
        y_target_blind = y_target * m_blind
        
        if loss_type == "l1":
            loss = F.l1_loss(y_hat_blind, y_target_blind, reduction='sum')
        elif loss_type == "l2":
            loss = F.mse_loss(y_hat_blind, y_target_blind, reduction='sum')
        elif loss_type == "charbonnier":
            eps = 1e-6
            diff = y_hat_blind - y_target_blind
            loss = torch.sqrt(diff.pow(2) + eps).sum()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Normalize by number of blind spot pixels
        num_blind = m_blind.sum().clamp(min=1.0)
        loss = loss / num_blind
        
        return loss
    
    def compute_tv_loss(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Total Variation regularization to prevent output instability.
        
        Computes on the residual (x - input) to preserve edges while
        smoothing the denoising residual.
        
        Args:
            x: [B, 3, H, W] input tensor (typically y_hat - y_input)
            mask: [B, 1, H, W] optional mask to apply TV only in known regions
        
        Returns:
            tv_loss: scalar TV loss
        """
        # Gradient in x direction
        diff_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Gradient in y direction
        diff_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        if mask is not None:
            # Apply mask (truncated to match gradient dimensions)
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]
            diff_x = diff_x * mask_x
            diff_y = diff_y * mask_y
            num_pixels = (mask_x.sum() + mask_y.sum()).clamp(min=1.0)
        else:
            num_pixels = diff_x.numel() + diff_y.numel()
        
        tv_loss = (diff_x.abs().sum() + diff_y.abs().sum()) / num_pixels
        return tv_loss
    
    def train_step(
        self,
        y_degraded: torch.Tensor,
        mask_known: torch.Tensor,
        confidence: torch.Tensor = None,
        lambda_ss: float = 1.0,
        lambda_tv: float = 0.01,
        corruption_method: str = "neighbor"
    ):
        """
        Complete training step for self-supervised denoising.
        
        Args:
            y_degraded: [B, 3, H, W] target-domain condition_lut image (0~1)
            mask_known: [B, 1, H, W] known mask (1=known, 0=hole) - SDE semantics!
            confidence: [B, 1, H, W] optional LUT confidence
            lambda_ss: weight for self-supervised loss
            lambda_tv: weight for TV regularization
            corruption_method: how to corrupt blind spots
        
        Returns:
            y_hat: [B, 3, H, W] denoised prediction
            losses: dict with 'l_ss', 'l_tv', 'l_total'
        """
        B = y_degraded.shape[0]
        device = y_degraded.device
        
        # Default confidence
        if confidence is None:
            confidence = torch.ones(B, 1, *y_degraded.shape[2:], device=device)
        
        # 1. Generate blind spots in known regions
        m_blind = self.generate_blind_mask(mask_known)
        
        # 2. Corrupt blind spot pixels
        y_corrupted = self.corrupt_blind_spots(y_degraded, m_blind, corruption_method)
        
        # 3. Build input and forward
        x_input = torch.cat([y_corrupted, mask_known, confidence], dim=1)
        y_hat = self.model(x_input)
        
        # 4. Compute losses
        l_ss = self.compute_self_supervised_loss(y_hat, y_degraded, m_blind)
        
        # TV on residual in known regions only
        residual = y_hat - y_degraded
        l_tv = self.compute_tv_loss(residual, mask_known)
        
        l_total = lambda_ss * l_ss + lambda_tv * l_tv
        
        losses = {
            'l_ss': l_ss.item(),
            'l_tv': l_tv.item(),
            'l_mu_total': l_total.item()
        }
        
        return y_hat, l_total, losses
    
    def inference(
        self,
        y_degraded: torch.Tensor,
        mask_known: torch.Tensor,
        confidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Inference without blind spots (for test/validation).
        
        Args:
            y_degraded: [B, 3, H, W] target-domain condition_lut image
            mask_known: [B, 1, H, W] known mask (SDE semantics)
            confidence: [B, 1, H, W] optional confidence
        
        Returns:
            y_hat: [B, 3, H, W] denoised image
        """
        B = y_degraded.shape[0]
        device = y_degraded.device
        
        if confidence is None:
            confidence = torch.ones(B, 1, *y_degraded.shape[2:], device=device)
        
        x_input = torch.cat([y_degraded, mask_known, confidence], dim=1)
        
        with torch.no_grad():
            y_hat = self.model(x_input)
        
        return y_hat
