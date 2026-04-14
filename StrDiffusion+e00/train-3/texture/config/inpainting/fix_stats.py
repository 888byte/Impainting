import os
from collections import OrderedDict

filepath = r'D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i == 725:
        break
    new_lines.append(line)

new_code = """    def _masked_mean_std(self, tensor, mask):
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
"""

for line in new_code.splitlines():
    new_lines.append(line + '\n')

for line in lines[736:]:
    new_lines.append(line)

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
