# E1 (BrushNet) code pack

This pack provides a repo-agnostic PyTorch implementation of:
- `BrushEncoder`: multi-scale feature extractor
- `BrushInjector`: ZeroConv residual injection helper

It **does not** include your StrDiffusion repo code. You must wire it into your Texture U-Net forward.

## What "E1" means (in your "no Stage-1 yet" plan)
- Add BrushNet branch only (no `C_prior/w` yet)
- Brush encoder input: `[I_deg*(1-M), M]` (hole pixels are 0, mask indicates missing)

## Integration checklist

### 1) Data / conditioning tensor
In your dataset __getitem__ or model forward, construct:

```python
I_ctx = I_deg * (1 - M)      # RGB in [0,1] or [-1,1], consistent with your repo
brush_in = torch.cat([I_ctx, M], dim=1)  # BCHW, M is 1-channel
```

### 2) Instantiate Brush modules
Decide how many injection sites you want (recommended: 5 scales).
You need `target_channels_per_site`: the channel count of the UNet hidden tensor where you inject.

Example:
```python
from brushnet import BrushEncoder, BrushEncoderConfig, BrushInjector

brush = BrushEncoder(BrushEncoderConfig(in_channels=4, base_channels=64, num_scales=5))
injector = BrushInjector(
    brush_channels=brush.out_channels_per_scale,
    target_channels=target_channels_per_site,  # you fill this
)
```

### 3) Modify Texture UNet forward
Wherever you compute a hidden feature `h` at each resolution, do:

```python
brush_feats = brush(brush_in)  # list
h = injector.inject(h, brush_feats[site_idx], site_idx)
```

Tips:
- `site_idx` should align to your UNet resolution order (e.g., 0=highest-res).
- If your UNet uses more/fewer resolutions, adjust `num_scales` and `target_channels_per_site` accordingly.

### 4) Training strategy (stable)
For E1 you usually:
- **Freeze** baseline modules (structure UNet, texture UNet, discriminator)
- **Train** only brush encoder + ZeroConvs

In PyTorch:
```python
for p in baseline.parameters():
    p.requires_grad_(False)
for p in brush.parameters():
    p.requires_grad_(True)
for p in injector.parameters():
    p.requires_grad_(True)
```

Important:
- Do **NOT** run baseline forward inside `torch.no_grad()`; you still need gradients to flow back to brush via the injected hidden states.
- But freezing baseline parameters is enough.

### 5) Run the same E0 train/eval pipeline
Only change: add brush branch + freeze baseline.

Compare E1 vs E0:
- focus mask LPIPS + boundary quality (subjective & ring metrics)
- keep the same split/seed/steps

## Files
- `brushnet.py`: modules described above
