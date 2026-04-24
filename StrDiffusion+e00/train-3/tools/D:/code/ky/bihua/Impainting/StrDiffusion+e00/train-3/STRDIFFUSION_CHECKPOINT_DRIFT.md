# StrDiffusion trunk checkpoint drift audit

- baseline: `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
- current: `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/32000_G.pth`
- shared trunk tensors: `231`
- extra current tensors: `189`

## Global trunk drift

- relative_rms: `0.060467`; rms_diff: `0.001221`; mean_abs_diff: `0.000919`

## Drift by module group

| group | tensors | relative_rms | rms_diff | mean_abs_diff |
|---|---:|---:|---:|---:|
| `mid_block1` | 4 | 0.092435 | 0.001053 | 0.000832 |
| `mid_block2` | 4 | 0.090084 | 0.001095 | 0.000872 |
| `ups.3` | 16 | 0.072375 | 0.002830 | 0.001665 |
| `ups.0` | 17 | 0.070741 | 0.001159 | 0.000916 |
| `ups.2` | 17 | 0.068244 | 0.002450 | 0.001748 |
| `downs.0` | 35 | 0.056192 | 0.002313 | 0.001439 |
| `ups.1` | 17 | 0.052852 | 0.001462 | 0.001116 |
| `downs.3` | 34 | 0.049144 | 0.000997 | 0.000786 |
| `downs.1` | 35 | 0.047262 | 0.001641 | 0.001175 |
| `downs.2` | 35 | 0.045511 | 0.001195 | 0.000907 |
| `final_res_block` | 5 | 0.044515 | 0.001842 | 0.001020 |
| `time_mlp` | 4 | 0.024817 | 0.001242 | 0.000881 |
| `final_conv` | 2 | 0.013526 | 0.000501 | 0.000381 |
| `init_conv` | 1 | 0.007157 | 0.000344 | 0.000241 |
| `mid_attn` | 5 | 0.003436 | 0.000231 | 0.000126 |

## Top changed trunk tensors

| key | relative_rms | rms_diff | mean_abs_diff |
|---|---:|---:|---:|
| `downs.0.0.block1.proj.weight` | 0.155902 | 0.005114 | 0.003305 |
| `mid_block1.block2.proj.weight` | 0.112917 | 0.001090 | 0.000861 |
| `ups.3.0.block1.proj.weight` | 0.110019 | 0.003781 | 0.002609 |
| `downs.0.4.conv_0.bias` | 0.105181 | 0.001337 | 0.000976 |
| `mid_block2.block2.proj.weight` | 0.104858 | 0.001109 | 0.000883 |
| `mid_block1.block1.proj.weight` | 0.103565 | 0.001027 | 0.000814 |
| `ups.3.0.block2.proj.weight` | 0.102574 | 0.003602 | 0.002426 |
| `mid_block2.block1.proj.weight` | 0.100860 | 0.001091 | 0.000869 |
| `ups.0.0.res_conv.weight` | 0.097763 | 0.001066 | 0.000849 |
| `downs.2.4.conv_0.weight_v` | 0.097643 | 0.001438 | 0.001075 |
| `ups.2.3.1.weight` | 0.097149 | 0.002240 | 0.001720 |
| `downs.0.0.block2.proj.weight` | 0.095995 | 0.003130 | 0.001850 |
| `ups.0.1.res_conv.weight` | 0.092081 | 0.001143 | 0.000910 |
| `ups.3.1.block1.proj.weight` | 0.089357 | 0.002524 | 0.001480 |
| `ups.0.3.1.weight` | 0.088821 | 0.001188 | 0.000945 |
| `downs.3.4.conv_1.weight_v` | 0.088206 | 0.000919 | 0.000730 |
| `downs.3.4.conv_0.weight_v` | 0.088149 | 0.000918 | 0.000720 |
| `ups.0.0.block1.proj.weight` | 0.085803 | 0.001137 | 0.000905 |
| `ups.0.1.block1.proj.weight` | 0.083958 | 0.001176 | 0.000934 |
| `ups.2.1.block2.proj.weight` | 0.080167 | 0.002781 | 0.002009 |
| `ups.0.0.block2.proj.weight` | 0.080127 | 0.001217 | 0.000968 |
| `downs.3.3.weight` | 0.080048 | 0.001050 | 0.000837 |
| `ups.3.0.mlp.1.weight` | 0.078332 | 0.003706 | 0.002272 |
| `ups.1.3.1.weight` | 0.076998 | 0.001518 | 0.001194 |
| `ups.2.1.block1.proj.weight` | 0.076120 | 0.002467 | 0.001809 |
| `ups.1.1.res_conv.weight` | 0.076064 | 0.001535 | 0.001173 |
| `downs.0.1.block1.proj.weight` | 0.075058 | 0.002931 | 0.001861 |
| `downs.0.1.block2.proj.weight` | 0.073558 | 0.002847 | 0.001827 |
| `ups.2.1.res_conv.weight` | 0.073471 | 0.002134 | 0.001547 |
| `ups.0.1.block2.proj.weight` | 0.071384 | 0.001154 | 0.000909 |

## Extra current keys sample

- `brushnet.downs.0.0.block1.proj.weight`
- `brushnet.downs.0.0.block2.proj.weight`
- `brushnet.downs.0.0.mlp.1.bias`
- `brushnet.downs.0.0.mlp.1.weight`
- `brushnet.downs.0.1.block1.proj.weight`
- `brushnet.downs.0.1.block2.proj.weight`
- `brushnet.downs.0.1.mlp.1.bias`
- `brushnet.downs.0.1.mlp.1.weight`
- `brushnet.downs.0.2.fn.fn.to_out.0.bias`
- `brushnet.downs.0.2.fn.fn.to_out.0.weight`
- `brushnet.downs.0.2.fn.fn.to_out.1.g`
- `brushnet.downs.0.2.fn.fn.to_qkv.weight`
- `brushnet.downs.0.2.fn.norm.g`
- `brushnet.downs.0.3.bias`
- `brushnet.downs.0.3.weight`
- `brushnet.downs.1.0.block1.proj.weight`
- `brushnet.downs.1.0.block2.proj.weight`
- `brushnet.downs.1.0.mlp.1.bias`
- `brushnet.downs.1.0.mlp.1.weight`
- `brushnet.downs.1.1.block1.proj.weight`
- `brushnet.downs.1.1.block2.proj.weight`
- `brushnet.downs.1.1.mlp.1.bias`
- `brushnet.downs.1.1.mlp.1.weight`
- `brushnet.downs.1.2.fn.fn.to_out.0.bias`
- `brushnet.downs.1.2.fn.fn.to_out.0.weight`
- `brushnet.downs.1.2.fn.fn.to_out.1.g`
- `brushnet.downs.1.2.fn.fn.to_qkv.weight`
- `brushnet.downs.1.2.fn.norm.g`
- `brushnet.downs.1.3.bias`
- `brushnet.downs.1.3.weight`
- `brushnet.downs.2.0.block1.proj.weight`
- `brushnet.downs.2.0.block2.proj.weight`
- `brushnet.downs.2.0.mlp.1.bias`
- `brushnet.downs.2.0.mlp.1.weight`
- `brushnet.downs.2.1.block1.proj.weight`
- `brushnet.downs.2.1.block2.proj.weight`
- `brushnet.downs.2.1.mlp.1.bias`
- `brushnet.downs.2.1.mlp.1.weight`
- `brushnet.downs.2.2.fn.fn.to_out.0.bias`
- `brushnet.downs.2.2.fn.fn.to_out.0.weight`
- `brushnet.downs.2.2.fn.fn.to_out.1.g`
- `brushnet.downs.2.2.fn.fn.to_qkv.weight`
- `brushnet.downs.2.2.fn.norm.g`
- `brushnet.downs.2.3.bias`
- `brushnet.downs.2.3.weight`
- `brushnet.downs.3.0.block1.proj.weight`
- `brushnet.downs.3.0.block2.proj.weight`
- `brushnet.downs.3.0.mlp.1.bias`
- `brushnet.downs.3.0.mlp.1.weight`
- `brushnet.downs.3.1.block1.proj.weight`
- `brushnet.downs.3.1.block2.proj.weight`
- `brushnet.downs.3.1.mlp.1.bias`
- `brushnet.downs.3.1.mlp.1.weight`
- `brushnet.downs.3.2.fn.fn.to_out.0.bias`
- `brushnet.downs.3.2.fn.fn.to_out.0.weight`
- `brushnet.downs.3.2.fn.fn.to_out.1.g`
- `brushnet.downs.3.2.fn.fn.to_qkv.weight`
- `brushnet.downs.3.2.fn.norm.g`
- `brushnet.downs.3.3.weight`
- `brushnet.init_conv.weight`
- `brushnet.mid_attn.fn.fn.to_out.0.bias`
- `brushnet.mid_attn.fn.fn.to_out.0.weight`
- `brushnet.mid_attn.fn.fn.to_out.1.g`
- `brushnet.mid_attn.fn.fn.to_qkv.weight`
- `brushnet.mid_attn.fn.norm.g`
- `brushnet.mid_block1.block1.proj.weight`
- `brushnet.mid_block1.block2.proj.weight`
- `brushnet.mid_block1.mlp.1.bias`
- `brushnet.mid_block1.mlp.1.weight`
- `brushnet.mid_block2.block1.proj.weight`
- `brushnet.mid_block2.block2.proj.weight`
- `brushnet.mid_block2.mlp.1.bias`
- `brushnet.mid_block2.mlp.1.weight`
- `brushnet.time_mlp.1.bias`
- `brushnet.time_mlp.1.weight`
- `brushnet.time_mlp.3.bias`
- `brushnet.time_mlp.3.weight`
- `brushnet.zero_conv_mid.conv.bias`
- `brushnet.zero_conv_mid.conv.weight`
- `brushnet.zero_convs_down.0.0.conv.bias`
