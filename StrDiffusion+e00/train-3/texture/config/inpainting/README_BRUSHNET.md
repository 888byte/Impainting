# 壁画修复训练说明

## 1. 当前训练目标

当前 `train-3` 纹理训练链在原版 StrDiffusion 主干上扩展了以下辅助模块：

- `BrushNet`
- `MGLC-Tex`
- `Mu-Denoiser`
- `restore_S_guidance`

但核心原则已经固定：

- texture 主干修复语义回到原版 StrDiffusion  
  `mu / cond = observed_degraded * mask_known`
- `BrushNet / color_prior / confidence / Mu-Denoiser / MGLC` 只做辅助引导
- `Gs / Dis` 官方结构链不改

## 2. 目录结构

```text
StrDiffusion+e00/train-3/texture/config/inpainting/
├── train.py
├── lut_processor.py
├── color_prior_generator.py
├── data/
│   ├── __init__.py
│   └── mural_inpainting_dataset.py
├── models/
│   ├── denoising_model.py
│   ├── brushnet_wrapper.py
│   ├── pixel_brushnet.py
│   ├── zero_conv.py
│   └── modules/
│       └── mglc_block.py
├── options/train/
│   ├── ir-sde-brushnet.yml
│   └── ir-sde-brushnet-ft.yml
└── README_BRUSHNET.md
```

## 3. 启动方式

标准训练：

```bash
python train.py -opt ./texture/config/inpainting/options/train/ir-sde-brushnet.yml
```

继续做干净的 finetune：

```bash
python train.py -opt ./texture/config/inpainting/options/train/ir-sde-brushnet-ft.yml
```

当前不再使用 `train_brushnet.py`。  
训练入口统一为 `train.py`。

## 4. LUT 与 prior 配置

训练配置里必须显式写出：

- `datasets.train.lut_path`
- `datasets.train.prior_method`
- `datasets.train.gt_mode`

当前不再允许 `mural_inpainting` 数据集静默回退到旧的 `pigment_lut33.npz` 默认路径。  
如果 `lut_path` 缺失，会直接报错。

`prior_method` 支持：

- `fast`
- `quality`

正式训练建议：

```yaml
datasets:
  train:
    lut_path: /path/to/your_lut.npz
    prior_method: quality
```

## 5. `full` 与 `partial`

训练目标支持两种模式：

- `gt_mode: full`
  整图颜色恢复 + 修补
- `gt_mode: partial`
  已知区尽量保留，仅在 hole 区使用 LUT 目标

评价结果时必须先确认当前模式，不能混用标准。

## 6. 当前训练数据流

训练时会同时维护两路输入：

- `original_degraded`
  真实缺损外观输入，模拟推理输入
- `reference_degraded`
  完整参考图，仅用于构造训练目标

关键点：

- texture 主干条件  
  `condition = original_degraded * mask_known`
- 训练目标  
  通过 `ColorPriorGenerator.build_target(...)` 从完整参考图构造
- `color_prior / confidence`
  由真实缺损输入生成，只做辅助引导

## 7. TensorBoard 重点看什么

训练排查时，优先看：

- `train/texture_condition_gap`
- `train/ema_texture_loss`
- `train/loss_main`
- `train/loss_hole_weighted`
- `train/loss_total`

重点图像：

- `train_vis/original_degraded`
- `train_vis/reference_degraded`
- `train_vis/color_prior`
- `train_vis/lut_transformed`
- `train_vis/mu_clean`
- `train_vis/mask_hole`
- `train_vis/mask_known`

判断规则：

- `texture_condition_gap` 应长期接近 `0`
- `original_degraded` 应为真实缺损输入
- `reference_degraded` 应为完整参考图
- `color_prior` 只应作为辅助参考，不应反客为主

## 8. 常见问题

### 8.1 结果整体发黄

当前训练和推理都已对 LUT 混合权重做 `clamp`。  
如果仍然发黄，优先检查：

- LUT 文件本身是否偏暖
- `color_prior_lut` 与 `color_prior_inpainted` 哪一环先带偏

### 8.2 为什么 `lut_transformed` 洞里还是白的

这是正常定义。  
`lut_transformed` 只是观测图的 LUT 结果，不负责补洞。  
hole 区真正的颜色参考来自 `color_prior`。

### 8.3 为什么 `final` 和 `gt.png` 差很多

先确认你比较的是不是训练目标同分布结果。  
训练真正学的是 `build_target(...)` 生成的目标，不一定等于你手头保存的原始完整图。

## 9. 推荐实验习惯

- 新一轮代码修正后，重新开一个实验名
- 用干净的 `pretrain_model_G` 做 finetune
- 不直接沿用已经在错误分布下训练过的 `resume_state`
- 固定 bad case，周期性导出：
  - `color_prior`
  - `raw_pred`
  - `final`
  做阶段对比
