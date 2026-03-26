# 最终增强版推理说明

## 1. 目标

这套推理代码位于 `test/texture-1/config/inpainting`，保留了官方 StrDiffusion 的推理骨架：

- `network_G / network_Gs / Dis`
- `pretrain_model_G / pretrain_model_Gs / pretrain_model_D`
- `test.py -> model.test(...) -> sde.reverse_sde(...)`

同时，它对齐了 `train-3` 当前真实生效的纹理链：

- `ConditionalUNetWithBrushNet`
- `BrushNet`
- `MGLC-Tex`
- `Mu-Denoiser`
- `restore_S_guidance`

当前版本的关键约束是：

- texture 主干修复语义回到原版 StrDiffusion  
  `mu / cond = observed_degraded * mask_known`
- `BrushNet / color_prior / confidence / Mu-Denoiser / MGLC` 只做辅助引导
- `Gs / Dis` 保持官方链路，不改数学逻辑

## 2. 目录结构

- `test.py`
  官方骨架兼容入口，支持 legacy 与最终增强版推理。
- `data/mural_inference_dataset.py`
  真实壁画推理数据集，读取 `degraded / mask / GT(optional)`。
- `models/networks.py`
  保留官方 `G / Gs / Dis` 创建方式，同时支持 `ConditionalUNetWithBrushNet`。
- `models/denoising_model.py`
  最终增强版推理主流程，负责 prior、mu、结构条件、中间结果保存。
- `utils/sde_utils.py`
  保留官方 `reverse_sde(...)` 入口，并在内部适配增强版 texture generator。
- `options/test/ir-sde-brushnet.yml`
  最终增强版推理配置。

## 3. 启动方式

默认启动：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml
```

常用覆盖：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false
python test.py -opt options/test/ir-sde-brushnet.yml --set restore_S_guidance=false
python test.py -opt options/test/ir-sde-brushnet.yml --set mu_denoiser.enabled=false
python test.py -opt options/test/ir-sde-brushnet.yml --set datasets.test.gt_mode=partial
python test.py -opt options/test/ir-sde-brushnet.yml --set datasets.test.prior_method=quality
```

## 4. 输入数据组织

建议结构：

```text
degraded_images/
  0001.png
  0002.png

masks/
  0001_mask.png
  0002_mask.png

gt/
  0001.png
  0002.png
```

匹配规则：

- 样本主键始终使用输入图像 stem
- `mask` 优先匹配 `<stem>_mask.*`
- 若找不到 `<stem>_mask.*`，兼容回退到 `<stem>.*`
- `GT / color_prior / confidence` 按同 stem 匹配

## 5. Mask 语义

外部输入约定：

- mask 白色/255 = 待修复区域
- mask 黑色/0 = 已知区域

内部统一拆成两个变量：

- `mask_hole`: `1 = 待修复区域`
- `mask_known`: `1 = 已知区域`

使用规则固定：

- `Gs / S_sde / 已知区域保留 / partial 合成` 使用 `mask_known`
- `BrushNet / MGLC / texture generator` 使用 `mask_hole`

不要在任何新代码里用一个变量同时表示这两种语义。

## 6. `full` 与 `partial`

当前推理支持两种模式：

- `gt_mode: full`
  整图颜色恢复 + 修补，`final.png` 直接输出整图结果。
- `gt_mode: partial`
  已知区尽量保持原输入，只替换 hole 区。

判断结果时必须先确认模式：

- `full` 模式下，不能用“已知区必须逐像素不变”来评价。
- `partial` 模式下，已知区应与原始输入一致，重点看 hole 区替换效果。

## 7. 输出说明

结果目录默认位于：

```text
results/inpainting/<name>/<dataset_name>/<image_stem>/
```

默认输出：

- `final.png`
- `raw_pred.png`
- `gt.png`（如果提供 GT）

当 `inference.save_intermediates=true` 时，还会额外输出：

- `mask_hole.png`
- `mask_known.png`
- `color_prior.png`
- `confidence.png`
- `denoised_original.png`
- `lut_transformed.png`
- `color_prior_lut.png`
- `color_prior_inpainted.png`
- `training_target_like.png`
- `mu_clean.png`
- `structure_gray.png`
- `structure_edge.png`
- `x_init.png`
- `state_100.png`（如启用状态保存）

这些图的含义：

- `lut_transformed.png`
  只是观测图做 LUT 后的结果，不负责补洞。洞里保持白色是正常的。
- `color_prior_lut.png`
  纯 LUT 映射结果，用来判断颜色偏差是不是 LUT 自身带来的。
- `color_prior_inpainted.png`
  inpaint 后的 hole 补色结果，用来判断是不是补洞先验太弱。
- `color_prior.png`
  最终送入 BrushNet 的颜色参考图。
- `training_target_like.png`
  使用训练同一套 `build_target(...)` 规则构造的“训练目标同分布”可视化。  
  当 `gt.png` 不是训练目标分布时，优先参考这张图。
- `raw_pred.png`
  仅看 texture 主干输出，不含 `partial` 模式下的已知区回填。
- `final.png`
  最终交付结果。`full` 与 `partial` 模式下含义不同。

## 8. `prior_method`

`datasets.test.prior_method` 支持：

- `fast`
  更快，适合快速调试。
- `quality`
  更稳，适合正式推理和诊断 bad case。

当前推荐正式推理使用：

```yaml
datasets:
  test:
    prior_method: quality
```

训练和推理都应该显式写出 `prior_method`，不要依赖隐式默认值。

## 9. 与训练侧的对齐点

当前推理代码已经对齐的关键点：

- texture 主干 `mu / cond = observed_degraded * mask_known`
- `ConditionalUNetWithBrushNet`
- `BrushNet` 条件注入
- `texture_core`
- `restore_S_guidance`
- `Mu-Denoiser`
- `LUT` 颜色路径
- `color_prior / confidence` 自动生成
- 官方 `G / Gs / Dis` 权重加载

需要你自己保证的一致性：

- 训练和推理使用同一版 LUT `.npz` 文件
- 推理 YAML 中的 `texture_core / restore_S_guidance / mu_denoiser` 与训练权重匹配
- `gt_mode` 与当前实验目标一致
- mask 输入遵守“白色 = 待修复区域”

## 10. 常见问题

### 10.1 `lut_transformed` 洞里为什么还是白的

这是正常现象。  
`lut_transformed` 只表示“观测图做 LUT 后的结果”，不是补洞结果。  
hole 区真正的颜色参考来自 `color_prior`。

### 10.2 `final` 为什么看起来几乎等于没修

优先按下面顺序检查：

1. `training_target_like.png` 是否和你主观拿来对比的 `gt.png` 属于同一目标分布
2. `color_prior_lut.png` 与 `color_prior_inpainted.png` 哪一张先带偏
3. `raw_pred.png` 是否只是贴着 `color_prior.png`
4. 当前是 `full` 还是 `partial`

### 10.3 已知区域也被改坏

检查：

- `datasets.test.gt_mode` 是否误设为 `full`
- 如果想保留已知区域，必须使用 `partial`

### 10.4 Mu-Denoiser 权重加载失败

检查：

- `mu_denoiser.enabled` 是否和训练时一致
- `pretrain_model_G` 是否为当前增强版 checkpoint

## 11. 最小验收标准

一张 fixed bad case 至少同时看：

- `color_prior.png`
- `color_prior_lut.png`
- `color_prior_inpainted.png`
- `training_target_like.png`
- `raw_pred.png`
- `final.png`

判断规则：

- 如果 `raw_pred ≈ color_prior`，说明模型还没学到
- 如果 `raw_pred` 明显优于 `color_prior`，但 `final` 仍差，再查模式或组合逻辑
- 如果 `training_target_like` 和 `gt.png` 差很多，不要再只拿 `gt.png` 评价训练是否有效
