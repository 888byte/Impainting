# 最终增强版推理说明

## 1. 目标

这套推理代码位于 `test/texture-1/config/inpainting`，保留了官方的推理骨架：

- `network_G / network_Gs / Dis`
- `pretrain_model_G / pretrain_model_Gs / pretrain_model_D`
- `test.py -> model.test(...) -> sde.reverse_sde(...)`

同时把 `train-3` 中真正生效的纹理增强链补到了推理侧：

- `ConditionalUNetWithBrushNet`
- `BrushNet`
- `MGLC-Tex`
- `Mu-Denoiser`
- `restore_S_guidance`

## 2. 目录结构

- `test.py`
  入口脚本。兼容原始官方测试，也支持新的壁画推理模式。
- `data/mural_inference_dataset.py`
  真实壁画推理数据集，负责读取 `degraded / mask / GT(optional)`。
- `models/networks.py`
  保留官方 `G / Gs / Dis` 创建方式，同时支持新的 `ConditionalUNetWithBrushNet`。
- `models/denoising_model.py`
  推理主流程，负责颜色先验、mu、结构条件、中间结果保存。
- `utils/sde_utils.py`
  保留官方 `reverse_sde(...)` 入口，并在内部新增最终增强版纹理反推分支。
- `options/test/ir-sde-brushnet.yml`
  最终增强版推理配置。

## 3. 启动方式

默认启动：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml
```

带消融覆盖启动：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false
python test.py -opt options/test/ir-sde-brushnet.yml --set restore_S_guidance=false
python test.py -opt options/test/ir-sde-brushnet.yml --set mu_denoiser.enabled=false
```

## 4. 输入数据组织

建议结构：

```text
degraded_images/
  0001.png
  0002.png

masks/
  0001.png
  0002.png

gt/
  0001.png
  0002.png
```

匹配规则：

- 按文件名 stem 对齐
- 例如 `0001.png` 会匹配 `degraded / mask / gt`

## 5. Mask 语义

这是整个推理最关键的部分。

外部输入约定：

- mask 白色/255 = 待修复区域
- mask 黑色/0 = 已知区域

内部统一拆成两个变量：

- `mask_hole`: `1 = 待修复区域`
- `mask_known`: `1 = 已知区域`

使用规则固定：

- `Gs / S_sde / 已知区域保留` 使用 `mask_known`
- `BrushNet / MGLC / 最终纹理网络` 使用 `mask_hole`

不要在任何新代码里把一个 `mask` 同时当这两种语义使用。

## 6. 结果输出

结果目录默认位于：

```text
results/inpainting/<name>/<dataset_name>/<image_stem>/
```

其中会输出：

- `final.png`
- `raw_pred.png`
- `gt.png`（如果提供 GT）

如果 `inference.save_intermediates=true`，还会额外输出：

- `mask_hole.png`
- `mask_known.png`
- `color_prior.png`
- `confidence.png`
- `denoised_original.png`
- `lut_transformed.png`
- `mu_clean.png`
- `structure_gray.png`
- `structure_edge.png`
- `raw_pred.png`
- `final.png`

这些文件用于定位：

- mask 是否翻转
- color prior 是否异常
- confidence 是否全黑或全白
- mu 是否退化
- 原始输出和最终组合是否一致

## 7. 与训练侧的对齐点

当前推理代码已经对齐的功能：

- `ConditionalUNetWithBrushNet`
- `BrushNet` 条件注入
- `texture_core`
- `restore_S_guidance`
- `Mu-Denoiser`
- `LUT` 颜色路径
- `color_prior / confidence` 自动生成
- 官方 `G / Gs / Dis` 权重加载

需要你自己保证的一致性：

- 训练和推理使用同一版 `pigment_lut33.npz`
- 推理 YAML 中的 `texture_core / restore_S_guidance / mu_denoiser` 与训练权重匹配
- mask 输入遵守“白色=待修复区域”

## 8. 常见问题

### 8.1 修复区整片发红

优先检查：

1. `mask_hole.png` 和 `mask_known.png` 是否互补
2. `color_prior.png` 在修复区是否异常
3. `confidence.png` 是否接近全 0
4. `final.png` 是否只是组合阶段出错，而 `raw_pred.png` 正常

### 8.2 已知区域也被改坏

检查：

- `datasets.test.gt_mode` 是否误设为 `full`
- 如果想保留已知区域，必须使用 `partial`

### 8.3 Mu-Denoiser 权重加载失败

检查：

- `mu_denoiser.enabled` 是否和训练时一致
- `pretrain_model_G` 是否为最终增强版 checkpoint

## 9. 论文写作建议

推理部分建议描述为：

1. 保留官方 `G/Gs/Dis` 推理骨架
2. 将纹理网络升级为 `ConditionalUNetWithBrushNet`
3. 推理时显式区分 `mask_known` 和 `mask_hole`
4. 使用训练同源的颜色先验与结构引导链
5. 通过中间结果导出保证推理过程可诊断
