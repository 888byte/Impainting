# SSD-TS 颜料褪色恢复项目

本项目当前的主任务是：输入褪色后的 RGB 颜色或颜色序列，预测褪色前的原始颜色。现有实现保留 RGB 主链和 diffusion 主体，并支持训练期使用 Raman/XRD、推理期通过 `pred/retrieval/posterior/prototype` bridge 显式工作。

## 当前真实状态

- 当前默认入口：`preprocess.py`、`train.py`、`infer.py`、`evaluate.py`、`build_prototypes.py`
- 当前活跃实现目录：`data/`、`models/`、`bridge/`、`training/`、`inference/`、`evaluation/`、`utils/`
- `pigment_task/` 主要保留旧兼容入口和历史脚本
- 预处理当前真实产物仍是 pair-only `L=2` 样本，不是完整长序列训练数据
- 当前推理链的单点 RGB 输出已经统一为：
  - `rgb`: 预测的原始 RGB
  - `lab`: 预测的原始 Lab
  - `conf`: 融合置信度
  - `std`: diffusion 不确定性
  - `cdiff`: diffusion confidence
  - `cret`: retrieval / bridge confidence

## 标准工作流

### 1. 预处理

```bash
python preprocess.py \
  --rgb_logs "data/all/all.txt,data/all/all_right.txt" \
  --output_dir data/pigment_npz \
  --use_patches "1-9" \
  --meta_json pigment_task/pigment_meta_example.json \
  --raman_excel "data/all/laman.xlsx" \
  --xrd_excel "data/all/xrd.xlsx" \
  --split_mode group_exp_patch
```

输出包括：

- `train.npz` / `val.npz` / `test.npz` / `all.npz`
- `sample_index.csv` / `train_index.csv` / `val_index.csv` / `test_index.csv`
- `preprocess_meta.json`

### 2. 训练

```bash
python train.py --config configs/lab_raman_xrd.json
```

训练会按验证集自动保存：

- `best_model.pt`
- `best_true_model.pt`
- `best_pred_model.pt`
- 周期性 `ckpt_ep*.pt`

并带有早停：

- `train.early_stopping_patience`
- `train.early_stopping_min_delta`

### 3. 构建 prototype bank

```bash
python build_prototypes.py \
  --config configs/lab_raman_xrd.json \
  --ckpt ckpt/lab_raman_xrd/best_model.pt
```

### 4. 推理

测试集评估：

```bash
python infer.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --test_npz data/pigment_npz/test.npz \
  --cond_method pred \
  --kalman_rts
```

单点 RGB 推理：

```bash
python infer.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --rgb "120,80,60" \
  --cond_method posterior \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --kalman_rts
```

`--rgb` 输出中，LUT 生成和实验脚本主要依赖以下字段：

- `rgb`
- `lab`
- `conf`
- `std`
- `cdiff`
- `cret`

### 5. 评估

```bash
python evaluate.py \
  --mode test \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --test_npz data/pigment_npz/test.npz \
  --cond_method pred
```

## LUT 生成

当前 LUT 构建脚本是：

- [pigment_task/build_pigment_lut33.py](/D:/code/ky/bihua/Impainting/SSD-TS/pigment_task/build_pigment_lut33.py)

它现在已经按当前推理链对齐，不再读取旧字段，而是直接消费 `infer.py --rgb` 输出的：

- `rgb`
- `lab`
- `conf`
- `std`
- `cdiff`
- `cret`

### 推荐用法

如果你想让 LUT 尽量贴近当前正式推理行为，推荐优先用 `auto` 或 `posterior_retrieval`：

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --cond_method auto \
  --num_samples 14 \
  --grid_size 33 \
  --kalman_rts \
  --out_npz pigment_lut33.npz
```

如果你想显式走联合 bridge：

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --cond_method posterior_retrieval \
  --num_samples 14 \
  --grid_size 33 \
  --kalman_rts \
  --out_npz pigment_lut33.npz
```

### 生成结果

LUT `.npz` 中的主要键：

- `grid`: 33 级 RGB 网格
- `lut_rgb`: 每个网格点对应的预测原始 RGB
- `lut_lab`: 每个网格点对应的预测原始 Lab
- `lut_conf`: 融合置信度
- `lut_std`: diffusion 不确定性标量
- `lut_cdiff`: diffusion confidence
- `lut_cret`: retrieval / bridge confidence
- `meta`: 生成参数

### 断点续跑

LUT 构建脚本支持 resume：

- 如果已有 `out_npz`，会优先加载已有网格点
- 如果已有 `done_npy`，会跳过已经完成的格点
- 中断后重新执行同一命令即可继续跑

## 可视化与实验脚本

这些脚本主要用于快速可视化，不属于训练主链，但都已经对齐到当前工程结构。

### t1.py

用途：批量颜色面板测试。

输入：一组测试颜色
输出：左侧输入 / 右侧恢复 的对比图

推荐：

```bash
python t1.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method auto \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --num_samples 30 \
  --palette hsv_fps \
  --n_test_colors 144 \
  --min_lab_dist 10 \
  --output_image batch_test_144_auto.png
```

如果要显式走联合 bridge：

```bash
python t1.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method posterior_retrieval \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --num_samples 30 \
  --palette hsv_fps \
  --n_test_colors 144 \
  --min_lab_dist 10 \
  --output_image batch_test_144_postret.png
```

### t2.py

用途：整图颜色替换测试。

流程：

1. 对整张图做 KMeans 聚类得到当前调色板
2. 批量恢复调色板颜色
3. 将恢复调色板映射回整图

示例：

```bash
python t2.py \
  --input_image demo.png \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method auto \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --num_samples 20 \
  --n_colors 16 \
  --output_image restored_batch.png
```

### t3.py

用途：基于模型生成掩膜区域的颜色先验图和置信图。

流程：

1. Telea 结构修补
2. 从掩膜外区域提取上下文调色板
3. 批量恢复调色板颜色
4. 按最近调色板重着色掩膜区域，并输出置信图

示例：

```bash
python t3.py \
  --img_path demo.png \
  --mask_path demo_mask.png \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --cond_method auto \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --num_samples 20 \
  --n_colors 32 \
  --output_dir results_priors_recolor
```

### t4.py

用途：检查 `.npz` 文件内容。

```bash
python t4.py pigment_lut33.npz
```

### t5.py

用途：基于 LUT 生成掩膜区域颜色先验图和置信图。

```bash
python t5.py \
  --img_path demo.png \
  --mask_path demo_mask.png \
  --lut_npz pigment_lut33.npz \
  --n_colors 64 \
  --output_dir results
```

### t6.py

用途：基于 LUT 对整张图做连续颜色替换，不使用 mask。

```bash
python t6.py \
  --img_path demo.png \
  --lut_npz pigment_lut33.npz \
  --use_lut lab \
  --keep_luminance \
  --output_dir results_mural_lut
```

### t7.py

用途：仅在掩膜区域应用 LUT 颜色替换。

```bash
python t7.py \
  --img_path demo.png \
  --mask_path demo_mask.png \
  --lut_npz pigment_lut33.npz \
  --use_lut lab \
  --keep_luminance \
  --mask_feather 7 \
  --output_dir results_mural_lut
```

## 配置说明

- 旧 `missing_modality` 配置继续保留，用于兼容旧 checkpoint 和 `pred/retrieval` 路径
- 新 bridge 配置统一位于 `bridge.*`
- 物理约束统一位于 `physics.*`
- 单点 RGB 推理稳定化配置位于 `inference.*`

## Physics 物理约束

当前仓库已接入最小侵入式 physics-informed soft constraints，默认关闭，不会改写主链，只在训练时作为附加 loss 工作。

可选子开关包括：

- `use_spec_color_consistency`
- `use_parent_consistency`
- `use_aug_consistency`
- `use_damage_constraint`

更详细说明见 [docs/PHYSICS_CONSTRAINTS_CN.md](/D:/code/ky/bihua/Impainting/SSD-TS/docs/PHYSICS_CONSTRAINTS_CN.md)。

## Legacy 兼容

旧命令仍保留：

```bash
python -m pigment_task.preprocess_pigment ...
python -m pigment_task.train_pigment --config pigment_task/configs/pigment_lab_raman_xrd_v2.json
python -m pigment_task.infer_pigment --ckpt ...
```

但默认文档、默认脚本和后续新功能都以自然命名入口为准。
