# SSD-TS 颜料褪色恢复项目

本项目的当前任务是：输入褪色后的 RGB 颜色或颜色序列，预测褪色前的原始颜色。现有实现保留 RGB 主链和 diffusion 主体，并支持训练期使用 Raman/XRD、推理期通过 `pred`、`retrieval`、`posterior`、`prototype` 等 bridge 显式工作。

## 当前真实状态

- 默认入口：`preprocess.py`、`train.py`、`infer.py`、`evaluate.py`、`build_prototypes.py`
- 当前活跃实现目录：`data/`、`models/`、`bridge/`、`training/`、`inference/`、`evaluation/`、`utils/`
- `pigment_task/` 主要保留兼容入口和历史脚本
- 预处理当前真实产物仍是 pair-only `L=2` 样本，不是完整长序列训练数据
- 当前单点 RGB 推理已经统一输出：
  - `rgb`：预测的原始 RGB
  - `lab`：预测的原始 Lab
  - `conf`：融合置信度
  - `std`：diffusion 不确定性
  - `cdiff`：diffusion confidence
  - `cret`：retrieval / bridge confidence

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

训练带有早停：

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

`--rgb` 输出中，LUT 生成和实验脚本主要依赖这些字段：

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

当前 LUT 构建脚本：

- [pigment_task/build_pigment_lut33.py](/D:/code/ky/bihua/Impainting/SSD-TS/pigment_task/build_pigment_lut33.py)

它现在直接消费 `infer.py --rgb` 的当前输出：

- `rgb`
- `lab`
- `conf`
- `std`
- `cdiff`
- `cret`

生成结果 `.npz` 中的主要键：

- `grid`：RGB 采样网格
- `lut_rgb`：每个网格点对应的预测原始 RGB
- `lut_lab`：每个网格点对应的预测原始 Lab
- `lut_conf`：融合置信度
- `lut_std`：diffusion 不确定性标量
- `lut_cdiff`：diffusion confidence
- `lut_cret`：retrieval / bridge confidence
- `meta`：本次生成使用的主要参数

### 推荐命令

稳妥正式版：

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --cond_method auto \
  --num_samples 24 \
  --grid_size 49 \
  --kalman_rts \
  --device cuda \
  --max_workers 24 \
  --max_inflight 96 \
  --timeout_sec 240 \
  --retries 8 \
  --save_every 500 \
  --out_npz pigment_lut49_hq.npz
```

高质量版：

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --cond_method auto \
  --num_samples 32 \
  --grid_size 49 \
  --kalman_rts \
  --device cuda \
  --max_workers 32 \
  --max_inflight 128 \
  --timeout_sec 300 \
  --retries 8 \
  --save_every 500 \
  --out_npz pigment_lut49_xhq.npz
```

极限细网格版：

```bash
python pigment_task/build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --cond_method auto \
  --num_samples 32 \
  --grid_size 65 \
  --kalman_rts \
  --device cuda \
  --max_workers 32 \
  --max_inflight 128 \
  --timeout_sec 360 \
  --retries 8 \
  --save_every 800 \
  --out_npz pigment_lut65_ultra.npz
```

### 关键参数说明

`--cond_method`

- `auto`：优先推荐。跟当前 checkpoint 的 bridge 配置保持一致，通常是最稳的正式选择。
- `posterior_retrieval`：如果你在单点测试里确认这条路径颜色更自然，可以直接拿它生成正式 LUT。
- `pred`：容易整体偏向单一色域，只建议做对照。
- `retrieval`：更依赖库和检索结果，也只建议做对照。

`--num_samples`

- 控制每个 RGB 网格点的扩散采样次数。
- 越大越稳，单点波动越小，`std/conf` 更可信，但耗时会线性增长。
- 推荐范围：`16`、`24`、`32`。
- 如果只是快速出结果，可以先用 `14`；如果是最终正式 LUT，建议至少 `24`。

`--grid_size`

- 控制 LUT 的空间分辨率。
- `33`：默认档，速度较快。
- `49`：明显更细，通常是高质量正式版的推荐档。
- `65`：极细，但时间和计算量会非常大。

网格点数量是立方增长：

- `33^3 = 35937`
- `49^3 = 117649`
- `65^3 = 274625`

所以 `grid_size` 从 `33` 提到 `49`，总任务量大约会变成 `3.3x`；从 `33` 提到 `65`，总任务量大约会变成 `7.6x`。

`--kalman_rts`

- 建议正式 LUT 一律打开。
- 它会让单点 RGB 结果更稳，减少局部异常漂色点。
- 对质量有帮助，代价通常小于直接继续增大 `grid_size`。

`--max_workers`

- 控制同时跑多少个 `infer.py --rgb` 子进程。
- 这个参数主要影响吞吐和 GPU 利用率。
- 太小：GPU 吃不满。
- 太大：进程切换、超时、模型反复加载会拖慢整体速度。

`--max_inflight`

- 控制排队中的最大任务数。
- 通常建议设成 `max_workers` 的 `3x` 到 `4x`。
- 常用搭配：
  - `24 / 96`
  - `32 / 128`
  - `40 / 160`

`--timeout_sec`

- 单个点推理超时时间。
- 并发高、`num_samples` 大、`grid_size` 大时要一起调大。
- 保守建议：`240` 到 `360`。

`--retries`

- 单个网格点失败后的重试次数。
- 建议高质量正式版至少设为 `6` 到 `8`，避免少量超时导致整张 LUT 有坏点。

`--save_every`

- 断点续存盘频率。
- 生成时间长时，建议不要太大。
- 常用值：`300`、`500`、`800`。

### 怎么调参数

如果你更关心质量：

- 优先提高 `--num_samples`
- 然后再提高 `--grid_size`
- `--kalman_rts` 保持开启
- `--cond_method` 优先用 `auto` 或你已经验证过更自然的 `posterior_retrieval`

建议顺序：

1. `num_samples: 14 -> 24`
2. `num_samples: 24 -> 32`
3. `grid_size: 33 -> 49`
4. 最后才考虑 `grid_size: 49 -> 65`

如果你更关心把大显存吃满：

- 主要调 `--max_workers`
- 配套调 `--max_inflight`
- 不要只盯显存占用率，更重要的是整体吞吐是否真的更高

推荐调法：

1. 先用 `--max_workers 24 --max_inflight 96`
2. 看 `nvidia-smi`
3. 如果 GPU 利用率和显存占用仍明显偏低，再加到 `32 / 128`
4. 如果还不够，再试 `40 / 160`

如果你只有一次正式生成机会：

- 不建议一开始就上 `grid_size 65`
- 更推荐先用 `grid_size 49 + num_samples 24/32`
- 这是质量和时间之间更稳的平衡点

### 断点续跑

LUT 构建脚本支持 resume：

- 如果已有 `out_npz`，会优先加载已完成的网格点
- 如果已有 `done_npy`，会跳过已经完成的格点
- 中断后重新执行同一命令即可继续跑

## 可视化与实验脚本

这些脚本主要用于快速可视化，不属于训练主链，但都已经对齐到当前工程结构。

### t1.py

用途：批量颜色面板测试。

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

### t2.py

用途：整图颜色替换测试。

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
