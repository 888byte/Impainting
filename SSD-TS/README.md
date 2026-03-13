# SSD-TS 颜料褪色恢复项目

本仓库当前聚焦的任务是：输入褪色后的 RGB/颜色序列，预测褪色前的原始颜色。现有实现保留 RGB 主链和 diffusion 主体，并在此基础上提供真实谱条件、RGB 预测条件、检索条件，以及正在逐步接入的 posterior/prototype bridge。

## 当前真实状态

- 当前活跃入口已经迁移为自然命名：`python preprocess.py`、`python train.py`、`python infer.py`、`python evaluate.py`、`python build_prototypes.py`。
- 新的活跃实现位于顶层包：`data/`、`models/`、`bridge/`、`training/`、`inference/`、`evaluation/`。
- `pigment_task/` 仍保留旧命令兼容层，例如 `python -m pigment_task.train_pigment`，但这些文件现在只做 wrapper。
- 当前预处理真实产物仍是 pair-only `L=2` 样本，不是完整 sequence schema。推理端支持基于最后观测点构造条件，但数据端尚未落地长序列训练。
- 当前仓库已经存在 `true` / `pred` / `retrieval` 缺模态路径；posterior/prototype bridge 已有模块骨架与 prototype bank 脚本，但仍属于渐进增强阶段。

## 已确认的真实差异

- 旧 README/研究描述把当前数据流写成了 sequence 模式；真实代码在预处理阶段只生成 `t0 -> t` 的 pair 样本。
- 旧数据集读取器声明支持 `meta_t_start/meta_t_end/meta_seq_len`，但实际预处理写的是 `meta_t`。新 `data/dataset.py` 已兼容这两种元数据。
- 真实 Raman/XRD 文件并不总是“每个材料一个 sheet”。你提供的两份 Excel 只有一个 `Sheet1`，而且是宽表多材料布局。新预处理会优先尝试宽表 adapter。
- left/right、augmentation parent、spectral parent 过去没有在代码中显式建模；现在 `preprocess.py` 会额外生成 `sample_index.csv`、`train_index.csv`、`val_index.csv`、`test_index.csv` sidecar 来承载这些关系，不修改 NPZ schema。

## 默认工作流

### 1. 预处理

```bash
python preprocess.py \
  --rgb_logs "path/to/1-3.txt,path/to/1-3_Right.txt" \
  --output_dir data/pigment_npz \
  --use_patches "1-9" \
  --meta_json pigment_task/pigment_meta_example.json \
  --raman_excel "path/to/拉曼.xlsx" \
  --xrd_excel "path/to/xrd.xlsx" \
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

### 3. 构建 prototype bank

```bash
python build_prototypes.py --config configs/lab_raman_xrd.json --ckpt ckpt/lab_raman_xrd/best_model.pt
```

### 4. 推理

```bash
python infer.py --ckpt ckpt/lab_raman_xrd/best_model.pt --test_npz data/pigment_npz/test.npz --cond_method pred
python infer.py --ckpt ckpt/lab_raman_xrd/best_model.pt --rgb "120,80,60" --cond_method posterior
```

### 5. 评估

```bash
python evaluate.py --mode test --ckpt ckpt/lab_raman_xrd/best_model.pt --test_npz data/pigment_npz/test.npz --cond_method pred
python evaluate.py --mode palette --ckpt ckpt/lab_raman_xrd/best_model.pt --out_csv palette_eval.csv --cond_method posterior
python evaluate.py --mode mine --ckpt ckpt/lab_raman_xrd/best_model.pt --test_npz data/pigment_npz/test.npz --out_csv mine.csv --cond_method pred
```

## 目录结构

```text
preprocess.py
train.py
infer.py
evaluate.py
build_prototypes.py

configs/
data/
models/
bridge/
training/
inference/
evaluation/
utils/
docs/
tests/
legacy/
```

## 配置说明

- 旧 `missing_modality` 配置继续保留，用于兼容旧 checkpoint 和 `pred/retrieval` 路径。
- 新 bridge 配置统一位于 `bridge.*`：
  - `bridge.enable`
  - `bridge.mode = pred|retrieval|posterior|posterior_retrieval`
  - `bridge.use_gate`
  - `bridge.use_distill`
  - `bridge.use_group_sampler`
  - `bridge.prototype_bank.path`

## Physics ??

???????????? physics-informed soft constraints????? `physics.*` ????????????????????????? loss ??????

???????

- `use_spec_color_consistency`
- `use_parent_consistency`
- `use_aug_consistency`
- `use_damage_constraint`

?????

- `physics.enable=false` ?????????????
- `physics.use_cycle_model="auto"` ??????? `FadingForwardModelLab` cycle loss?
- ????????????? checkpoint ???????????????????

?????????

```json
{
  "physics": {
    "enable": true,
    "use_cycle_model": false,
    "use_spec_color_consistency": true,
    "lambda_spec_color": 0.1,
    "lambda_spec_pred_consistency": 0.0
  }
}
```

??????? [docs/PHYSICS_CONSTRAINTS_CN.md](/D:/code/ky/bihua/Impainting/SSD-TS/docs/PHYSICS_CONSTRAINTS_CN.md)?

## Legacy 兼容

旧命令仍保留：

```bash
python -m pigment_task.preprocess_pigment ...
python -m pigment_task.train_pigment --config pigment_task/configs/pigment_lab_raman_xrd_v2.json
python -m pigment_task.infer_pigment --ckpt ...
```

但默认文档、默认脚本和后续新功能都以自然命名入口为准。
