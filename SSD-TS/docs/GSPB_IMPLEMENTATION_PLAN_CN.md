# GSPB 实施计划（中文）

## 目标

GSPB（Group-aware Spectral Posterior Bridge）的目标是：在不破坏现有 RGB 主链和 NPZ schema 的前提下，让 Raman/XRD 在训练期和推理期都以显式条件的形式发挥作用。

## 当前仓库如何接入 GSPB

当前仓库已经有：

- 真谱条件编码器；
- RGB -> embedding 预测器；
- retrieval 路径；
- 条件 diffusion 恢复。

因此 GSPB 的接入策略不是推翻重写，而是新增：

- `bridge/posterior_head.py`
- `bridge/prototype_bank.py`
- `bridge/confidence_gate.py`
- `bridge/distill.py`
- `build_prototypes.py`

## 各部件解决的问题

- posterior：把 RGB-only 推理从“单点 embedding 回归”改成“prototype 分布预测”。
- prototype bank：把真实谱知识压缩为 train-fold 局部记忆。
- retrieval：保留已有 missing-modality 显式路径作为兼容与对照。
- confidence gate：在 posterior 与 retrieval 都可用时做稳妥融合。
- distillation：用真谱 teacher posterior/condition 约束 RGB 分支。

## 如何处理 RGB 增强与谱未增强的不平衡

- 不改 NPZ schema。
- 用 `sample_index.csv` 记录 `spectral_parent_id` / `augmentation_parent_id`。
- sampler 和 split 按 parent 而不是按单条增强样本工作。
- prototype bank 仅使用 train fold parent 聚合。

## 不改 NPZ schema 的实现方式

新增信息放在 sidecar：

- `sample_index.csv`
- `train_index.csv`
- `val_index.csv`
- `test_index.csv`
- `prototype_bank.npz`

运行时通过 `data/dataset.py`、`data/npz_view.py` 绑定 sidecar 与旧 NPZ。

## 建议目录结构

- `bridge/`：bridge 相关逻辑
- `data/`：adapter/index/split
- `training/`：训练与 sampler
- `inference/`：推理与 uncertainty
- `evaluation/`：协议和消融

## 分阶段实施建议

### 阶段 1

- 完成自然命名迁移
- 完成文档和 sidecar index
- 保持 `true/pred/retrieval` 路径可运行

### 阶段 2

- 构建 train-fold prototype bank
- 新增 posterior head
- inference 支持 `posterior`

### 阶段 3

- inference 支持 `posterior_retrieval`
- 训练接入 distill 和 gate
- 加入 uncertainty 输出

## 验收标准

- 关闭 `bridge.enable` 后旧行为仍然可运行。
- `posterior` / `posterior_retrieval` 在推理时不访问真实 Raman/XRD。
- prototype bank 仅由 train fold 构建。
- 宽表 Excel 可以通过新 adapter 正确解析。
