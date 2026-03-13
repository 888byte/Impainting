# 实验协议（中文）

## 数据划分 protocol

默认推荐：`group_exp_patch` 或更严格的 parent-aware split。核心原则是：同一个 `spectral_parent_id` 不能跨 train/val/test。

## 如何防 augmentation leakage

- 通过 `sample_index.csv` 记录 `augmentation_parent_id`。
- sampler 和 split 必须按 parent 聚合，而不是按增强样本独立打散。
- 如果后续接入 2000+ RGB 增强样本，必须先补充 sidecar parent mapping。

## prototype bank 构建规则

- 只使用 train fold 样本。
- 按 `spectral_parent_id` 聚合真谱条件。
- 不允许直接混入 val/test fold 条件。

## baseline / ablation / upper bound / control

建议至少报告：

- `true`：上界/teacher/analysis
- `pred`：旧 RGB-only embedding 路径
- `retrieval`：旧检索路径
- `posterior`
- `posterior_retrieval`
- shuffled-spectrum control

## 如何证明 Raman/XRD 真正起作用

- 对比 `pred` 与 `posterior` / `posterior_retrieval`
- 做 shuffled-spectrum control，验证谱条件被打乱后性能下降
- 报告 prototype bank 只用 train fold 的条件下，posterior 仍有稳定贡献

## uncertainty 指标建议

- diffusion sampling std / `conf_diffusion`
- retrieval entropy / `confidence_retrieval`
- posterior entropy / max-prob / `confidence_bridge`

## 当前已知限制

- 当前数据端真实仍是 pair-only。
- left/right parent 关系已写入 sidecar，但更细粒度 sequence 协议仍待扩展。
- 宽表 Excel adapter 已实现第一版，但复杂表头变体仍需要继续验证。
