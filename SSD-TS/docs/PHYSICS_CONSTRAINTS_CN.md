# Physics-Informed Soft Constraints 说明

## 目标

本次新增的是附加型 physics-informed soft constraints，不替代现有 RGB 主链、posterior / prototype / retrieval / gate，也不修改 NPZ schema。

## 包含的约束

### 1. Spec-Color Consistency

- 用 `spec_color_head(pseudo_cond)` 预测辅助颜色。
- 主监督目标是真实 `x0[:,0,:]`。
- 可选地再与 `x0_pred[:,0,:].detach()` 做弱一致性。
- 推理阶段不需要真实 Raman/XRD，只使用当前 bridge 已有的 pseudo spectral condition。

### 2. Parent / Augmentation Consistency

- 优先约束 posterior logits 或 posterior 分布。
- 没有 posterior 时退化为约束 pseudo spectral latent。
- `augmentation_parent_id` 用于 `L_aug_consistency`。
- `parent_consistency_level="auto"` 时，优先级为 `spectral_parent_id > sequence_parent_id > side`。
- 当只剩 `side` 可用时，按 `side_consistency_scale` 降低权重，group size < 2 自动跳过。

### 3. Damage Score Constraint

- 用 `damage_head([zc, pseudo_cond])` 或 `damage_head(zc)` 输出 `damage_score`。
- 只有 batch 同时具有 `sequence_parent_id` 和 `t` 时，才计算 `L_damage_mono` 和 `L_damage_smooth`。
- 顺序信息缺失时自动返回 0，不会报错。

## 置信度与兼容性

- `low_confidence_skip_physics=true` 时，physics loss 使用 `detach()` 后的 confidence 做软加权，避免模型通过压低 confidence 逃避约束。
- `physics.enable=false` 时，新增约束与旧 cycle loss 都关闭。
- `physics.use_cycle_model="auto"` 时，保持对旧 `FadingForwardModelLab` 的兼容。

## 推理诊断

- 可选输出 `posterior_entropy`、`confidence_bridge`、`spec_color_agreement`、`damage_score`。
- 只有在配置启用且 checkpoint 中存在对应 head 权重时才会输出；否则会 warning 并自动禁用，不会随机初始化。

## 限制

- 当前颜色一致性默认基于归一化 Lab。
- Parent / augmentation consistency 依赖 batch 内出现重复 group。
- Damage constraint 依赖可信的顺序元信息。
- 这次改动是轻量 soft constraints，不是新的物理主模型。
