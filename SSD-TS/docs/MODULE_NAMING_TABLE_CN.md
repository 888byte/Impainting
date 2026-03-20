# 模块命名建议表（文档建议，不改代码）

> 说明：本表只给出“如果后续要进一步朝顶会风格代码整理，可以考虑的命名建议”。它**不等于必须改名**。当前已经自然、职责清晰、被 README 和脚本广泛引用的名字，应优先保留，避免为了好看而增加迁移成本。

## 命名原则

1. 能保留就保留，不为了改名而改名。
2. 名字尽量简洁、中性、功能导向。
3. 避免 task-specific 冗余后缀，如 `_pigment`。
4. 方法核心模块可以略带学术意味，但不能脱离真实职责。
5. 历史实验脚本建议给出“未来规范名”，但当前仍可继续保留短名 wrapper。

## 建议表

| 当前模块/脚本名 | 当前职责 | 是否建议保留 | 建议名 | 命名风格说明 | 对应论文术语 | 备注 |
|---|---|---|---|---|---|---|
| `models/color_encoder.py` | 将当前颜色编码为 `z_color` | 保留 | `color_encoder.py` | 已经简洁、自然、职责明确 | Color Encoder | 不建议再改 |
| `models/cond_predictor.py` | 从颜色表征预测谱 embedding | 可视情况优化 | `spectral_predictor.py` | 当前名偏泛，建议名更贴近真实职责 | Spectral Predictor | 若后续要统一学术语义，可考虑文档层面用建议名 |
| `models/spectral_encoder.py` | 编码 Raman/XRD 并融合成条件向量 | 保留 | `spectral_encoder.py` | 简洁、准确 | Multimodal Spectral Encoder | 不建议再改 |
| `models/denoiser.py` | diffusion 主 denoiser | 保留 | `denoiser.py` | 顶会代码里常见、自然 | Conditional Denoiser | 不建议再改 |
| `models/physics.py` | 旧 cycle-style 物理模块 | 可保留 | `physics.py` | 名称中性，便于兼容旧逻辑 | Physics Prior / Fading Forward Model | 可在论文中单独注明这是 legacy-compatible module |
| `bridge/condition_builder.py` | 统一构造 true/pred/retrieval/posterior 条件 | 可视情况优化 | `condition_bridge.py` | 当前名偏实现，建议名更强调“桥接”角色 | Condition Bridge | 当前文件名可继续用，论文中可称 bridge module |
| `bridge/posterior_head.py` | 预测 prototype posterior | 保留 | `posterior_head.py` | 学术语义清晰 | Posterior Head | 不建议再改 |
| `bridge/prototype_bank.py` | 构建并聚合 prototype bank | 保留 | `prototype_bank.py` | 简洁且贴合方法 | Prototype Bank | 不建议再改 |
| `bridge/confidence_gate.py` | posterior 与 retrieval 置信度融合 | 保留 | `confidence_gate.py` | 名字清楚直接 | Confidence Gate | 不建议再改 |
| `bridge/distill.py` | posterior / embedding distillation | 可保留 | `distill.py` | 对熟悉训练代码的人足够自然 | Distillation Module | 若想更直观，可在文档里称 `distillation module` |
| `bridge/physics_heads.py` | `spec_color_head` 与 `damage_head` | 可保留 | `physics_heads.py` | 轻量、直接 | Physics Heads | 不建议再拆更多文件 |
| `training/trainer.py` | 主训练循环、验证、保存 ckpt、组装 loss | 保留 | `trainer.py` | 顶会代码常见命名 | Training Loop / Trainer | 不建议再改 |
| `training/diffusion.py` | diffusion schedule、loss、sampling | 保留 | `diffusion.py` | 清晰自然 | Diffusion Utilities | 不建议再改 |
| `training/physics_losses.py` | 物理软约束 loss | 可视情况优化 | `physics_regularizer.py` | 若更强调“软正则”，建议名更学术 | Physics Regularizer | 当前名已能工作，论文可称 regularizer |
| `training/samplers.py` | parent-aware sampler 等采样逻辑 | 保留 | `samplers.py` | 清晰直接 | Group-aware Sampler | 不建议再改 |
| `data/dataset.py` | 读取 NPZ 与 sidecar runtime metadata | 保留 | `dataset.py` | 简洁、自然 | Dataset / Runtime View | 不建议再改 |
| `data/npz_view.py` | sidecar 与 NPZ 绑定视图 | 保留 | `npz_view.py` | 语义明确 | Runtime NPZ View | 不建议再改 |
| `data/parsing/preprocess.py` | 从原始 RGB log / Excel 生成 NPZ 和 index | 可视情况优化 | `data_builder.py` | 当前名强调入口过程，建议名强调数据构建 | Data Builder | 因已有根入口 `preprocess.py`，内部文件不必强改 |
| `data/index/sample_index.py` | 生成 sidecar index | 保留 | `sample_index.py` | 自然清楚 | Sample Index Builder | 不建议再改 |
| `data/splits/grouping.py` | group-aware split 逻辑 | 可保留 | `grouping.py` | 足够自然 | Group-aware Splitter | 若后续拆分更多策略，可考虑 `splitter.py` |
| `inference/pipeline.py` | checkpoint 加载、条件分流、采样、诊断输出 | 保留 | `pipeline.py` | 顶会工程里常见写法 | Inference Pipeline | 不建议再改 |
| `inference/retrieval.py` | retrieval 相关辅助逻辑 | 保留 | `retrieval.py` | 直接明了 | Retrieval Module | 不建议再改 |
| `inference/uncertainty.py` | 多样本采样与置信度 | 保留 | `uncertainty.py` | 清晰自然 | Uncertainty Estimator | 不建议再改 |
| `evaluation/metrics.py` | 评估指标 | 保留 | `metrics.py` | 标准命名 | Metrics | 不建议再改 |
| `evaluation/protocols.py` | 评估协议 | 保留 | `protocols.py` | 标准命名 | Evaluation Protocols | 不建议再改 |
| `evaluation/ablations.py` | 消融入口与组织 | 保留 | `ablations.py` | 简洁自然 | Ablation Suite | 不建议再改 |
| `utils/config_utils.py` | 配置加载、归一化、默认值补齐 | 保留 | `config_utils.py` | 清晰自然 | Config Utilities | 不建议再改 |
| `utils/color_utils.py` | RGB/Lab 转换与色差工具 | 保留 | `color_utils.py` | 清晰自然 | Color Utilities | 不建议再改 |
| `preprocess.py` | 主预处理入口 | 保留 | `preprocess.py` | 自然命名，符合常见项目风格 | Preprocess Entry | 已符合目标风格 |
| `train.py` | 主训练入口 | 保留 | `train.py` | 自然命名 | Train Entry | 已符合目标风格 |
| `infer.py` | 主推理入口 | 保留 | `infer.py` | 自然命名 | Inference Entry | 已符合目标风格 |
| `evaluate.py` | 主评估入口 | 保留 | `evaluate.py` | 自然命名 | Evaluation Entry | 已符合目标风格 |
| `build_prototypes.py` | 构建 prototype bank | 保留 | `build_prototypes.py` | 准确、清晰 | Prototype Builder | 已符合当前方法语义 |
| `pigment_task/build_pigment_lut33.py` | 构建 3D LUT，支持 batch engine/断点续跑 | 建议未来提供规范别名 | `build_lut.py` | 未来主入口建议更自然，不带 task-specific 历史痕迹 | LUT Builder | 当前文件可继续保留为兼容脚本 |
| `t1.py` | 常见颜色/面板测试 | 建议未来提供规范别名 | `palette_panel.py` | 比 `t1` 更可读、更像论文配图脚本 | Palette Panel Test | 当前可继续保留短名 wrapper |
| `t2.py` | 整图聚类调色板恢复 | 建议未来提供规范别名 | `image_palette_restore.py` | 说明输入是整图、方法是 palette restore | Palette-based Image Restore | 当前可继续保留短名 wrapper |
| `t3.py` | 模型驱动的掩膜区颜色先验图 | 建议未来提供规范别名 | `masked_prior_from_model.py` | 能直接看出“掩膜 + 模型生成先验” | Model-driven Mask Prior | 当前可继续保留短名 wrapper |
| `t4.py` | 检查 NPZ/LUT 文件内容 | 建议未来提供规范别名 | `inspect_npz.py` | 标准工具脚本命名 | NPZ Inspector | 很适合未来替代 `t4.py` |
| `t5.py` | LUT 驱动的掩膜区颜色先验图 | 建议未来提供规范别名 | `masked_prior_from_lut.py` | 与 `t3` 形成成对命名 | LUT-driven Mask Prior | 当前可继续保留短名 wrapper |
| `t6.py` | 基于 LUT 的整图颜色替换 | 建议未来提供规范别名 | `full_image_lut.py` | 表意直接 | Full-image LUT Recoloring | 当前可继续保留短名 wrapper |
| `t7.py` | 仅在掩膜区域内应用 LUT | 建议未来提供规范别名 | `masked_lut_apply.py` | 与 `t6` 形成配对命名 | Masked LUT Application | 当前可继续保留短名 wrapper |

## 推荐解释方式

如果后续要在论文或答辩材料里描述代码结构，建议采用“双层命名”：

- 代码里继续使用当前稳定文件名，避免破坏兼容性。
- 文档和图中可以使用更学术、但与代码一一对应的模块术语。

例如：

- `models/color_encoder.py` -> 论文中写作 `Color Encoder`
- `bridge/condition_builder.py` -> 论文中写作 `Condition Bridge`
- `bridge/posterior_head.py` -> 论文中写作 `Posterior Head`
- `bridge/prototype_bank.py` -> 论文中写作 `Prototype Bank`
- `bridge/confidence_gate.py` -> 论文中写作 `Confidence Gate`

## 建议的未来演进

如果未来真的要进一步整理命名，建议优先级如下：

1. 优先给 `t1.py` 到 `t7.py` 增加规范别名入口。
2. 若需要统一 bridge 语义，再考虑把 `condition_builder.py` 在文档层面统一叫 `condition_bridge.py`。
3. 除非有充分收益，不建议再改动已经稳定的 `train.py / infer.py / dataset.py / denoiser.py / posterior_head.py / prototype_bank.py`。