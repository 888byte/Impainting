# 相对原始 SSD-TS 的改动对比表

> 说明：本表以“原始 SSD-TS”作为 baseline，以当前颜料恢复项目作为当前实现。`代码证据` 列给的是当前仓库中的主要实现位置。`是否属于核心创新` 不是绝对学术判断，而是服务于后续论文写作分层。

| 改动类别 | baseline（原始 SSD-TS） | 当前实现 | 代码证据 | 是否属于核心创新 | 对训练的影响 | 对推理的影响 | 论文写作建议标签 |
|---|---|---|---|---|---|---|---|
| 任务目标 | 通用时序缺失值插补 | 颜料褪色前颜色恢复 | `imputers/SSMImputer.py`、`config/config_bissm2_mujoco_90.json`、`training/trainer.py` | 否 | 重新定义样本语义与损失目标 | 重新定义输入输出语义 | Task Redefinition |
| 样本表示 | 通用时间序列段 + 缺失 mask | pair-only 颜色样本，`x0[:,1,:]` 为当前颜色，`x0[:,0,:]` 为原始颜色 | `data/parsing/preprocess.py`、`data/dataset.py` | 否 | 训练从序列插补转为颜色对恢复 | 推理从补缺失值转为恢复原始颜色 | Problem Formulation |
| Raman/XRD 接入 | 无 | 支持真实 Raman/XRD 作为训练期 teacher condition | `models/spectral_encoder.py`、`training/trainer.py` | 是 | 增加多模态条件编码与 teacher signal | 真谱不能直接用于正式推理 | Core Method |
| 颜色编码分支 | 无 | `ColorEncoder` 从当前颜色提取 `z_color` | `models/color_encoder.py` | 是 | 为 missing-modality 学习提供颜色侧表征 | 支持 RGB-only 条件推理 | Core Method |
| 条件预测器 | 无 | `ColorToSpecPredictor` 从 `z_color` 预测谱 embedding | `models/cond_predictor.py`、`bridge/condition_builder.py` | 是 | 增加 pred condition 监督 | 形成 `pred` 推理路径 | Core Method |
| true / pred / retrieval 路径 | 无 | 统一支持 `true` / `pred` / `retrieval` | `bridge/condition_builder.py`、`inference/pipeline.py` | 是 | 支持 teacher、pred 对齐与 retrieval 对照 | 推理可在无真谱时显式走 bridge | Core Method |
| posterior head | 无 | RGB 预测 prototype posterior，而不是只回归单点 embedding | `bridge/posterior_head.py`、`training/trainer.py` | 是 | 新增 posterior KL / teacher posterior 监督 | 支持 `posterior` 推理 | Core Method |
| prototype bank | 无 | 从 train fold 真实谱条件聚合 prototype memory | `bridge/prototype_bank.py`、`build_prototypes.py` | 是 | 训练和验证需严格按 train-fold bank 使用 | 推理可聚合 pseudo spectral condition | Core Method |
| posterior + retrieval + gate | 无 | `posterior_retrieval` 联合，并由 confidence gate 融合 | `bridge/confidence_gate.py`、`bridge/condition_builder.py` | 是 | 训练可兼容多路径与 distillation | 推理更稳，不再只信单一路径 | Core Method |
| distillation | 无 | 用真实谱条件导出的 teacher 信息约束 RGB 分支 | `bridge/distill.py`、`training/trainer.py` | 是 | 增加 KL / embedding distill | 间接改善 RGB-only 推理 | Core Method |
| 数据 sidecar index | 无 | 使用 `sample_index.csv` 补充 `side` / `spectral_parent_id` / `augmentation_parent_id` 等 | `data/index/sample_index.py`、`data/dataset.py` | 否 | 训练可获取 group/runtime metadata | 推理本身不直接依赖 | Supporting Design |
| group-aware split | 一般随机或通用掩码协议 | 支持 `group_exp_patch` 等 group-aware 划分 | `data/splits/grouping.py`、`data/parsing/preprocess.py` | 否 | 降低 augmentation leakage 风险 | 提高评估可信度 | Experimental Protocol |
| parent-aware sampler | 无 | 可按 parent 采样/加权，避免 many-RGB-to-one-spectrum 偏置 | `training/samplers.py`、`training/trainer.py` | 否 | 训练 batch 分布更合理 | 不直接影响单次推理接口 | Supporting Design |
| 宽表 Excel 适配 | 无颜料 Excel 解析需求 | 支持 `meta_json` + 单 `Sheet1` 宽表自动兜底 | `data/parsing/preprocess.py`、`utils/io_utils.py` | 否 | 提高真实实验文件可用性 | 不直接影响推理接口 | Engineering Support |
| pair-only 与 `meta_t` 兼容 | baseline 无此问题 | 显式兼容 `meta_t` 与历史字段差异 | `data/dataset.py` | 否 | 修正训练数据读取 | 保持历史数据兼容 | Engineering Support |
| 旧 cycle physics | 无针对颜料恢复的物理模块 | 兼容保留 `FadingForwardModelLab` | `models/physics.py`、`training/trainer.py` | 否 | 可作为旧版 physics 路径 | 不改变默认推理接口 | Legacy-Compatible Support |
| Spec-Color Consistency | 无 | 用 `spec_color_head(pseudo_cond)` 对齐真实 `x0`，可选弱对齐 `x0_pred.detach()` | `bridge/physics_heads.py`、`training/physics_losses.py` | 是 | 增加轻量 physics 正则 | 仅可输出诊断，不依赖真谱 | Core Method |
| Parent / Augmentation Consistency | 无 | 对同 parent 样本的 posterior / latent 做一致性约束 | `training/physics_losses.py` | 是 | 降低 RGB 扩增导致的谱条件漂移 | 不改主推理接口 | Core Method |
| Damage Score Constraint | 无 | 引入轻量 `damage_head` 与顺序约束接口 | `bridge/physics_heads.py`、`training/physics_losses.py` | 部分 | 仅在有顺序元信息时启用 | 可选输出 `damage_score` 诊断 | Supporting Method |
| 低置信软加权 physics | 无 | physics loss 使用 `detach()` 后的 confidence 软加权 | `training/physics_losses.py`、`training/trainer.py` | 否 | 减少模型靠压低 confidence 逃避正则 | 不改变主推理链 | Stability Design |
| 不确定性与诊断输出 | 无 | 输出 posterior entropy、bridge confidence、physics diagnostics | `inference/pipeline.py` | 否 | 可辅助验证和消融 | 推理可给出更丰富诊断 | Supporting Design |
| 单点推理稳定化 | 无 | 低置信回拉、颜色漂移约束、Kalman/RTS 平滑 | `inference/pipeline.py` | 否 | 不影响训练 | 提升单点/面板/LUT 结果稳定性 | Inference Stabilization |
| LUT 生成器 | baseline 无 LUT | 生成 `lut_rgb/lut_lab/lut_conf/lut_std/lut_cdiff/lut_cret` | `pigment_task/build_pigment_lut33.py` | 否 | 不影响训练 | 形成部署与图像实验的缓存中间件 | Engineering Support |
| LUT 引擎加速 | 无 | 从逐点 subprocess 改为默认 batch engine，支持 ETA/断点续跑 | `pigment_task/build_pigment_lut33.py` | 否 | 不影响训练 | 推理效率显著提升 | Engineering Optimization |
| 自然命名入口 | baseline 的入口面向原始 SSD-TS 任务 | 统一为 `preprocess.py/train.py/infer.py/evaluate.py/build_prototypes.py` | 根目录主入口、`training/trainer.py`、`inference/pipeline.py` | 否 | 降低使用复杂度 | 降低实验/复现实验门槛 | Engineering Cleanup |
| legacy wrapper 保留 | 无该颜料兼容层 | `pigment_task/` 作为兼容层，旧命令不立即删除 | `pigment_task/` | 否 | 避免训练脚本历史漂移导致中断 | 保持旧推理/脚本可用 | Compatibility Support |
| t1-t7 实验脚本收敛 | baseline 无这一组脚本 | 统一对接当前 infer/LUT 语义 | `t1.py`、`t2.py`、`t3.py`、`t4.py`、`t5.py`、`t6.py`、`t7.py` | 否 | 不影响训练 | 形成可视化与图像实验链 | Experimental Tooling |
| smoke tests / docs | baseline 不面向当前颜料任务 | 增加 CLI、bridge、physics、preprocess、LUT 相关保护 | `tests/`、`docs/` | 否 | 降低重构破坏风险 | 提高可复现性与可维护性 | Engineering Support |

## 使用建议

在论文写作时，可以按下面的层级引用本表：

- 方法章节优先写：Raman/XRD 多模态条件、RGB-only bridge、posterior/prototype/retrieval/gate、physics soft constraints。
- 实验设置章节重点写：sidecar index、group-aware split、prototype bank 只用 train fold、pair-only 真实数据约束。
- 工程实现章节再写：自然命名入口、legacy wrapper、LUT batch engine、tests/docs 整理。

## 必须保持诚实的点

- 当前真实训练数据仍是 pair-only，不是完整 sequence 训练。
- `true` 路径不能作为正式推理链。
- 宽表 Excel 适配、physics soft constraints、batch LUT 引擎都属于后续增强，不应写成 baseline 原生能力。