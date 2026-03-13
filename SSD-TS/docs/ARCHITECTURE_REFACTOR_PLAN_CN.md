# 架构重构计划（中文）

## 当前真实调用链

- 预处理：`preprocess.py` -> `data/parsing/preprocess.py`
- 训练：`train.py` -> `training/trainer.py`
- 推理：`infer.py` -> `inference/pipeline.py`
- 评估：`evaluate.py` -> `inference/pipeline.py` / `evaluation/ablations.py`
- bridge：`bridge/condition_builder.py`

## 与研究结论一致的部分

- 已有 RGB 主链。
- 已有多模态 conditioner。
- 已有 missing-modality 的 `pred/retrieval/true` 雏形。
- 推理阶段已经明确区分“有真谱上界”和“RGB-only 条件路径”。

## 与研究结论不一致的部分

- 数据端真实还是 pair-only，不是 sequence 训练。
- 历史 dataset 声明与真实 NPZ 元信息不一致。
- left/right、parent-aware、augmentation leakage guard 过去没有被正式建模。
- 真实宽表 Excel 与旧 sheet-based loader 不一致。

## 最小侵入式重构方案

- 不重写 denoiser 主体和 diffusion schedule。
- 先迁目录和命名，再抽象 bridge API，再加 posterior/prototype。
- 通过 sidecar index 承载新信息，不修改 NPZ schema。
- 旧 `pigment_task/*` 保留 wrapper，不直接删除。

## 文件迁移策略

- 保留为 canonical：`data/`、`models/`、`bridge/`、`training/`、`inference/`、`evaluation/`、`utils/`
- 保留兼容：`pigment_task/`
- 标记 legacy：顶层 `t1.py`-`t7.py`、原始 SSD-TS 资产目录

## 模块职责边界

- `data/`：原始文件解析、NPZ 视图、split、sidecar index
- `models/`：encoder / denoiser / physics
- `bridge/`：条件构造、prototype bank、posterior、gate、distill
- `training/`：训练循环、loss、sampler
- `inference/`：checkpoint 加载、采样、不确定性、推理流程
- `evaluation/`：指标、协议、palette scan、mining

## 向后兼容策略

- 旧命令继续可用。
- 旧 checkpoint 继续可加载。
- 旧 `missing_modality` 配置继续保留。
- 旧实验脚本不立即删除。

## 当前最高风险点

- 宽表 Excel 适配。
- parent mapping 不完整时的 split 风险。
- posterior/prototype bank 与旧 pred/retrieval 路径的兼容。
