# 项目总览（中文）

## 1. 当前项目目标

当前项目目标是：给定颜料褪色后的 RGB / Lab 观测，恢复褪色前的原始颜色。仓库当前保留了 RGB 主链和 diffusion 主体，并通过条件分支将 Raman / XRD 作为 teacher condition、pred condition、retrieval condition，以及 posterior bridge 的候选知识源。

## 2. 当前真实结构概览

当前活跃结构已经整理为：

- 根入口：`preprocess.py`、`train.py`、`infer.py`、`evaluate.py`、`build_prototypes.py`
- 数据层：`data/`
- 模型层：`models/`
- bridge 层：`bridge/`
- 训练层：`training/`
- 推理层：`inference/`
- 评估层：`evaluation/`
- 工具层：`utils/`
- 文档与测试：`docs/`、`tests/`
- 兼容层：`pigment_task/`

## 3. 当前已有能力

- 预处理 RGB log -> pair-based NPZ。
- 多模态条件编码：`models/spectral_encoder.py`。
- RGB -> 条件 embedding 预测：`models/color_encoder.py` + `models/cond_predictor.py`。
- 条件 diffusion 恢复：`models/denoiser.py` + `training/diffusion.py`。
- 推理阶段三条历史路径：`true` / `pred` / `retrieval`。
- posterior/prototype 模块骨架：`bridge/posterior_head.py`、`bridge/prototype_bank.py`、`bridge/confidence_gate.py`、`bridge/distill.py`。

## 4. 当前真实问题

- 当前预处理真实产物仍是 pair-only，不是完整 sequence schema。
- 历史文档把 sequence 已落地写得过于超前。
- `meta_t` 与旧 dataset 声明存在不一致，现在已通过新 dataset 兼容。
- left/right、augmentation parent、spectral parent 过去没有显式 sidecar 表达。
- 真实 Raman/XRD Excel 可能是单 `Sheet1` 宽表，而不是按材料分 sheet；原始 loader 无法直接覆盖这类文件。
- 顶层仍有历史实验脚本 `t1.py`-`t7.py`，它们属于 legacy 资产而不是当前主入口。

## 5. 为什么不能简单 concat

在当前数据条件下，简单把 RGB 特征和谱 embedding 直接 concat 存在三个问题：

1. 训练时有真谱，推理时没有真谱，训练/推理条件不一致。
2. RGB 增强到 2000+，而 Raman/XRD 没有同步增强，many-RGB-to-one-spectrum 关系会被随机打散。
3. 样本极少，直接回归单个谱 embedding 很容易退化成“训练时看过谱、推理时只剩 RGB shortcut”。

因此更合理的方向是让 RGB 在推理时显式走一条“伪谱条件”路径，而不是训练时用过谱、推理时完全消失。

## 6. 为什么选择 posterior / prototype / retrieval

当前 GSPB 方向的核心是：

- 使用真实谱条件构建 train-fold prototype bank；
- 让 RGB 学习 posterior，而不是只回归一个点 embedding；
- 推理时由 RGB 生成 posterior，再聚合 prototype cond，得到 pseudo spectral condition；
- 必要时与 retrieval 条件通过 gate 融合，显式保留谱知识在推理阶段的作用。

## 7. 为什么“训练期多模态、推理期 RGB-only”仍然成立

这里的关键不是“推理时访问真谱”，而是：

- 训练期用真谱教出 spectral condition space；
- 推理期用 RGB 进入这个 condition space；
- posterior / retrieval / pseudo spectral cond 继续作为条件进入 diffusion，而不是在推理时消失。

因此最终推理链仍然满足 RGB-only 约束。

## 8. 当前数据限制及其含义

已确认的真实限制：

- 原始 patch/材料规模很小；
- left/right 对照存在，但过去没有被显式 sidecar 建模；
- RGB 增强多，谱增强少；
- 真实 Excel 可能是宽表而不是多 sheet；
- 当前代码历史上把这些约束表达得不充分。

这些限制意味着：

- split 必须 group-aware；
- prototype bank 只能用 train fold 构建；
- 需要 sidecar index 才能做 leakage guard；
- 任何“随机打散增强样本”的实验结论都需要谨慎修正。
