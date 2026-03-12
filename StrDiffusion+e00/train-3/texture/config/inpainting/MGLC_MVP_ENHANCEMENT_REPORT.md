# MGLC-Tex MVP 跑通后的增强路线报告

## 1. 报告目标

本报告说明：

1. 在 MGLC-Tex MVP 跑通之后，下一步应该如何增强
2. 各增强项的优先级、目标、收益与风险
3. 什么应该先做，什么必须后做

这里的“跑通”至少指：

- 训练链正常启动
- `texture_core.enabled=true` 时完整前向可运行
- baseline 和 `Baseline + MGLC(mid-only)` 能产出可比较结果


## 2. 增强路线总原则

增强必须继续遵守当前的职责边界：

> BrushNet 继续负责条件注入；MGLC-Tex 只负责 active texture 主干的纹理建模增强。

因此后续增强不应演变成：

- 新增第二条 ControlNet/BrushNet 式条件分支
- 重复消费 `color_prior + confidence` 再做一遍条件生成
- 用更多条件链掩盖主干结构改造的贡献


## 3. 推荐的增强顺序

推荐顺序固定为 4 个阶段：

1. 先做稳定性与有效性验证
2. 再做结构增强
3. 再做 baseline-fix 对照
4. 最后再做 loss / 训练策略增强

原因很直接：

- 先确认 MVP 本身有增益
- 再增加复杂度
- 避免把多个变量混在一起导致实验不可解释


## 4. 第一阶段：先做“跑通后的必要验证”

这一阶段不是新增创新，而是把 MVP 结果做扎实。

### 4.1 必做实验

至少做以下 4 组：

1. `Baseline`
2. `Baseline + MGLC(mid-only)`
3. `MGLC(mid-only, no gate)`
4. `MGLC(mid-only, local-only / context-only)`

其中第 4 组当前代码还没有单独开关，因此建议在下一小步增加最小实验开关，而不是立刻改大结构。


### 4.2 关注指标

建议同时看两类指标：

1. 通用图像质量指标
   - PSNR
   - SSIM
   - LPIPS（如果环境允许）

2. 任务特定观察
   - repair 区域中心纹理是否更自然
   - boundary band 是否过渡更连续
   - 是否出现新纹理振铃或局部重复


### 4.3 第一阶段的成功标准

至少满足其中两条：

1. `Baseline + MGLC(mid-only)` 对比 baseline 有稳定正增益
2. 无 gate 版本弱于有 gate 版本，说明 gate 有实际贡献
3. local-only / context-only 单独都不如双分支版本，说明结构设计合理

若这些结论成立，才适合进入下一阶段。


## 5. 第二阶段：结构增强

这是 MVP 之后最值得做的增强阶段。

### 5.1 V1：增加 branch 消融开关

建议先加最小配置：

- `branch_mode: both | local_only | context_only`

目的：

- 补齐当前实验矩阵
- 精确回答“性能增益主要来自哪一类分支”

这一步优先级很高，因为它成本低、解释力强。


### 5.2 V2：将 backend 从 `conv_surrogate` 升级到 `sem_lite`

这是最优先的结构增强路线。

目标：

- 保持 MGLC 的职责边界不变
- 只升级 context branch 的建模能力

建议做法：

- 保留 local branch 不动
- 只替换 context branch
- 将 `backend` 扩展为：
  - `conv_surrogate`
  - `sem_lite`

建议 `sem_lite` 的设计原则：

- 不引入重量级外部依赖
- 尽量保持与当前 channel shape 完全兼容
- 不改变输入输出接口

何时做：

- 只有在 MVP 已证明“context branch 确实贡献显著”之后再做


### 5.3 V3：增加 decoder 插点 `mglc_dec`

这属于第二优先级结构增强。

设计建议：

- 只增加一个 decoder 末端插点
- 不要一上来在多层 decoder 全部插

推荐位置：

- 接近输出侧、分辨率较高的一个 decoder stage

目的：

- 强化高分辨率纹理与边界过渡

风险：

- 更容易与局部结构细节耦合
- 也更容易引入 checkerboard 或细纹伪影

因此建议把它放在 `sem_lite` 之后，而不是之前。


## 6. 第三阶段：baseline-fix 与对照增强

### 6.1 `restore_S_guidance`

当前应明确把它视为：

- baseline compatibility fix
- 不是第三创新点的一部分

什么时候做：

- 在 MVP 结构已稳定之后
- 作为独立 patch 或独立实验组加入

建议实验方式：

1. baseline
2. baseline + restore_S_guidance
3. baseline + MGLC
4. baseline + restore_S_guidance + MGLC

这样可以避免把 wrapper 原本缺失的 `S` 引导收益误记到 MGLC 上。


### 6.2 test / inference tree 同步

当训练侧收益确定后，才同步到推理侧。

建议顺序：

1. 先确认训练 checkpoint 加载正常
2. 再复制/同步推理侧 wrapper 和 model factory
3. 最后做单张图推理与可视化评估

不要在 MVP 刚落地时就同步推理树，否则会同时引入两类问题。


## 7. 第四阶段：训练目标与损失增强

这一阶段应该放到最后。

### 7.1 边界辅助损失

这是后续最值得考虑的 loss 增强。

目标：

- 把 MGLC 的 boundary-aware 设计进一步外化成训练约束

建议方法：

- 在 boundary band 上增加额外一致性或重建约束
- loss 只作用在边界带区域，不覆盖全图

原因：

- MGLC 的 gate 本来就在区分 boundary 和 repair center
- 边界损失能和模块设计形成闭环

前提：

- 先确认结构本身有效


### 7.2 prior consistency loss

这是可选增强，不是首选。

理由：

- 当前主链的条件已经较多
- 过早引入 prior consistency loss，容易把“主干结构增强”混成“训练技巧增强”

建议：

- 等结构增强与 baseline-fix 对照都完成后，再考虑加入


## 8. 不推荐过早做的增强

以下内容不建议在 MVP 刚跑通后立即进入：

1. 真 Mamba / 真 SSM 重型依赖
2. 多个 decoder 插点同时引入
3. patch-level refinement
4. 复杂多项 loss 联合改造
5. 再造一条 side branch
6. 改动 `PixelBrushNet`
7. 改动 `denoising_model.py` 主训练逻辑

原因：

- 会显著提高调试成本
- 破坏实验可解释性
- 容易模糊第三创新点的边界


## 9. 推荐的增强版本规划

建议按以下版本推进：

### V1：实验完备版

新增：

- `branch_mode` 开关
- 完整消融实验

目标：

- 回答 local / context / gate 的真实贡献


### V2：结构增强版

新增：

- `backend=sem_lite`

保留：

- mid-only 插点
- 其余链路不变

目标：

- 提升 context branch 的表达能力


### V3：双插点版

新增：

- `mglc_dec`

目标：

- 在高分辨率阶段强化纹理与边界表现


### V4：对照闭环版

新增：

- `restore_S_guidance` baseline-fix 对照
- test / inference tree 同步

目标：

- 把 MGLC 的真实收益与 wrapper 兼容性修复剥离开


### V5：训练增强版

新增：

- boundary auxiliary loss
- 可选 prior consistency loss

目标：

- 在结构稳定后进一步提升边界与细节质量


## 10. 建议的下一步执行顺序

如果 MVP 已经能完整跑通，建议立刻按下面顺序执行：

1. 跑 `Baseline` 与 `Baseline + MGLC(mid-only)` 两组
2. 若有效，再补 `no gate` 与 `local/context` 消融
3. 增加 `branch_mode` 开关，完成实验矩阵
4. 结构侧升级 `backend=sem_lite`
5. 再考虑加入 `mglc_dec`
6. 最后才做 `restore_S_guidance` 对照与边界损失

这个顺序能最大限度保证：

- 结论清晰
- 风险可控
- 每一步都能解释“为什么提升”


## 11. 结论

MVP 跑通之后，最合理的增强路线不是“立刻做更大”，而是：

1. 先把当前 MGLC 结构的真实贡献验证清楚
2. 再升级 context branch
3. 再扩展插点
4. 最后才做 baseline-fix 与 loss 增强

换句话说，后续增强的核心不是堆更多模块，而是沿着当前已经明确的结构边界，逐步扩大 `MGLC-Tex` 的表达能力与实验说服力。
