# 推理与结构增强消融建议

## 1. 前提

做消融前先确认两件事：

- 当前实验是 `full` 还是 `partial`
- 推理对照时优先同时看：
  - `color_prior.png`
  - `raw_pred.png`
  - `final.png`
  - `training_target_like.png`

不要在 `full` 模式下用“已知区必须不变”的标准评价，也不要只拿原始 `gt.png` 作为唯一参照。

## 2. 推荐主实验

建议至少保留以下组：

1. 官方原始纹理推理
2. BrushNet 增强推理
3. BrushNet + MGLC-Tex
4. BrushNet + MGLC-Tex + restore_S_guidance
5. BrushNet + MGLC-Tex + restore_S_guidance + Mu-Denoiser

如需强调官方兼容性，再单独加：

6. `discriminator_guidance=false`
7. `discriminator_guidance=true`

## 3. 双模式评价建议

### `full`

关注：

- 整图颜色恢复是否更接近训练目标分布
- `final` 是否比 `training_target_like` 更接近
- 结构修复是否自然

### `partial`

关注：

- 已知区是否基本等于原输入
- hole 区是否明显优于 `color_prior`
- `raw_pred` 与 `final` 的差异是否只来自已知区回填

## 4. 推荐消融顺序

### 4.1 纹理增强主消融

建议顺序：

1. `texture_core.enabled=false`
2. `texture_core.enabled=true, insert_mid=true, insert_dec=false`
3. `texture_core.enabled=true, insert_mid=true, insert_dec=true`

回答的问题：

- MGLC 是否有效
- 提升主要来自 bottleneck 插点还是 decoder 插点

### 4.2 backend 消融

建议顺序：

1. `backend=conv_surrogate`
2. `backend=sem_lite`

回答的问题：

- 轻量语义上下文分支是否比卷积代理分支更有效

### 4.3 branch 消融

建议顺序：

1. `branch_mode=local_only`
2. `branch_mode=context_only`
3. `branch_mode=both`

回答的问题：

- 提升主要来自局部纹理连续性还是上下文纹理感知

### 4.4 gate 消融

建议顺序：

1. `use_mask_gate=false`
2. `use_mask_gate=true`

回答的问题：

- boundary-aware 门控是否真的有帮助

### 4.5 结构与条件链消融

建议顺序：

1. `restore_S_guidance=false`
2. `restore_S_guidance=true`
3. `mu_denoiser.enabled=false`
4. `mu_denoiser.enabled=true`

回答的问题：

- 结构引导是否稳定提升纹理恢复
- Mu-Denoiser 是否改善条件噪声而不是破坏主干修复

### 4.6 prior 生成方法消融

建议顺序：

1. `datasets.test.prior_method=fast`
2. `datasets.test.prior_method=quality`

回答的问题：

- 当前 bad case 是模型没学到，还是 `color_prior` 本身太弱

## 5. 判别器引导的定位

`inference.discriminator_guidance.enabled` 建议单独写成官方兼容对照项：

1. `false`
2. `true`

这一项更适合表述为：

- 官方推理兼容项
- 最终结果增强策略对照项

不建议把它写成第三创新点主体。

## 6. 常用命令

关闭 MGLC：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false
```

只保留 mid 插点：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.insert_dec=false
```

切到 `conv_surrogate`：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.backend=conv_surrogate
```

只保留 local branch：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.branch_mode=local_only
```

关闭结构引导：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set restore_S_guidance=false
```

关闭 Mu-Denoiser：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set mu_denoiser.enabled=false
```

切换 prior 方法：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set datasets.test.prior_method=fast
python test.py -opt options/test/ir-sde-brushnet.yml --set datasets.test.prior_method=quality
```

切到 `partial`：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set datasets.test.gt_mode=partial
```

## 7. 论文表述建议

建议这样区分：

- `restore_S_guidance`
  官方结构链兼容项 / baseline alignment

- `discriminator_guidance`
  官方推理兼容项

- `MGLC-Tex`
  第三个创新点主体

- `Mu-Denoiser`
  条件均值清理模块，属于辅助增强模块

- `prior_method`
  先验生成质量控制项，不是主干结构创新
