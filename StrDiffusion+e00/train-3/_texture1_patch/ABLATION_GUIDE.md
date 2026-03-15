# 推理与结构增强消融建议

## 1. 目标

这份文档给出后续论文实验建议，重点围绕：

- 最终增强版是否有效
- 提升来自哪一部分
- 哪些模块是 baseline compatibility，哪些属于创新增强

## 2. 推荐主实验

建议至少保留以下组：

1. 官方原始纹理推理
2. BrushNet 增强推理
3. BrushNet + MGLC-Tex
4. BrushNet + MGLC-Tex + restore_S_guidance
5. BrushNet + MGLC-Tex + restore_S_guidance + Mu-Denoiser

## 3. 推荐消融顺序

### 3.1 纹理增强主消融

建议顺序：

1. `texture_core.enabled=false`
2. `texture_core.enabled=true, insert_mid=true, insert_dec=false`
3. `texture_core.enabled=true, insert_mid=true, insert_dec=true`

要回答的问题：

- MGLC 是否有效
- 提升主要来自 bottleneck 插点还是 decoder 插点

### 3.2 backend 消融

建议顺序：

1. `backend=conv_surrogate`
2. `backend=sem_lite`

要回答的问题：

- 轻量语义上下文分支是否比卷积代理分支更有效

### 3.3 branch 消融

建议顺序：

1. `branch_mode=local_only`
2. `branch_mode=context_only`
3. `branch_mode=both`

要回答的问题：

- 提升主要来自局部纹理连续性还是上下文纹理感知

### 3.4 gate 消融

建议顺序：

1. `use_mask_gate=false`
2. `use_mask_gate=true`

要回答的问题：

- boundary-aware 的门控是否真的有帮助

### 3.5 结构与条件链消融

建议顺序：

1. `restore_S_guidance=false`
2. `restore_S_guidance=true`
3. `mu_denoiser.enabled=false`
4. `mu_denoiser.enabled=true`

要回答的问题：

- 结构引导是否稳定提升纹理恢复
- Mu-Denoiser 是否能稳定改善颜色与噪声条件

## 4. 判别器引导的定位

`inference.discriminator_guidance.enabled` 建议单独做对照：

1. `false`
2. `true`

这一项更适合写成：

- 官方推理兼容项
- 推理增强策略对照项

不建议把它作为第三创新点主体。

## 5. 配置示例

关闭 MGLC：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.enabled=false
```

只保留 mid 插点：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.insert_dec=false
```

切到 conv_surrogate：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.backend=conv_surrogate
```

只保留 local branch：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set texture_core.branch_mode=local_only
```

关闭 restore_S_guidance：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set restore_S_guidance=false
```

关闭 Mu-Denoiser：

```bash
python test.py -opt options/test/ir-sde-brushnet.yml --set mu_denoiser.enabled=false
```

## 6. 论文表述建议

建议在论文里这样区分：

- `restore_S_guidance`
  baseline compatibility / official-structure alignment

- `discriminator_guidance`
  official inference strategy compatibility item

- `MGLC-Tex`
  最核心的第三创新点

- `Mu-Denoiser`
  条件均值清理模块，属于辅助增强模块
