# StrDiffusion + BrushNet + MGLC-Tex 项目总说明

## 1. 项目定位

这是一个面向壁画褪色修复与缺损修补的训练项目，核心目标是同时解决两类问题：

1. **颜色恢复**：把退化、偏色、褪色的图像颜色恢复到目标分布。
2. **区域修补**：对缺损或需要修复的区域完成结构与纹理补全。

当前项目最终采用的是一个两阶段框架：

- **Stage 1**：生成颜色先验 `color_prior` 和置信度 `confidence`
- **Stage 2**：由 StrDiffusion / IR-SDE 主干在 BrushNet 与 MGLC-Tex 的辅助下完成最终修复


## 2. 最终增强版的核心结构

当前最终增强版的 active generator 为：

- `models/brushnet_wrapper.py::ConditionalUNetWithBrushNet`

该版本在 MVP 基础上进一步增强为：

1. **BrushNet 条件注入**
   - 输入：`mask + color_prior + confidence`
   - 输出：多尺度 `down_features + mid_feature`
   - 作用：负责把外部条件提示注入主干

2. **MGLC-Tex 主干纹理增强**
   - 插入位置：
     - bottleneck mid
     - decoder 高分辨率阶段
   - 作用：只增强已经融合条件后的主干 feature，不重复做条件生成

3. **restore_S_guidance（可选）**
   - 用于恢复 legacy `S` 引导路径
   - 这是 **baseline compatibility fix**
   - 不作为第三创新点本身


## 3. 当前版本的主要创新点

如果你写论文，建议把创新点组织成下面的结构。

### 创新点 1：颜色先验驱动的两阶段修复框架

项目不是单纯的 inpainting，而是先做颜色先验构建，再做扩散修复：

- Stage 1 先生成 `color_prior` 与 `confidence`
- Stage 2 再让扩散模型在颜色提示下完成恢复

优点：

- 先把颜色恢复目标显式化
- 降低扩散模型直接从退化颜色端到端恢复的难度
- 更适合壁画这类“颜色退化 + 局部缺损”混合问题


### 创新点 2：Pixel-space BrushNet 引导的 StrDiffusion 修复

当前 BrushNet 不是 latent 空间版本，而是像素空间版本：

- `PixelBrushNet` 直接在像素空间处理 `noisy + mask + prior + confidence`
- 生成 `down_features + mid_feature`
- 在主 U-Net encoder 与 bottleneck 中做 additive fusion

这一点可以强调为：

- 将 BrushNet 条件控制机制与 StrDiffusion 有机结合
- 避免重新设计第二套条件扩散框架
- 保持主干结构清晰、条件链职责单一


### 创新点 3：MGLC-Tex 纹理核心增强模块

第三创新点建议作为你论文里的重点：

- 中文名：**掩膜门控局部连续纹理核心**
- 英文名：**Mask-Gated Locality-Continuity Texture Core**
- 缩写：**MGLC-Tex**

它的核心特征有：

1. **只处理主干特征，不重复生成条件**
2. **使用 mask 与 boundary band 做门控**
3. **同时建模局部连续纹理与上下文语义纹理**
4. **支持 mid + decoder 双插点**
5. **支持 `conv_surrogate` 与 `sem_lite` 两类 context backend**

当前最终增强版默认使用：

- `backend = sem_lite`
- `branch_mode = both`
- `insert_mid = true`
- `insert_dec = true`


## 4. MGLC-Tex 的实现逻辑

### 4.1 输入输出

`MGLCBlock` 输入：

- `feat: [B, C, h, w]`
- `mask: [B, 1, H, W]`

输出：

- `feat_out: [B, C, h, w]`

说明：

- `mask` 语义沿用 BrushNet 路径：`1 = repair region`
- block 内部自行 resize mask 到 feature 尺寸


### 4.2 模块组成

MGLC-Tex 由 3 部分组成：

1. **Local branch**
   - 负责局部连续纹理建模

2. **Context branch**
   - `conv_surrogate`：卷积版大感受野上下文
   - `sem_lite`：轻量语义上下文分支，增强全局与长程纹理感知

3. **Mask gate**
   - 输入 `mask_resized + boundary_band`
   - 输出 `g_local` 和 `g_ctx`
   - 控制不同位置更依赖局部纹理还是上下文纹理


### 4.3 Boundary band 的作用

MGLC-Tex 不是只看 repair region，还显式构造了 boundary band：

- repair 区域中心更偏向 context 建模
- 边界带更强调局部连续和自然过渡

这对于壁画修复非常重要，因为边界过渡是否自然，往往比单点颜色是否正确更影响主观效果。


### 4.4 最终增强版比 MVP 多了什么

在 MVP 基础上，最终增强版新增了 4 个能力：

1. **`sem_lite` backend**
2. **`branch_mode` 消融开关**
3. **decoder 插点 `mglc_dec`**
4. **`restore_S_guidance` 兼容开关**

因此当前版本比 MVP 更适合做：

- 最终模型训练
- 消融实验
- 论文中的方法主实验


## 5. 关键代码路径

你可以按下面这条主链理解整个系统：

1. `train.py`
   - 训练入口
   - 读取配置
   - 构造数据集、SDE、model

2. `data/mural_inpainting_dataset.py`
   - 生成 `degraded / GT / mask / color_prior / confidence`

3. `models/denoising_model.py`
   - 组织训练逻辑
   - 构造 `brushnet_kwargs`
   - 把 `mask / color_prior / confidence` 传入 active generator

4. `models/networks.py`
   - 根据配置实例化当前 active generator

5. `models/brushnet_wrapper.py`
   - 真实运行的主干
   - 融合 BrushNet、MGLC-Tex、可选 `S` guidance

6. `models/modules/mglc_block.py`
   - 纹理增强模块的核心实现


## 6. 当前项目结构建议理解方式

建议把项目按 5 层理解：

### 第 1 层：训练入口层

- `train.py`
- `options.py`
- `options/train/ir-sde-brushnet.yml`

作用：

- 决定怎么启动训练
- 决定当前启用哪条模型路径


### 第 2 层：数据与先验层

- `data/mural_inpainting_dataset.py`
- `color_prior_generator.py`
- `lut_processor.py`

作用：

- 生成颜色先验与置信度
- 提供 repair mask
- 决定训练目标图像的构造方式


### 第 3 层：主模型层

- `models/denoising_model.py`
- `models/networks.py`
- `models/brushnet_wrapper.py`

作用：

- 组织扩散训练
- 组装 active generator


### 第 4 层：条件控制层

- `models/pixel_brushnet.py`

作用：

- 负责外部条件注入
- 不负责最终纹理重建


### 第 5 层：纹理增强层

- `models/modules/mglc_block.py`

作用：

- 对主干内部 feature 做纹理增强
- 是最终增强版的第三创新点核心


## 7. 如何使用当前最终增强版

### 7.1 默认训练

当前增强版默认配置文件：

- `options/train/ir-sde-brushnet.yml`

训练命令建议在 `texture/config/inpainting` 目录下执行：

```bash
python train.py -opt options/train/ir-sde-brushnet.yml
```


### 7.2 关键配置项

#### Texture core

```yaml
texture_core:
  enabled: true
  insert_mid: true
  insert_dec: true
  backend: sem_lite
  branch_mode: both
  use_mask_gate: true
```

含义：

- `enabled`
  - 是否启用 MGLC-Tex

- `insert_mid`
  - 是否在 bottleneck 插入

- `insert_dec`
  - 是否在 decoder 高分辨率阶段插入

- `backend`
  - `conv_surrogate`
  - `sem_lite`

- `branch_mode`
  - `both`
  - `local_only`
  - `context_only`

- `use_mask_gate`
  - 是否使用 mask/boundary 驱动的空间门控


### 7.3 `restore_S_guidance`

```yaml
restore_S_guidance: true
```

说明：

- 这是 baseline compatibility fix
- 用于恢复旧版 `S` 引导
- 建议在论文中注明：
  - 它不是 MGLC-Tex 本身的创新
  - 它只用于保证基线与增强模型对比公平


## 8. 如何做论文实验

建议最少跑以下几组：

1. `Baseline`
2. `Baseline + restore_S_guidance`
3. `Baseline + MGLC(mid-only)`
4. `Baseline + MGLC(mid + dec)`
5. `MGLC(backend=conv_surrogate)`
6. `MGLC(backend=sem_lite)`
7. `MGLC(branch_mode=local_only)`
8. `MGLC(branch_mode=context_only)`
9. `MGLC(use_mask_gate=false)`

这样你可以清楚回答：

- MGLC 是否有效
- decoder 插点是否有效
- `sem_lite` 是否比 `conv_surrogate` 更强
- local 与 context 分支分别贡献多少
- gate 是否真的有意义
- `restore_S_guidance` 与 MGLC 的收益是否可区分


## 9. 论文写作建议

### 9.1 方法部分推荐表述

建议把方法写成三段：

1. **颜色先验构建**
   - 说明为什么需要 `color_prior + confidence`

2. **BrushNet 引导扩散修复**
   - 说明条件如何进入主干

3. **MGLC-Tex 纹理增强**
   - 说明局部 / 上下文 / boundary-aware gate
   - 说明 mid + decoder 双插点
   - 说明 `sem_lite` backend 的意义


### 9.2 创新点写法建议

建议不要把所有模块堆成很多创新点，而是整理成“一个系统 + 一个核心增强模块”。

更合适的写法是：

- 系统层创新：
  - 颜色先验引导的壁画修复两阶段框架

- 结构层创新：
  - Pixel-space BrushNet 与 StrDiffusion 的融合
  - MGLC-Tex 纹理增强核心

- 说明性补充：
  - `restore_S_guidance` 只是兼容性修复，不单列为创新


### 9.3 实验部分建议

实验章节建议分成：

1. 主结果对比
2. 消融实验
3. 可视化分析
4. 边界区域分析

其中“边界区域分析”很适合体现你 gate + boundary band 的设计价值。


## 10. 当前版本与后续工作边界

虽然这是最终增强版，但仍有可以继续扩展的方向：

- 边界辅助 loss
- prior consistency loss
- 推理侧镜像同步
- 更重型的 context backend

但这些不建议在当前论文主版本里继续叠加，否则会降低系统解释性。


## 11. 总结

现在这个项目可以按下面一句话来概括：

> 本项目通过颜色先验引导、Pixel-space BrushNet 条件注入以及 MGLC-Tex 主干纹理增强，在 StrDiffusion 框架下实现了面向壁画图像的颜色恢复与缺损修复。

如果你要写论文，建议以 **MGLC-Tex** 为第三创新点核心展开，而把 `restore_S_guidance` 明确作为 baseline fix 说明，不把它并入创新贡献。
