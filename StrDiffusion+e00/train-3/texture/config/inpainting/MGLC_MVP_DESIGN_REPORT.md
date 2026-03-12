# MGLC-Tex MVP 详细设计报告

## 1. 背景与目标

当前 BrushNet 路线下的 active generator 是 `models/brushnet_wrapper.py` 中的 `ConditionalUNetWithBrushNet`。该模型已经完成两件事：

1. 通过 `PixelBrushNet` 消费 `mask + color_prior + confidence`，生成多尺度条件特征。
2. 在主 U-Net 的 down path 和 bottleneck mid 位置做 additive fusion。

本次修改的目标不是再增加一条条件支路，而是在已经完成 BrushNet 条件融合之后，对 bottleneck 特征增加一个最小可行的纹理增强块：

- 名称：`MGLCBlock`
- 位置：`ConditionalUNetWithBrushNet.forward()` 的 bottleneck 末端
- 范围：只做训练侧 MVP
- 不改：`PixelBrushNet`、`denoising_model.py`、loss、legacy `ConditionalUNet`

核心原则：

> BrushNet 负责多尺度条件注入；MGLC-Tex 只负责 active texture 主干 bottleneck 的内部纹理增强。


## 2. 改动范围

本次只涉及以下 4 处：

1. 新增 `models/modules/mglc_block.py`
2. 修改 `models/brushnet_wrapper.py`
3. 修改 `models/networks.py`
4. 修改 `options/train/ir-sde-brushnet.yml`

未改动部分：

- `models/pixel_brushnet.py`
- `models/denoising_model.py`
- `models/modules/DenoisingUNet_arch.py`
- `color_prior_generator.py`


## 3. 新增模块设计

### 3.1 文件

新增文件：`models/modules/mglc_block.py`

该文件中包含 3 个核心实现：

- `_ConvSurrogateBranch`
- `MaskGate`
- `MGLCBlock`


### 3.2 MGLCBlock 输入输出

输入：

- `feat`: `[B, C, h, w]`
- `mask`: `[B, 1, H, W]`

其中 `mask` 语义沿用 wrapper 当前 BrushNet 语义：

- `1 = repair region`
- `0 = known region`

输出：

- `feat_out`: `[B, C, h, w]`

输出 shape 与输入 feature 完全一致。


### 3.3 结构组成

#### 3.3.1 预归一化

`MGLCBlock` 使用 `GroupNorm` 做预归一化。

组数策略：

- 默认上限 `32`
- 若 `channels` 不能被当前组数整除，则逐步减小
- 保证构造时不会出现非法 `GroupNorm`

这样在 `mid_dim` 变化时无需硬编码组数。


#### 3.3.2 Local Branch

Local branch 用于建模局部连续纹理，结构固定为：

1. depthwise `3x3`
2. `GELU`
3. depthwise `5x5`
4. pointwise `1x1`

它的作用是保持局部纹理和局部连续性建模能力。


#### 3.3.3 Context Branch

Context branch 是 MVP 中的 `conv_surrogate` 实现，用于提供更大感受野的上下文纹理信息，结构固定为：

1. pointwise `1x1`
2. `GELU`
3. depthwise `1x7`
4. `GELU`
5. depthwise `7x1`
6. `GELU`
7. depthwise `3x3`, `dilation=3`
8. pointwise `1x1`

该分支不引入真实 SSM / Mamba 依赖，只用卷积 surrogate 实现上下文建模。


#### 3.3.4 Mask Gate

Gate 分支只消费两类信息：

- resize 后的 mask
- 由 mask 构造的 boundary band

输入：

- `mask_resized`: `[B, 1, h, w]`
- `boundary_band`: `[B, 1, h, w]`

拼接后：

- `gate_in = cat([mask_resized, boundary_band], dim=1)`，shape 为 `[B, 2, h, w]`

网络结构：

1. `Conv3x3(2 -> gate_hidden)`
2. `GELU`
3. `Conv1x1(gate_hidden -> 2)`
4. `softmax(dim=1)`

输出两个空间门控图：

- `g_local`: `[B, 1, h, w]`
- `g_ctx`: `[B, 1, h, w]`


### 3.4 边界带构造

MVP 不引入额外依赖，边界带通过 max-pool / min-pool 近似形态学边界：

1. 先将 `mask` resize 到 feature 尺寸
2. 用 max-pool 得到 `dilated`
3. 用 `1 - max_pool(1 - mask)` 得到 `eroded`
4. `boundary_band = clamp(dilated - eroded, 0, 1)`

设计目的：

- repair 区域中心更偏向 context 建模
- repair 与 known 的过渡带更强调局部连续性与边界过渡


### 3.5 残差融合

有 gate 时：

`feat_out = feat + g_local * feat_local + g_ctx * feat_ctx`

无 gate 时：

`feat_out = feat + feat_local + feat_ctx`

这保证：

- 模块本质是残差增强
- 关闭 gate 时可自然退化成无门控版本


### 3.6 初始化策略

当 `zero_init_last = true` 时：

- local branch 最后一个 `1x1` 置零
- context branch 最后一个 `1x1` 置零

目的：

- 更好兼容已有 pretrain
- 避免刚接入时对主干造成过大扰动

gate 分支保持默认初始化，不做零初始化。


### 3.7 容错与参数约束

`MGLCBlock` 中加入了以下保险逻辑：

1. `mask is None` 时直接返回 `feat`
2. `boundary_width < 0` 时在 `__init__` 直接 `ValueError`
3. `backend != "conv_surrogate"` 时在 `__init__` 直接 `NotImplementedError`
4. `boundary_width = 0` 时合法，边界带退化为零图

这样可以避免在第一次 forward 时才暴露配置错误。


## 4. Wrapper 集成设计

### 4.1 文件

修改文件：`models/brushnet_wrapper.py`


### 4.2 构造函数扩展

`ConditionalUNetWithBrushNet.__init__()` 新增参数：

- `texture_core_opt: Optional[dict] = None`

并新增两个成员：

- `self.texture_core_enabled`
- `self.mglc_mid`

启用条件：

- `texture_core.enabled = true`
- `texture_core.insert_mid = true`

仅在同时满足时构造 `self.mglc_mid`。


### 4.3 插入位置

插入点固定且唯一：

```python
x = self.mid_block1(x, t)
x = self.mid_attn(x)
x = self.mid_block2(x, t)

if brushnet_mid is not None:
    x = x + brushnet_mid

if self.texture_core_enabled and self.mglc_mid is not None:
    x = self.mglc_mid(x, mask)
```

也就是说，`MGLCBlock` 处理的是：

- 已经过主干 mid block 的特征
- 也已经融合了 `brushnet_mid` 的特征

这符合职责边界：MGLC 不负责新条件生成，只负责“已融合条件后的 bottleneck 特征增强”。


### 4.4 保持不变的行为

以下行为全部保留：

- `S` 参数位置不变
- `PixelBrushNet` 接口不变
- BrushNet down path 注入逻辑不变
- BrushNet mid 注入逻辑不变
- 返回值格式仍为 `return x, x`


## 5. Model Factory 透传设计

### 5.1 文件

修改文件：`models/networks.py`


### 5.2 改动内容

在 `define_G(opt)` 的 BrushNet 分支中新增：

```python
texture_core_opt = opt.get("texture_core", {})
```

并透传给：

```python
ConditionalUNetWithBrushNet(..., texture_core_opt=texture_core_opt)
```

未改动内容：

- generator 选择逻辑
- discriminator 定义
- legacy 分支


## 6. 配置设计

### 6.1 文件

修改文件：`options/train/ir-sde-brushnet.yml`


### 6.2 新增配置块

```yaml
texture_core:
  enabled: false
  name: MGLCBlock
  insert_mid: true
  backend: conv_surrogate
  use_mask_gate: true
  gate_hidden: 16
  boundary_width: 3
  zero_init_last: true
```

默认行为：

- 默认关闭，不影响现有训练链
- 打开后只在 bottleneck 插入一个 mid-only 模块
- 当前只允许 `backend=conv_surrogate`


## 7. 数据流与职责边界

当前完整链路如下：

1. 数据集或 `ColorPriorGenerator` 提供 `mask / color_prior / confidence`
2. `denoising_model.py` 构造 `brushnet_kwargs`
3. `sde.noise_fn(..., **brushnet_kwargs)` 把它们传给 active generator
4. `ConditionalUNetWithBrushNet` 调用 `PixelBrushNet`
5. `PixelBrushNet` 生成 `down_features + mid_feature`
6. wrapper 在 encoder 和 bottleneck 融合这些特征
7. 新增的 `MGLCBlock` 只在 bottleneck 末端进一步增强特征

职责边界明确为：

- BrushNet：负责条件注入
- MGLC-Tex：负责 bottleneck 内部纹理增强


## 8. 已完成验证

当前已完成的验证：

1. `mglc_block.py`、`brushnet_wrapper.py`、`networks.py` 已通过 `py_compile`
2. `MGLCBlock` 已完成最小运行验证：
   - `mask=None` 直接返回输入
   - `boundary_width=0` 正常运行
   - 非法 `backend` 正常报错
   - 非法 `boundary_width` 正常报错

当前未完成的验证：

- 完整 wrapper 前向联调

原因：

- 本地环境缺少仓库原有依赖 `einops`
- 该依赖由 `models/modules/module_util.py` 间接要求
- 所以当前无法在本机完成完整 active generator 前向验证

这属于环境缺失，不是本次 MGLC 代码本身的接口问题。


## 9. 当前 MVP 的边界

本次实现明确不包含：

- decoder 第二插点
- `sem_lite` backend
- `restore_S_guidance`
- local-only / context-only 可配置开关
- boundary auxiliary loss
- test / inference tree 同步

这些内容保留到后续增强版本。


## 10. 结论

这次改动把第三创新点严格收敛到了一个 mid-only、低侵入、可配置关闭的 bottleneck 模块：

- 不与 BrushNet 功能重叠
- 不破坏现有训练主链
- 能兼容旧 checkpoint
- 具备清晰的后续增强空间

这使它适合作为当前仓库中的第三创新点 MVP 基线实现。
