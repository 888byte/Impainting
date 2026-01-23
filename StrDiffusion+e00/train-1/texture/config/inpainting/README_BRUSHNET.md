# 壁画修复系统 - BrushNet集成指南

## 📋 概述

本项目实现了一个双阶段壁画图像修复与颜色复原系统：

- **第一阶段**：利用预训练LUT生成颜色先验图和置信度图
- **第二阶段**：改进的BrushNet架构指导StrDiffusion完成修复

## 📁 新增文件结构

```
StrDiffusion+e00/train-2/texture/config/inpainting/
├── lut_processor.py              # LUT三线性插值处理器
├── color_prior_generator.py      # 颜色先验与置信度生成器
├── pigment_lut33.npz             # [需用户提供] LUT文件
├── debug_logs/                   # Debug输出目录
├── data/
│   └── mural_inpainting_dataset.py  # 壁画修复数据集
├── models/
│   ├── pixel_brushnet.py         # 像素空间BrushNet
│   ├── zero_conv.py              # Zero-Convolution
│   └── brushnet_wrapper.py       # UNet与BrushNet集成包装器
├── options/train/
│   └── ir-sde-brushnet.yml       # BrushNet训练配置
└── README_BRUSHNET.md            # 本文档
```

## 🔧 配置说明

### 1. LUT文件配置

将 `pigment_lut33.npz` 放置到项目目录，并在配置文件中设置路径：

```yaml
lut:
  path: ./pigment_lut33.npz
  alpha: 0.7  # LUT置信度权重
  beta: 0.3   # 修复置信度权重
```

### 2. GT生成模式

支持三种模式，在配置文件中设置：

```yaml
datasets:
  train:
    gt_mode: mixed  # 可选: full, partial, mixed
```

| 模式 | 说明 |
|------|------|
| `full` | 全图LUT映射，模型学习恢复整张图颜色 |
| `partial` | 仅Mask区域LUT映射，保留背景纹理 |
| `mixed` | 随机选择（各50%），增强泛化能力 |

### 3. BrushNet配置

```yaml
brushnet:
  enabled: true   # 启用/禁用BrushNet
  in_nc: 8        # 输入通道: Noisy(3) + Mask(1) + Prior(3) + Conf(1)
  nf: 64          # 与主UNet对齐
  depth: 4        # 与主UNet对齐
  lite: false     # 轻量版（资源受限时使用）
```

## 🚀 训练启动

### 使用新配置训练

```bash
cd d:\code\ky\bihua\Impainting\StrDiffusion+e00\train-2\texture\config\inpainting
python train.py -opt options/train/ir-sde-brushnet.yml
```

### 开启Debug模式

修改配置文件：

```yaml
debug:
  enabled: true
  log_dir: debug_logs/
  save_freq: 500
```

Debug模式将保存以下张量可视化：
- `input_image.png` - 输入褪色图像
- `transformed_gt.png` - 变换后的GT
- `generated_prior.png` - 生成的颜色先验
- `confidence_map.png` - 置信度图
- `masked_input.png` - 掩码后的输入

## 🧪 单元测试

### 测试LUT处理器

```bash
python lut_processor.py
```

### 测试颜色先验生成器

```bash
python color_prior_generator.py
```

### 测试PixelBrushNet

```bash
python models/pixel_brushnet.py
```

### 测试集成包装器

```bash
python models/brushnet_wrapper.py
```

## 📐 架构说明

### PixelBrushNet输入

| 通道 | 名称 | 范围 |
|------|------|------|
| 0-2 | Noisy_Image | [0, 1] |
| 3 | Mask | {0, 1} |
| 4-6 | Color_Prior | [0, 1] |
| 7 | Confidence | [0, 1] |

### 特征注入机制

```
BrushNet Encoder        Main UNet
    │                      │
    ▼                      ▼
 Layer 1 ──Zero-Conv──► + Layer 1
    │                      │
    ▼                      ▼
 Layer 2 ──Zero-Conv──► + Layer 2
    │                      │
   ...                    ...
    │                      │
    ▼                      ▼
   Mid   ──Zero-Conv──► + Mid
```

## ⚠️ 注意事项

1. **不影响原始代码**：所有新增文件独立于原有代码，原始 `train.py` 和 `ir-sde.yml` 保持不变

2. **LUT文件**：必须提供 `pigment_lut33.npz`，包含以下键：
   - `grid`: 网格坐标
   - `lut_rgb`: RGB映射表
   - `lut_conf`: 置信度映射表

3. **显存需求**：BrushNet增加约30%显存占用，可使用 `lite: true` 减少开销

4. **数据对齐**：确保训练数据的颜色先验与GT使用相同LUT生成

## 📊 性能参考

| 配置 | 显存 (256x256) | 训练速度 |
|------|----------------|----------|
| 原始UNet | ~6GB | 1x |
| UNet + BrushNet | ~8GB | 0.8x |
| UNet + BrushNetLite | ~7GB | 0.9x |

## 🔍 问题排查

### LUT文件未找到

```
FileNotFoundError: LUT文件不存在
```

**解决**：确认 `pigment_lut33.npz` 路径正确

### 颜色断层

如果颜色先验出现明显断层，检查：
1. LUT网格密度是否足够（建议 >= 33）
2. 三线性插值是否正确启用

### 显存不足

尝试：
1. 减小 `batch_size`
2. 启用 `brushnet.lite: true`
3. 减小 `GT_size`
