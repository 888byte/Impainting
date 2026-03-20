# LUT 生成脚本修复说明

## 问题描述

### 现象
- `pigment_lut49_xhq.npz` 文件覆盖率只有 0.6%（715/117,649 个点）
- LUT 点只分布在 R=0 的超平面上
- 每次重新运行时，LUT 都从零开始，导致无法断点续生成

### 根本原因

在 `build_pigment_lut33.py` 的 `save_all` 函数中，存在以下问题：

1. **done 数组没有保存到 npz 文件中**
   - 原代码只保存到 `state['done_npy']`（独立的 .npy 文件）
   - npz 文件中不包含 `done` 键
   - 恢复时 `load_resume_arrays` 无法从 npz 加载 done 状态
   - 导致每次都从零开始

2. **恢复逻辑不完整**
   - `load_resume_arrays` 函数中，只有 `_done.npy` 存在时才加载 done
   - 但 npz 文件没有 done，导致无法正确恢复

3. **生成中断后无法继续**
   - 如果生成过程被中断（用户停止、崩溃、超时）
   - 已完成的进度会丢失
   - 下次运行时从头开始

### 修复方案

#### 修改内容

在 `save_all` 函数中添加 `done` 数组到 npz 文件：

```python
def save_all(args: argparse.Namespace, grid: np.ndarray, state: dict) -> None:
    np.save(state['done_npy'], state['done'])
    
    # 修复：将 done 数组也保存到 npz 文件中
    _atomic_save_npz(
        state['out_npz'],
        grid=grid,
        lut_rgb=state['lut_rgb'],
        lut_lab=state['lut_lab'],
        lut_conf=state['lut_conf'],
        lut_std=state['lut_std'],
        lut_cdiff=state['lut_cdiff'],
        lut_cret=state['lut_cret'],
        done=state['done'],  # 关键修复：添加 done 数组
        meta=dict(...)
    )
```

#### 修复说明

1. **`done` 数组的作用**
   - 记录哪些网格点已经完成推理
   - 恢复时可以跳过已完成的点
   - 支持断点续生成

2. **npz 文件的作用**
   - 存储所有 LUT 数据和元数据
   - 应该包含完成状态，以便正确恢复

3. **保存策略**
   - `_done.npy`: 快速标记文件，用于增量保存
   - `.npz`: 完整的数据文件，包含 done 标记

## 使用说明

### 重新生成完整的 LUT

修复后，可以重新生成完整的 LUT：

```bash
cd /home/610-wws/Impainting/SSD-TS/pigment_task
python build_pigment_lut33.py \
  --ckpt ckpt/lab_raman_xrd/best_model.pt \
  --library_npz data/standard_alignment/library_embeddings.npz \
  --prototype_bank data/pigment_npz/prototype_bank.npz \
  --cond_method auto \
  --num_samples 14 \
  --grid_size 49 \
  --device cuda \
  --out_npz pigment_lut49_xhq.npz \
  --engine batch \
  --batch_size 2048 \
  --save_every 300
```

### 验证 LUT 完整性

```bash
python -c "
import numpy as np
data = np.load('pigment_lut49_xhq.npz', allow_pickle=True)
if 'done' in data:
    done = data['done']
    print('Done array found!')
    print(f'Total entries: {done.size}')
    print(f'Completed: {np.sum(done)}')
    print(f'Coverage: {np.sum(done)/done.size*100:.1f}%')
else:
    print('Done array NOT found in npz!')
"
```

## LUT 数据结构

### 必需字段
- `grid`: [N] float32 - 网格坐标点（0-255 等间隔划分）
- `lut_rgb`: [N, N, N, 3] uint8 - RGB 颜色映射表
- `lut_conf`: [N, N, N] float32 - 置信度映射表（0-1）

### 可选字段
- `lut_lab`: [N, N, N, 3] float32 - Lab 颜色空间映射
- `lut_std`: [N, N, N] float32 - 预测标准差
- `lut_cdiff`: [N, N, N] float32 - 差分置信度
- `lut_cret`: [N, N, N] float32 - 检索置信度
- `done`: [N, N, N] uint8 - 完成标记（1=已完成，0=未完成）
- `meta`: dict - 元数据（配置信息）

### 使用方说明

#### lut_processor.py

LUT 处理器需要以下字段：

```python
required_keys = ['grid', 'lut_rgb', 'lut_conf']
optional_keys = ['lut_lab', 'lut_std', 'lut_cdiff', 'lut_cret', 'done', 'meta']
```

#### color_prior_generator.py

颜色先验生成器使用：

```python
# 必需
self.lut.trilinear_interpolate(image)  # 返回 (color_prior, confidence)

# 可选
self.lut.lut_lab  # 如果存在，可以用于 Lab 空间处理
```

## 注意事项

1. **断点续传**
   - 确保生成过程可以中断和恢复
   - 定期保存进度（通过 `--save_every` 参数控制）

2. **完整性验证**
   - 生成完成后检查覆盖率应该接近 100%
   - 覆盖率 < 95% 可能表示生成未完成

3. **存储效率**
   - npz 文件是压缩的，适合存储完整 LUT
   - _done.npy 是未压缩的，用于快速增量更新

## 常见问题

### Q: 为什么 LUT 覆盖率只有 0.6%？
A: 因为 `done` 数组没有保存到 npz 文件中，导致恢复时从零开始，且生成过程可能被中断。

### Q: 如何验证 LUT 是否完整？
A: 检查 done 数组的完成比例，应该在 95-100% 范围内：
```python
data = np.load('pigment_lut49_xhq.npz', allow_pickle=True)
done = data['done']
print(f'Coverage: {np.sum(done)/done.size*100:.1f}%')
```

### Q: 可以继续未完成的 LUT 吗？
A: 可以，直接运行相同的命令，脚本会从 npz 中读取 done 状态并继续生成。
