# MVP: 用 Mamba/SSM 逆推颜料褪色前初始颜色 (Lab0)

> 输入：颜色时间序列 Lab(t) + 拉曼谱库稀疏系数（强先验注入）
> 输出：初始颜色 Lab(t0) 的分布参数（对角高斯：mu, sigma）

## 0) 准备
把你给我的原始文件已经复制到 `data/raw/` 目录（本工程包内自带）。

创建环境并安装依赖：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 1) 一键生成处理后的数据
```bash
python scripts/prepare_data.py --config config.yaml
```
输出：
- `data/processed/raman_dict.npz`  (拉曼谱库字典：D_532, D_785, 及网格与名字)
- `data/processed/dataset_sequences.npz` (18条序列：2个环境 * 9个体系)

## 2) 训练 MVP（Mamba 主干）
```bash
python scripts/train_mvp.py --config config.yaml
```
输出：
- `runs/mvp/best.pt`  最佳模型（按验证集 NLL）
- `runs/mvp/metrics.json`  训练/验证指标

## 3) 在“未见体系”上评估（leave-one-system-out）
```bash
python scripts/eval_mvp.py --config config.yaml --ckpt runs/mvp/best.pt
```
输出：
- ΔE76, Lab RMSE, NLL, 以及 90% 置信区间覆盖率

## 4) 推理：只用后期若干点逆推 Lab0
例：使用 env=76, NO=3，取最后 10 个时间点做反推
```bash
python scripts/predict_mvp.py --config config.yaml --ckpt runs/mvp/best.pt --env 76 --no 3 --use_last_n 10
```

---

## 参考
- Mamba 官方实现与论文入口： https://github.com/state-spaces/mamba
- scikit-image `rgb2lab`: https://scikit-image.org/docs/stable/api/skimage.color.html
-（二阶段）可微稀疏编码可参考深度展开：ISTA-Net++ (PyTorch) https://github.com/jianzhangcs/ISTA-Netpp
