# Legacy Compatibility

这个目录保留给历史脚本、一次性实验文件和未来需要降级归档的兼容资产。

当前兼容关系如下：

- `python -m pigment_task.preprocess_pigment` -> `preprocess.py`
- `python -m pigment_task.train_pigment` -> `train.py`
- `python -m pigment_task.infer_pigment` -> `infer.py`
- `python -m pigment_task.active_color_eval` -> `evaluate.py`
- 根目录 `t1.py` 到 `t7.py` 属于历史实验/演示脚本，已改为依赖当前顶层模块与 bundle 结构，但不作为主入口维护。

活跃实现已经迁移到根目录自然命名入口与顶层模块：`data/`、`models/`、`bridge/`、`training/`、`inference/`、`evaluation/`。
