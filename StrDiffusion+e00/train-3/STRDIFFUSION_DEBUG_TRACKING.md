# StrDiffusion 排错跟踪：no-retrain 原版路径消融

更新时间：2026-04-22
工作区：`D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3`

## 当前结论

- 这轮不重新训练，只做推理期消融：保持 `ConditionalUNetWithBrushNet` wrapper 以兼容当前 checkpoint key，但关闭新增模块。
- 保留 `restore_S_guidance=true`，因为它是原版 StrDiffusion 的结构引导路径，不计入 BrushNet/MGLC/Mu-Denoiser 新增分支。
- `infer_x0` 属于训练侧辅助 loss/监控；本轮推理不使用。训练配置已把 `train.infer_x0_loss_weight` 置为 `0.0`，并固定 `infer_x0_grad=false`。
- 后续只维护主推理树：`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting`；不再继续处理 `_texture1_patch`。

## 已排除/不再优先追的方向

- 不再把问题单独归因于 infer_x0 反传：`infer_x0_weighted` 反传会明显增加显存，当前先关闭。
- 不再继续盲目加强 MGLC 或扩大 BrushNet 注入：先证明去掉新增分支后主干是否能恢复。
- 不再仅看最终图判断；必须同时看日志中的 route、checkpoint、condition/mu、hole 区统计和 `state_*` 轨迹。
- 不直接切回原始 `ConditionalUNet` 类；这样可能导致当前增强 checkpoint 的 key 大量不匹配。当前用 wrapper 但 bypass 新增模块。

## 本轮新增 no-retrain 配置

主推理入口：
`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\test.py`

可直接使用的配置：

```powershell
cd D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting

# 推荐先跑：关闭 BrushNet/MGLC/Mu-Denoiser，但保持当前 x7 训练域 condition_lut
python test.py -opt options/test/ir-sde-no-extra-current-domain.yml

# 对照：更接近原版 hole 侧 known_only 语义，但可能和当前 x7 训练域不一致
python test.py -opt options/test/ir-sde-no-extra-original-semantics.yml
```

如果 YAML 里的 `path.pretrain_model_G` 不是你要测的当前 checkpoint，用命令行覆盖，不需要改文件：

```powershell
python test.py -opt options/test/ir-sde-no-extra-current-domain.yml `
  --set path.pretrain_model_G=/path/to/current_G.pth
```

## 每次运行必须检查的日志

查找这些日志标签：

- `[RouteCheck] network_G=ConditionalUNetWithBrushNet model_class=ConditionalUNetWithBrushNet no_extra_route=True`
- `[RouteCheck] brushnet.enabled(config/runtime)=False/False`
- `[RouteCheck] texture_core.enabled(config/runtime)=False/False`
- `[RouteCheck] mu_denoiser.enabled(config/available/runtime/has_weights)=False/.../False/False`
- `[RouteCheck] restore_S_guidance=True`
- `[RouteCheck] sde_mu_hole_mode=condition_lut` 或 `known_only`
- `[RouteCheck] pretrain_model_G=...`，确认不是误用了旧/随机 checkpoint
- `[LoadCheck] loaded ... missing=... unexpected=...`，确认 checkpoint 真实加载情况
- `[NoExtraRoute] BrushNet/MGLC/Mu-Denoiser are bypassed...`
- `[Inference Debug] ... cond_hole ... raw_hole ... final_hole ...`，观察 hole 区是否快速灰/黑/白塌缩

## 判断分支

1. 如果 `no_extra_current_domain` 明显改善：优先回查 BrushNet prior、MGLC 注入、Mu-Denoiser 的任一分支是否破坏主干。
2. 如果 `no_extra_current_domain` 仍差，但 `known_only` 改善：优先回查 `condition_lut` hole anchor 与当前 checkpoint 训练目标是否不一致。
3. 如果两个 no-extra 都差：优先回查原版不变量是否仍被破坏：`training_target/GT/x0`、`condition_mu`、`sde.set_mu()`、`reverse_optimum_step` target 必须同域一致。
4. 如果日志里 `no_extra_route=False`：先不要看图，说明配置没有真正关闭新增模块。

## 下次继续排查的最短路径

- 先贴对应 run 的 `test_*.log` 中 `[RouteCheck]`、`[LoadCheck]`、`[Inference Debug]` 几行。
- 同时贴同一样本的 `x_init.png`、`condition_mu.png`、`state_1/state_10/state_25/state_50/state_100/final.png`。
- 只在 no-extra 路径复现后，再决定是否逐个打开 BrushNet、MGLC、Mu-Denoiser 做二分定位。
