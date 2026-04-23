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

## 2026-04-22 21:20 日志复盘：no-extra 仍发白

已查看两份日志：

- `C:\Users\admin\Desktop\test_ir-sde-no-extra-current-domain_260422-204651.log`
- `C:\Users\admin\Desktop\test_ir-sde-no-extra-original-semantics_260422-205916.log`

关键事实：

- `no_extra_route=True`，说明 BrushNet/MGLC/Mu-Denoiser 已经真正关闭。
- 当前权重 `32000_G.pth` 主干加载完整：`loaded 231/231`，新增分支为 `unexpected=147`，符合关闭新增模块后的预期。
- 但两次运行仍是 `discriminator_guidance=True` 且 `deterministic_reverse=False`。
- 因此这两次还不是“纯 no-extra sampler”诊断；仍混入判别器候选选择和随机反推噪声。
- `condition_lut` 版本中，`cond_hole` 和 `prior_hole` 都不是白色，但 `raw_hole/final_hole` 白色比例很高，说明发白发生在 reverse sampler/score 轨迹中，而不是输入先验直接是白色。

已更新两个 no-extra YAML：

- `inference.deterministic_reverse: true`
- `inference.discriminator_guidance.enabled: false`

下一步先重跑 `ir-sde-no-extra-current-domain.yml`，日志应出现：

- `no_extra_route=True pure_no_extra_route=True`
- `discriminator_guidance=False`
- `deterministic_reverse=True`

如果 pure no-extra 仍白，再回查主干 score / S guidance / checkpoint 主干训练分布；如果 pure no-extra 不白，问题优先归因于判别器引导或随机 reverse noise。

## 2026-04-22 21:50 继续复盘：纯 no-extra 仍发白后的下一步

如果新 run 已确认：

- `no_extra_route=True`
- `pure_no_extra_route=True`
- `discriminator_guidance=False`
- `deterministic_reverse=True`

但 mask 区仍然发白，则可以排除 BrushNet / MGLC / Mu-Denoiser / 判别器 / 随机 reverse noise 是主因。

下一步优先隔离 `restore_S_guidance` / SPADE 结构路径：新增两个诊断配置：

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-no-structure-current-domain.yml`
- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-no-structure-known-only.yml`

它们保持：

- BrushNet=false
- MGLC=false
- Mu-Denoiser=false
- D guidance=false
- deterministic_reverse=true
- 但 `restore_S_guidance=false`

同时在 `utils/sde_utils.py` 增加 `[Trajectory Debug]`，会在若干关键步记录 hole 的 mean/min/max/white 和 score_abs_mean，用来确认是哪个阶段开始推白。

判断：

1. 关掉 `restore_S_guidance` 后明显不白：优先修结构 S/edge/SPADE 路径。
2. 关掉 `restore_S_guidance` 后仍白：优先查当前 32000_G 主干 score 是否已被训练分布带偏，或者当前推理 condition/x0 与训练域仍不一致。

## 2026-04-22 23:00 去噪起点修正

用户指出推理去噪起点不对：原版 StrDiffusion inpainting 的 clean start/mu 应该是 `known_pixels * mask_known`，hole 区在加噪前是黑色；`noise_state()` 之后 hole 才会有小幅 Gaussian noise。

日志确认之前 `no-structure-current-domain` 使用了：

- `sde_mu_hole_mode=condition_lut`
- `cond_hole(mean≈0.69~0.81)`

这说明 hole 被 LUT 内容预填了，确实不是原版黑洞起点，也会导致去噪过程看起来只是轻微调整。

已修改：

- `ir-sde-no-extra-current-domain.yml` 改为 `sde_mu_hole_mode: known_only`
- `ir-sde-no-extra-no-structure-current-domain.yml` 改为 `sde_mu_hole_mode: known_only`
- 推理日志新增 `x_init_hole(...)` 和 `noisy_start_hole(...)`
- 中间图新增 `x_start_noisy.png`

下一次应优先跑带结构的原版路径：

`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-current-domain.yml`

期望日志：

- `sde_mu_hole_mode=known_only`
- `x_init_hole(mean≈0,min≈0,max≈0,white=0)`
- `noisy_start_hole` 为接近 0 的小噪声，而不是 LUT 灰/白块

如果这个起点正确后仍没有有效去噪，再查主干 score/训练分布。

## 2026-04-22 23:35 黑洞起点正确，但纯 mean sampler 无纹理

新日志 `test_ir-sde-no-extra-current-domain_260422-232202.log` 说明：

- `x_init_hole(mean=0,min=0,max=0)`，原版黑洞起点已经正确。
- `noisy_start_hole` 只有小噪声，起点问题已修正。
- 在纯诊断路径中：`deterministic_reverse=True` 且 `discriminator_guidance=False`，hole 从黑色逐渐变成浅色/白色均值块，但没有纹理。

这说明当前问题不再是起点错误，而是：关闭所有辅助分支并使用 deterministic mean sampler 时，当前主干只给出均值/颜色趋势，不能生成纹理细节。

新增两个下一步配置：

1. `ir-sde-no-extra-original-sampler-known-only.yml`
   - BrushNet/MGLC/Mu-Denoiser 仍关闭
   - `known_only` 黑洞起点
   - 恢复原版 sampler 风格：`deterministic_reverse=false` + `discriminator_guidance=true`
   - 用来确认原版随机+D sampler 是否能给当前主干带回纹理。

2. `ir-sde-brushnet-only-known-start.yml`
   - 黑洞起点 + 原版结构引导
   - 只打开 BrushNet，关闭 MGLC/Mu-Denoiser/D guidance
   - 用来验证“只注入颜色先验图，不改主干结构”是否能提供纹理/颜色参考。

判断：

- 如果 original-sampler 有纹理：之前纯 deterministic 诊断过于保守，最终路径需要保留原版 sampler。
- 如果 original-sampler 仍无纹理，但 brushnet-only 有纹理：说明当前主干单独不够，BrushNet prior 是必要条件。
- 如果两者都无纹理：优先查 BrushNet 输入/特征注入强度，或当前 32000_G 主干训练分布已经偏成均值填充。

## 2026-04-23 00:25 original-sampler / BrushNet-only 仍失败后的结论

新日志：

- `test_ir-sde-no-extra-original-sampler-known-only_260422-235949.log`
  - no-extra 路由成立：BrushNet=false, MGLC=false, Mu-Denoiser=false。
  - 黑洞起点成立：`x_init_hole(mean=0,min=0,max=0)`。
  - 恢复随机 reverse + D guidance 后，hole 仍主要变成浅灰/白块，没有纹理。
- `test_ir-sde-brushnet-only-known-start_260423-000710.log`
  - BrushNet 权重确实加载并参与：`loaded 326/326`，BrushNet runtime=true。
  - 起点仍正确，但输出只出现黑/深色块，不是有效纹理修复。

因此已排除：

1. “只是起点不是黑洞” —— 已修正，仍失败。
2. “只是 deterministic mean sampler 太保守” —— 恢复随机+D 后仍失败。
3. “BrushNet 没加载/没生效” —— BrushNet-only 明显改变轨迹，但方向错误。
4. “关掉新增模块就能恢复原版能力” —— 当前 32000_G 的 no-extra 主干路径仍不能恢复纹理。

下一步不再继续盲调 x7 推理参数，而是做 **原版 StrDiffusion sampler parity**：当前 enhanced 路径虽然模拟了原版采样，但仍不是逐行原版 `reverse_sde` 分支。已新增两个配置，用同一个 wrapper/checkpoint 做无重训对照：

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-legacy-reverse-current-domain.yml`
  - no BrushNet/MGLC/Mu
  - `force_legacy_reverse=true`
  - `condition_known_source=lut`, `structure_source=lut`
  - 目的：验证是否是 enhanced reverse 分支本身偏离原版。

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-legacy-reverse-gt-parity.yml`
  - no BrushNet/MGLC/Mu
  - `force_legacy_reverse=true`
  - `condition_known_source=gt_if_available`, `structure_source=gt_if_available`
  - 目的：在训练集/有 GT 的样本上尽量贴近原版 StrDiffusion 测试语义，判断当前 x7 主干是否还保留原版修复能力。

新增日志应出现：

- `force_legacy_reverse=True`
- `[LegacyReverseRoute] ... enhanced_inference=false`
- `[StructureRoute] ... resolved=lut` 或 `resolved=gt`

判断：

- current-domain legacy 能改善：之前问题集中在 enhanced reverse/composite 逻辑。
- current-domain 仍差但 gt-parity 改善：问题集中在当前推理 condition/structure 构造和原版训练/测试语义不一致。
- 两者仍差：当前 x7 checkpoint 的主干 score 已经被后续训练/新增模块带偏；无重训关闭模块无法恢复原版能力，需要回到原版收敛 checkpoint 或做主干冻结/小学习率恢复训练。

## 2026-04-23 10:50 legacy reverse + GT parity 仍失败

新日志：

- `test_ir-sde-no-extra-legacy-reverse-current-domain_260423-103607.log`
- `test_ir-sde-no-extra-legacy-reverse-gt-parity_260423-104051.log`

确认信息：

- `force_legacy_reverse=True`，确实走了原版 `reverse_sde` 分支（`enhanced_inference=false`）。
- no-extra 路由成立：BrushNet=false, MGLC=false, Mu-Denoiser=false。
- `loaded 231/231 tensors into ConditionalUNetWithBrushNet`，当前 x7 checkpoint 的原版主干权重全部加载。
- 起点仍正确：`x_init_hole(mean=0,min=0,max=0)`。
- GT parity 中结构来源确实为 GT：`[StructureRoute] ... resolved=gt has_gt=True`。

结论：

即使使用原版 sampler 分支，并且在训练集/有 GT 情况下把 condition/structure 尽量贴近原版 StrDiffusion，当前 x7 的 `32000_G.pth` 主干仍不能修复。此时问题基本不在推理分支、起点、D guidance、BrushNet/MGLC/Mu 开关，而是当前 checkpoint 的主干 score 已经偏离原版可修复解。

下一步验证不再继续用 x7 盲调，而是做两件事：

1. **原版 baseline checkpoint parity**
   - 新增配置：
     `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-baseline-original-checkpoint-gt-parity.yml`
   - 默认加载：
     `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
   - 若它能正常修复，则说明当前推理代码已经足够接近原版，问题集中在 x7 checkpoint 主干被训练带偏。
   - 若它也不能修复，则还需要继续对齐当前 `texture-1` 测试树与原版 `StrDiffusion/test/texture` 的数据/结构生成。 

2. **checkpoint 主干漂移审计**
   - 新增脚本：
     `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\tools\checkpoint_trunk_audit.py`
   - 比较原版 baseline 和 x7 之间共享的 ConditionalUNet 主干权重漂移。
   - 输出：
     `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\STRDIFFUSION_CHECKPOINT_DRIFT.md`

如果 baseline parity 正常且 drift 很大，后续无重训可选方案只有：

- 直接使用原版 baseline checkpoint 推理；或
- 做 checkpoint surgery：把 x7 checkpoint 中原版主干权重替换回 baseline，只保留新增模块权重，再做 no-extra/BrushNet-only 推理消融。

## 2026-04-23 12:10 baseline checkpoint parity 有效后的下一步

用户反馈 `ir-sde-baseline-original-checkpoint-gt-parity`：

- 原版 baseline checkpoint 已经能修，边缘/统一颜色区域明显恢复。
- 仍存在局部黑/深色斑块，复杂纹理区域细节不足。

结合 `STRDIFFUSION_CHECKPOINT_DRIFT.md`：

- baseline vs x7 共享主干 `231` 个 tensor。
- x7 多出 `189` 个新增模块 tensor。
- 主干 global `relative_rms≈0.060467`，漂移最大在 `mid_block*`、`ups.*` 和早期 `downs.0`。

结论：

1. 当前 `texture-1` 推理链路已经足够接近原版，原版 baseline checkpoint 可以恢复基本修复能力。
2. x7 关闭新增模块仍失败，说明 x7 的原版主干被后续训练带偏，而不是推理路径本身坏。
3. baseline 的黑斑更多像原版随机+D adaptive sampler 的候选选择伪影，尤其在参考先验/GT/LUT 的 hole 区没有暗色内容时，D 仍可能选到局部过暗 proposal。

已新增/修改：

- 增强版 D guard：`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\utils\sde_utils.py`
  - 增加 dark-ratio 检查。
  - enhanced guarded sampler 使用 color_prior/GT/LUT 作为安全参考，而不是黑洞 `mu`。
  - 日志新增：`[DiscriminatorGuard] rejected_candidates=...`。

- 新增 baseline sampler 对照配置：
  - `ir-sde-baseline-original-checkpoint-guarded-sampler-gt-parity.yml`
    - baseline G，enhanced guarded stochastic + D。
    - 目标：保留纹理同时抑制黑斑。
  - `ir-sde-baseline-original-checkpoint-deterministic-gt-parity.yml`
    - baseline G，deterministic + no D。
    - 目标：验证黑斑是否由 stochastic/D 造成；预期更平滑、纹理更少。

- 新增 checkpoint surgery 脚本：
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\tools\make_baseline_trunk_hybrid.py`
  - 生成：baseline trunk + x7 added modules。
  - 默认输出：
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/32000_G.baseline_trunk_x7_extra.pth`

- 新增 hybrid BrushNet-only 对照配置：
  - `ir-sde-hybrid-baseline-trunk-brushnet-only-guarded-gt-parity.yml`
  - `ir-sde-hybrid-baseline-trunk-brushnet-only-deterministic-gt-parity.yml`

下一步顺序：

1. 先跑 baseline guarded sampler，看黑斑是否消失同时保留纹理。
2. 再跑 baseline deterministic，确认黑斑是否由 D/stochastic 引入。
3. 如果 baseline guarded 较好，再生成 hybrid checkpoint 并跑 hybrid BrushNet-only，验证 x7 的 BrushNet 能否在 baseline 主干上提供颜色/纹理先验。

## 2026-04-23 14:35 switch to 48000 checkpoint

Result summary for the 32000 comparison set:

- `baseline-original-checkpoint-guarded-sampler-gt-parity`
  - best current no-retrain result; edge repair is the best and hole region keeps some texture.
- `baseline-original-checkpoint-deterministic-gt-parity`
  - stable but smoother; fewer artifacts and fewer details.
- `hybrid-baseline-trunk-brushnet-only-guarded-gt-parity`
  - full-scale BrushNet (`feature_scale=0.3`) injects unrelated content.
- `hybrid-baseline-trunk-brushnet-only-deterministic-gt-parity`
  - BrushNet still causes split / dark shadow even without stochastic sampler.

Decision:

1. Current best no-retrain route is still **baseline original checkpoint + guarded sampler**.
2. The x7 `32000_G.pth` BrushNet extra branch is not safe to use directly.
3. Move to user-requested `48000_G.pth` and test in this order:
   - current x7 trunk only (no extra modules), guarded sampler;
   - current x7 trunk only (no extra modules), deterministic no-D;
   - baseline trunk + x7 48000 extra modules, weak BrushNet only (`feature_scale=0.03` then `0.01`).

New configs:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-gt-parity.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-deterministic-gt-parity.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak003-guarded-gt-parity.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-gt-parity.yml`

Scripts updated for 48000 defaults:

- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/tools/make_baseline_trunk_hybrid.py`
  - default current/out -> `48000_G.pth` / `48000_G.baseline_trunk_x7_extra.pth`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/tools/checkpoint_trunk_audit.py`
  - default current -> `48000_G.pth`
  - default markdown out -> `STRDIFFUSION_CHECKPOINT_DRIFT_48000.md`

Extra inference log fields added:

- `brushnet.feature_scale(runtime)=...`
- `brushnet.use_spatial_gate(runtime)=...`

These two fields are required to verify that the weak BrushNet configs are actually active at runtime.


## 2026-04-23 16:45 48000 results: likely train/inference mu-hole mismatch

New user feedback for 48000:

- `ir-sde-no-extra-x7-48000-guarded-gt-parity` is usable but still not ideal.
- `ir-sde-no-extra-x7-48000-deterministic-gt-parity` is smoother/gray, as expected for deterministic no-D.
- `hybrid-baseline-trunk-x7-48000-brushnet-weak003` is gray-ish and does not clearly improve.
- `hybrid-baseline-trunk-x7-48000-brushnet-weak001` is close to the no-extra guarded route; BrushNet at 0.01 is almost neutral.

Important evidence:

- The uploaded drift report is still for `32000_G.pth`, not 48000:
  - report current path: `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/32000_G.pth`
  - so do not use that report as 48000 evidence yet.
- The active training config `options/train/ir-sde-brushnet-ft.yml` contains:
  - `train.sde_mu_hole_mode: condition_lut`
- But the latest 48000 test configs used:
  - `inference.sde_mu_hole_mode: known_only`

This is now the clearest concrete mismatch: the 48000 checkpoint was trained with non-black, target-domain hole mu (`condition_lut`), but the current no-extra/weak-BrushNet inference was run from a black-hole mu (`known_only`). That is a real distribution shift. It can explain why:

- deterministic reverse becomes gray/smooth;
- BrushNet at weak scale does not add useful detail;
- the guarded stochastic sampler partly recovers but remains slightly off from GT.

Code changes added for verification:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - logs `expected_train_sde_mu_hole_mode=...`
  - warns when train/inference mu-hole modes differ
  - logs `[Target Debug] final_gt_l1/raw_gt_l1/prior_gt_l1/lut_gt_l1/final_prior_l1/final_lut_l1` for GT-parity runs

New configs for the next direct test:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-train-mu-gt-parity.yml`
  - no extra modules, 48000 checkpoint, guarded sampler
  - `sde_mu_hole_mode: condition_lut`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-train-mu-gt-parity.yml`
  - baseline trunk + 48000 extra, weak BrushNet 0.01
  - `sde_mu_hole_mode: condition_lut`

Expected decision:

- If train-mu config improves color/texture and reduces `final_gt_l1`, the confirmed problem is train/inference `sde_mu_hole_mode` mismatch.
- If it worsens or becomes white/over-smooth, then x7 training itself over-relied on `condition_lut` in the hole and did not learn real inpainting from blank-hole starts; future training should use `known_only` or re-enable a gradient-carrying inference-like x0 loss with microbatch/checkpointing.


## 2026-04-23 17:50 train-mu parity failed: condition_lut hole anchor is not the fix

New evidence:

- `STRDIFFUSION_CHECKPOINT_DRIFT_48000.md`
  - 48000 trunk drift relative_rms = `0.062552`.
  - This is larger than the previous 32000 report (`0.060467`), so continued x7 training did not move the original trunk back toward the known-good baseline; it drifted slightly more.
- `ir-sde-no-extra-x7-48000-guarded-train-mu-gt-parity`
  - `sde_mu_hole_mode=condition_lut` removed the train/inference mode mismatch, but the result became white / plain.
  - Example `000098_bottom`: x_init hole mean `0.7871`, final hole mean `0.9425`.
  - `final_gt_l1=0.080698`, while `prior_gt_l1=0.075773`; the generated result is not better than the color prior.
- `ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-train-mu-gt-parity`
  - Output is almost exactly the LUT anchor, not a repaired texture.
  - Example `000098_bottom`: `final_lut_l1=0.000458`, `final_gt_l1=0.112950`, `prior_gt_l1=0.075773`.

Conclusion:

The mismatch test is decisive: matching the training `condition_lut` hole mu does **not** restore repair ability. Instead it reveals the current x7 training problem more clearly:

1. The x7 training objective allowed/encouraged a non-black hole mu (`condition_lut`) that already contains a smooth bright fill.
2. The normal one-step SDE loss is teacher-forced from states generated around that filled mu, so the model can minimize loss without learning to synthesize missing texture from a blank/noisy hole.
3. The inference-like blank-hole x0 branch in `models/denoising_model.py` is monitor-only (`with torch.no_grad()` and `infer_x0_weighted=None`), so it does not correct this failure.
4. BrushNet weak injection at 0.01 is almost neutral; at larger scales it injects artifacts. It is not currently a usable texture restoration branch.

Current no-retrain best route remains:

- use the known-good original baseline checkpoint with guarded stochastic sampler; or
- use x7 48000 known-only guarded as an ablation, but it is not structurally better than the baseline route.

Next no-retrain diagnostic added:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-gt-parity.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-safe-prior-gt-parity.yml`

These set `sde_mu_hole_mode=safe_prior`, using the confidence-gated color prior as the hole mu. This is not claiming to match training; it checks whether a better hole anchor than `condition_lut` can give an acceptable no-retrain inference fallback. Logs now include `[MuAnchor Debug]` to detect anchor pass-through.

