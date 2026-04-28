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
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
4. BrushNet weak injection at 0.01 is almost neutral; at larger scales it injects artifacts. It is not currently a usable texture restoration branch.

Current no-retrain best route remains:

- use the known-good original baseline checkpoint with guarded stochastic sampler; or
- use x7 48000 known-only guarded as an ablation, but it is not structurally better than the baseline route.

Next no-retrain diagnostic added:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-gt-parity.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-safe-prior-gt-parity.yml`

These set `sde_mu_hole_mode=safe_prior`, using the confidence-gated color prior as the hole mu. This is not claiming to match training; it checks whether a better hole anchor than `condition_lut` can give an acceptable no-retrain inference fallback. Logs now include `[MuAnchor Debug]` to detect anchor pass-through.


## 2026-04-23 18:45 safe-prior result: useful correction is in x7 trunk, not BrushNet extra

New evidence from the two safe-prior GT-parity logs:

- `test_ir-sde-no-extra-x7-48000-guarded-safe-prior-gt-parity_260423-181721.log`
  - runtime route:
    - `pretrain_model_G=.../48000_G.pth`
    - `no_extra_route=True`
    - `brushnet.enabled(config/runtime)=False/False`
    - `texture_core.enabled=False`
    - `mu_denoiser.enabled=False`
    - `sde_mu_hole_mode=safe_prior`
  - `000098_bottom`:
    - `prior_gt_l1=0.075773`
    - `lut_gt_l1=0.113057`
    - `final_gt_l1=0.041911`
    - `final_prior_l1=0.090143`
    - `[MuAnchor Debug] ... note=not a pure anchor copy`
  - `000098_center`:
    - `prior_gt_l1=0.204111`
    - `lut_gt_l1=0.271651`
    - `final_gt_l1=0.127347`
    - `final_prior_l1=0.124635`
    - `[MuAnchor Debug] ... note=not a pure anchor copy`

- `test_ir-sde-hybrid-baseline-trunk-x7-48000-brushnet-weak001-guarded-safe-prior-gt-parity_260423-182212.log`
  - runtime route:
    - `pretrain_model_G=.../48000_G.baseline_trunk_x7_extra.pth`
    - `brushnet.enabled(config/runtime)=True/True`
    - `brushnet.feature_scale(runtime)=0.01`
    - `texture_core.enabled=False`
    - `mu_denoiser.enabled=False`
    - `sde_mu_hole_mode=safe_prior`
  - `000098_bottom`:
    - `prior_gt_l1=0.075773`
    - `final_gt_l1=0.075746`
    - `final_prior_l1=0.000506`
    - `[MuAnchor Debug] ... note=pass-through`
  - `000098_center`:
    - `prior_gt_l1=0.204111`
    - `final_gt_l1=0.205065`
    - `final_prior_l1=0.001741`
    - `[MuAnchor Debug] ... note=pass-through`

Decision:

1. The good part of the 48000 no-retrain result is **not** BrushNet.
2. The useful correction is carried by the **x7 48000 main trunk itself** when all extra modules are bypassed.
3. The hybrid checkpoint (`baseline trunk + x7 extra tensors`) proves the extra branch is not an independent usable adapter: with BrushNet scale `0.01`, it almost exactly copies the safe-prior anchor; with larger scales it previously injected unrelated texture/artifacts.
4. Therefore do **not** use the hybrid checkpoint as a repair route. It is a diagnostic only.

Current best no-retrain route:

- wrapper class: `ConditionalUNetWithBrushNet` for key compatibility
- checkpoint: `.../ir-sde-brushnet-ft-x7/models/48000_G.pth`
- disabled modules:
  - `brushnet.enabled=false`
  - `texture_core.enabled=false`
  - `mu_denoiser.enabled=false`
- sampler:
  - guarded stochastic sampler
  - `sde_mu_hole_mode=safe_prior`
  - `restore_S_guidance=true`

Important caveat:

- The good safe-prior run above still used GT-parity routing:
  - `condition_known_source=gt_if_available`
  - `structure_source=gt_if_available`
  - logs show `StructureRoute ... resolved=gt has_gt=True`
- Before treating it as the actual test/deploy route, it must be checked without GT routing.

New no-GT/current-domain config:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-current-domain.yml`
  - same as the good no-extra safe-prior route, but:
    - `condition_known_source: lut`
    - `structure_source: lut`

Next validation:

```bash
cd /home/610-wws/Impainting/StrDiffusion/test/texture-1/config/inpainting
python test.py -opt options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-current-domain.yml
```

Expected log checks:

- `[RouteCheck] ... no_extra_route=True`
- `brushnet.enabled(config/runtime)=False/False`
- `sde_mu_hole_mode=safe_prior`
- `condition_known_source=lut structure_source=lut`
- `[StructureRoute] ... resolved=lut`
- `[MuAnchor Debug] ... note=not a pure anchor copy`

Interpretation:

- If current-domain safe-prior remains close to the GT-parity result, the practical no-retrain solution is confirmed.
- If it collapses, the remaining dependency is not BrushNet but **GT-derived structure/known-source parity**; the next target should be the LUT/current-domain structure source quality, not further tuning BrushNet.


## 2026-04-23 21:00 current-domain safe-prior: remaining problem is confidence/LUT fallback

User feedback:

- `ir-sde-no-extra-x7-48000-guarded-safe-prior-current-domain`
  - high-confidence hole regions now show some texture;
  - low-confidence regions still remain white / over-bright;
  - overall is usable-ish but worse than the best GT-parity run.

Log evidence:

- The route is correct:
  - `no_extra_route=True`
  - `brushnet.enabled(config/runtime)=False/False`
  - `texture_core.enabled=False`
  - `mu_denoiser.enabled=False`
  - `sde_mu_hole_mode=safe_prior`
  - `condition_known_source=lut`
  - `structure_source=lut`
  - `StructureRoute ... resolved=lut`
- The current-domain run is still not a pure anchor copy:
  - `000098_bottom`: `final_gt_l1=0.068701`, `prior_gt_l1=0.075773`, `lut_gt_l1=0.113057`, `final_prior_l1=0.113304`
  - `000098_center`: `final_gt_l1=0.170197`, `prior_gt_l1=0.204111`, `lut_gt_l1=0.271651`, `final_prior_l1=0.148450`
  - `000098_left`: `final_gt_l1=0.202886`, `prior_gt_l1=0.218738`, `lut_gt_l1=0.300428`, `final_prior_l1=0.245952`
- Compared to GT-parity, current-domain is consistently worse:
  - bottom worsens from `0.041911` to `0.068701`
  - center worsens from `0.127347` to `0.170197`

Current interpretation:

1. The no-extra 48000 trunk still repairs in current-domain, so the route is not fundamentally broken.
2. The degradation is now concentrated in the current-domain conditioning, not BrushNet/MGLC/Mu.
3. The user-observed high/low-confidence split matches the code path:
   - `_build_safe_brushnet_prior()` blends hole mu as
     - `confidence * color_prior + (1 - confidence) * condition_lut`
   - `ColorPriorGenerator.get_spatial_confidence()` intentionally makes boundary confidence high and center confidence low.
   - Therefore low-confidence hole pixels fall back toward `condition_lut`, which is the smooth/white fill that previously caused plain white regions.

Code instrumentation added:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - new inference knobs:
    - `inference.safe_prior_min_reliability`
    - `inference.safe_prior_confidence_power`
    - `inference.confidence_debug_threshold`
  - new logs:
    - `[Confidence Debug] reliability(min,p10,p50,p90,max), low_ratio`
    - `[ConfidenceSlice Debug] final_low/final_high mean and white ratio`
    - `[TargetByConfidence Debug] final/prior/lut GT L1 split by low/high confidence`

New ablation configs:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-floor06-current-domain.yml`
  - same current-domain route, but `safe_prior_min_reliability: 0.6`
  - purpose: test whether forcing the low-confidence area to use more color prior reduces white fallback.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-lut-known-gt-structure.yml`
  - `condition_known_source=lut`
  - `structure_source=gt_if_available`
  - purpose: isolate whether the lost quality is mainly from LUT-derived structure.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-gt-known-lut-structure.yml`
  - `condition_known_source=gt_if_available`
  - `structure_source=lut`
  - purpose: isolate whether known/source color parity is the main cause.

Decision rules:

- If `floor06-current-domain` improves low-confidence white regions, the concrete issue is overly conservative `safe_prior` confidence gating.
- If `floor06-current-domain` becomes brighter/worse, the raw color prior itself is too white in low-confidence regions; then the next fix should be color-prior generation, not the diffusion sampler.
- If `lut-known-gt-structure` recovers most of the GT-parity quality, current LUT structure is the bottleneck.
- If `gt-known-lut-structure` recovers most quality, the known-source / boundary target-domain alignment is the bottleneck.


## 2026-04-23 21:55 structure/known/floor ablation: structure is not the main bottleneck

User visual feedback:

- `floor06-current-domain`
  - more obvious split/white-edge behavior; forcing low-confidence area to trust prior more is not clean.
- `gt-known-lut-structure`
  - visually smoother / blurrier, but less broken.
- `lut-known-gt-structure`
  - still has split/white-edge; using GT structure alone does not fix the current-domain artifact.

Log evidence on `000098_bottom`:

- `lut-known-gt-structure`
  - `condition_known_source=lut`
  - `structure_source=gt_if_available`
  - `final_gt_l1=0.063383`
  - `final_prior_l1=0.098316`
  - low-confidence split:
    - `low_ratio=0.4281`
    - `final_gt_low=0.053104`
    - `final_gt_high=0.071078`
  - GT structure alone did not reproduce the earlier good GT-parity quality.
- `gt-known-lut-structure`
  - `condition_known_source=gt_if_available`
  - `structure_source=lut`
  - `final_gt_l1=0.048132`
  - `final_prior_l1=0.077393`
  - this is close to `floor06` numerically, but visually smoother.
- `floor06-current-domain`
  - `condition_known_source=lut`
  - `structure_source=lut`
  - `safe_prior_min_reliability=0.6`
  - `final_gt_l1=0.048420`
  - `prior_gt_l1=0.054512`
  - `final_prior_l1=0.052530`
  - `low_ratio=0` because the reliability floor turned the whole hole into high-confidence.

Interpretation:

1. `structure_source=gt_if_available` by itself is **not** the key. It leaves the LUT-known boundary / white-edge problem.
2. `condition_known_source=gt_if_available` recovers much more quality, even while `structure_source=lut`; the major gap is therefore the known/source-side conditioning and boundary consistency, not the structure network.
3. `floor06` proves that simply raising confidence globally can improve L1 but worsens visual naturalness. The color prior is not trustworthy enough to dominate the whole low-confidence hole.
4. Current best practical route is still the original `safe_prior-current-domain` or a mild reliability lift, not `floor06`.

New follow-up configs:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-floor03-current-domain.yml`
  - weaker floor (`0.3`) to see if it helps low-confidence white without the floor06 split.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-degraded-known-lut-structure.yml`
  - no-GT replacement for the helpful GT-known route; tests whether the observed degraded known pixels preserve more boundary texture than LUT-known.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-condition-mu-structure.yml`
  - tests structure generated from the actual `condition_mu`, so edge guidance matches the hole prior instead of pure LUT.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-no-extra-x7-48000-guarded-safe-prior-no-compose-guard-current-domain.yml`
  - disables final compose dilation/feather/white-guard to test whether visible white rims are introduced by final compositing rather than SDE prediction.

Decision rules:

- If `degraded-known-lut-structure` approaches `gt-known-lut-structure`, use degraded/observed known conditioning instead of LUT-known in current-domain.
- If `condition-mu-structure` reduces split/white edge, structure should be built from the same `condition_mu` used by the texture SDE.
- If `no-compose-guard` removes white rims but creates hard mask edges, the final fix should retune compose alpha, not the diffusion path.
- If `floor03` is better than both floor06 and no-floor, keep a mild reliability floor; otherwise abandon global floor and fix color-prior generation.


## 2026-04-23 23:15 latest 4-way current-domain ablation: compose / structure / degraded-known are not the root cause

Files:

- `C:/Users/admin/Desktop/test_ir-sde-no-extra-x7-48000-guarded-safe-prior-floor03-current-domain_260423-225011.log`
- `C:/Users/admin/Desktop/test_ir-sde-no-extra-x7-48000-guarded-safe-prior-degraded-known-lut-structure_260423-224812.log`
- `C:/Users/admin/Desktop/test_ir-sde-no-extra-x7-48000-guarded-safe-prior-no-compose-guard-current-domain_260423-224823.log`
- `C:/Users/admin/Desktop/test_ir-sde-no-extra-x7-48000-guarded-safe-prior-condition-mu-structure_260423-225559.log`

User visual feedback:

- `floor03-current-domain`: still the same failure mode; high-confidence area slightly better, low-confidence area still pale/white.
- `degraded-known-lut-structure`: close to `floor03`, maybe slightly worse in texture.
- `no-compose-guard-current-domain`: almost full-white hole; removing compose guard does **not** solve white regions.
- `condition-mu-structure`: close to the base current-domain route, no decisive improvement.

Key log evidence on `000098_bottom`:

- `floor03-current-domain`
  - `condition_known_source=lut`
  - `structure_source=lut`
  - `safe_prior_min_reliability=0.300`
  - `final_gt_l1=0.080157`
  - `final_gt_low=0.067269`
  - `final_gt_high=0.089806`
  - `x_init_hole(mean=0.8167,min=0.5815,max=0.9978,white=0.0000)`
- `degraded-known-lut-structure`
  - `condition_known_source=degraded`
  - `structure_source=lut`
  - `final_gt_l1=0.102415`
  - `final_gt_low=0.085890`
  - `final_gt_high=0.114786`
  - `x_init_hole(mean=0.8138,min=0.5815,max=0.9978,white=0.0000)`
- `no-compose-guard-current-domain`
  - `compose_mask_dilate=0`
  - `compose_feather=0`
  - `compose_white_guard=False`
  - `final_gt_l1=0.096035`
  - `final_gt_low=0.092341`
  - `final_gt_high=0.098800`
  - `x_init_hole(mean=0.8138,min=0.5815,max=0.9978,white=0.0000)`
- `condition-mu-structure`
  - `structure_source=condition_mu`
  - `final_gt_l1=0.082993`
  - `final_gt_low=0.079377`
  - `final_gt_high=0.085700`
  - `x_init_hole(mean=0.8138,min=0.5815,max=0.9978,white=0.0000)`

Cross-run interpretation:

1. `no-compose-guard` becoming almost full-white means the white block is **not** a final compose artifact. The compose guard is actually suppressing part of the white failure.
2. `degraded-known` does not approach the earlier `gt-known` quality, so simply swapping known/source from LUT to degraded is not the main fix.
3. `condition-mu-structure` is not materially better than the default LUT structure, so structure routing is still secondary.
4. `floor03` helps some numeric slices but does not change the failure mode. Reliability floor tuning is not the root fix.

Current strongest hypothesis:

- The real structural issue is now narrowed to **training semantics + weak target-domain color transform**:
  - x7 was trained with `train.sde_mu_hole_mode=condition_lut`, so the hole clean state seen during training is a bright filled target-domain estimate, **not a black/empty hole**.
  - In current inference runs, `x_init_hole(mean)` is still very bright (`0.61~0.82` depending on sample), so the reverse path starts from a pale anchor rather than a blank hole.
  - This matches the user observation: high-confidence areas can keep some texture, but low-confidence areas do not synthesize structure and instead stay pale/white.
  - Therefore the remaining issue is likely **not** “which inference switch to toggle”, but that the x7 checkpoint has learned to denoise around a filled hole anchor instead of learning robust hole generation from noise.

Secondary suspicion confirmed by code inspection:

- The current color transform path is intentionally conservative:
  - `_build_lut_transformed()` blends LUT output back with the original image using `effective_weight = lut_confidence * lut_strength`.
  - `ColorPriorGenerator.generate_quality()` heavily smooths Lab deltas (multi-scale + guided/bilateral filtering).
  - So even `lut_strength: 1.0` does **not** imply a strong visible domain shift; the actual transform can still be weak.
- This matches the user’s feeling that “变色了和没变色差不多”.

New instrumentation added:

- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - added `[LUTDelta Debug]`
    - `denoised_to_lut_known`
    - `denoised_to_lut_hole`
    - `condmu_to_lut_known`
    - `condmu_to_lut_hole`
    - `rawprior_to_safeprior_hole`
  - added `[ColorTransform Debug]`
    - `degraded_to_prefill_known`
    - `degraded_to_prefill_hole`
    - `prefill_to_lut_known`
    - `prefill_to_lut_hole`

Purpose of the new logs:

- directly quantify whether the LUT / color-change branch is actually doing a meaningful transformation,
- and whether `condition_mu` is already so close to `lut_transformed` / `safe_prior` that the reverse trajectory has no incentive to generate texture in hole pixels.

Practical implication for the next training fix (if retraining is needed):

- We do **not** need to delete BrushNet / color prior / confidence.
- The likely fix is to keep those modules as auxiliary guidance, but stop using a filled hole anchor as the main SDE clean state:
  - prefer `train.sde_mu_hole_mode=known_only`
  - or introduce a mixed schedule where only part of training uses filled hole anchors
- This preserves the innovation while preventing the color prior from dominating the restoration trajectory.



### 2026-04-24 restore note

- Restored the x8 code/config changes after accidental deletion.
- x8 config comments are ASCII-only to avoid Chinese mojibake in Windows/remote terminals.
- Chinese dataset path values are still saved as UTF-8 because the Linux dataset paths require them.


### 2026-04-25 x8 restore verification

This pass re-checked the restored x8 files and fixed one important config issue.

Files restored/verified:

- Train code:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/train.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/color_prior_generator.py`
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/data/__init__.py`
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/data/mural_inpainting_dataset.py`
- Test code:
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/color_prior_generator.py`
- Wrapper cleanup:
  - `PixelBrushNetLite` / `brushnet_lite` were removed from the PriorBrushNet route.
  - Only `PixelBrushNet` remains as the color-prior branch used by `ConditionalUNetWithBrushNet`.
- x8 configs:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x8-knownonly.yml`
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x8-knownonly-current-domain.yml`

Important fix:

- The two x8 YAML files had their Linux Chinese dataset directory accidentally saved as `?????`.
- Restored it to UTF-8 `裁剪的图片` in:
  - `degradation.mask_root`
  - `datasets.train.dataroot_GT`
  - `datasets.train.dataroot_mask`
  - `datasets.test.dataroot_degraded`
  - `datasets.test.dataroot_mask`
  - `datasets.test.dataroot_GT`
- Keep comments in these YAML files ASCII-only, but keep path values UTF-8.

x8 behavior that must stay fixed:

- `train.sde_mu_hole_mode: known_only`
  - SDE `condition_mu` keeps only known pixels: `condition_mu = mu_clean_lut * mask_known`.
  - The hole is not filled by `condition_lut`, `safe_prior`, or raw `color_prior`.
- `inference.sde_mu_hole_mode: known_only`
  - The clean `x_init` hole should be black/empty before reverse noise.
- `BrushNet/PriorBrushNet`
  - enabled as weak guidance only.
  - `feature_scale: 0.01`.
  - color prior is passed through confidence/consistency gating before injection.
- `MGLC` / `MuCleaner`
  - train config keeps them enabled for x8 training, as requested.
  - test config keeps module switches so later ablation can use `--set texture_core.enabled=false` and `--set mu_denoiser.enabled=false`.
- `lut_delta_gain: 1.5`
  - applied in both `ColorPriorGenerator` and LUT target construction to make the target-domain color shift visible.
  - It only strengthens color transform; it must not become a hole SDE anchor.

Validation commands run on 2026-04-25:

- `python -m py_compile` over restored train/test Python files.
- YAML parse check over all train/test option files.
- Static x8 route assertion: `X8_ROUTE_STATIC_ASSERT_OK`.
- Search check for removed PriorBrushNet lite route:
  - no `PixelBrushNetLite`
  - no `brushnet_lite`
  - no remaining `lite:` matches in train/test inpainting Python/YAML files
- Search check for corrupted `?????` dataset paths:
  - no remaining `?????` in train/test option YAML files.

Expected logs for the next x8 training/test run:

- Train:
  - `train.sde_mu_hole_mode=known_only`
  - `infer_x0_loss_weight=0.0`
  - `infer_x0_grad=False`
  - `LUT delta gain=1.500`
  - `stats_prefill_to_lut_known`
  - `stats_prefill_to_lut_hole`
  - `stats_training_target_to_lut`
- Test:
  - `sde_mu_hole_mode=known_only`
  - `expected_train_sde_mu_hole_mode=known_only`
  - `lut_delta_gain=1.500`
  - `final_white_ratio_hole`

Next recommended command:

```bash
cd /home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting
python train.py -opt options/train/ir-sde-brushnet-ft-x8-knownonly.yml
```


### 2026-04-25 x8 white-mask regression: root cause and config fix

User-reported symptom:

- `train_ir-sde-brushnet-ft-x8-knownonly_260425-110746.log` + `test_ir-sde-brushnet-x8-knownonly-current-domain_260425-172411.log`
- Inference output regressed to a bright/white mask area again, although `x_init` / `condition_mu` hole looked black.
- LUT/color shift was still visually weak.

Confirmed from logs:

- `known_only` itself was active, so the route did not silently fall back to `safe_prior` / `condition_lut` as the SDE hole anchor.
  - Train: `stats_sde_mu_hole_mean: 0.0000e+00`
  - Test: `x_init_hole(mean=0.0000,min=0.0000,max=0.0000,white=0.0000)`
- The actual failure was checkpoint/config drift:
  - Train loaded the wrong x7 initializer:
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x7/models/best_G.pth`
  - Intended x7 initializer is:
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/48000_G.pth`
  - Train writes x8 checkpoints under:
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x8-knownonly/models`
  - Test loaded a stale/different x8 checkpoint path:
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x8-knownonly/models/best_G.pth`
- Therefore the 2026-04-25 white-mask test result should not be used to judge the corrected x8 design: it was not testing the intended x8 checkpoint flow.

Config changes made in this pass:

- Train config fixed:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x8-knownonly.yml`
  - `path.pretrain_model_G` now points to the x7 `48000_G.pth` initializer.
  - `datasets.train.lut_delta_gain` raised from `1.5` to `3.0`.
- Test config fixed:
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x8-knownonly-current-domain.yml`
  - `path.pretrain_model_G` now points to the actual x8 experiment output:
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x8-knownonly/models/best_G.pth`
  - `datasets.test.lut_delta_gain` raised from `1.5` to `3.0`.

Why LUT gain was changed:

- Failed test log showed weak visible color shift:
  - `prefill_to_lut_known=0.016292`
  - `prefill_to_lut_hole=0.016817`
- Target for the next run: move typical `prefill_to_lut_known/hole` closer to roughly `0.03-0.06` without using LUT/color prior as the SDE hole anchor.

Validation run locally after the fix:

- YAML parse OK for both x8 train/test configs.
- UTF-8 Chinese dataset path values survived; no `?` replacement in dataset path lines.
- No `PixelBrushNetLite`, `brushnet_lite`, or `lite:` references in the two x8 configs.
- Python compile OK for touched train/test code paths.
- Static checks confirm:
  - train config still has `train.sde_mu_hole_mode=known_only`
  - test config still has `inference.sde_mu_hole_mode=known_only`
  - `brushnet.feature_scale=0.01`
  - train/test both have `lut_delta_gain=3.0`

Next commands that should be used:

```bash
cd /home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting
python train.py -opt options/train/ir-sde-brushnet-ft-x8-knownonly.yml
```

Expected train log must include:

```text
Loading model for G [/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/48000_G.pth]
train.sde_mu_hole_mode=known_only
LUT delta gain=3.000
```

After a checkpoint is produced, infer with:

```bash
cd /home/610-wws/Impainting/StrDiffusion/test/texture-1/config/inpainting
python test.py -opt options/test/ir-sde-brushnet-x8-knownonly-current-domain.yml --set inference.save_intermediates=true
```

Expected test log must include:

```text
Loading model for G [/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x8-knownonly/models/best_G.pth]
sde_mu_hole_mode=known_only
expected_train_sde_mu_hole_mode=known_only
lut_delta_gain=3.000
x_init_hole(mean=0.0000
```

If testing a numbered checkpoint before `best_G.pth` is updated, override explicitly:

```bash
--set path.pretrain_model_G=/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x8-knownonly/models/10000_G.pth
```


## 2026-04-25 correction: checkpoint path was not the root cause; x8 needed real blank-hole training signal

User confirmed the two x8 checkpoint directories contain the same weights, so the previous checkpoint-path-only explanation is not the root cause.

What the x8 run actually showed:

- `known_only` route was active: `stats_sde_mu_hole_mean=0`, and inference `x_init_hole(mean=0)`.
- However the normal training state still comes from `sde.generate_random_states(x0=training_target, mu=condition_mu)`, so the hole part of `state` contains the teacher-forced `B(t) * training_target` component.
- Inference does not have this target component in the hole. It starts from `condition_mu + noise`, with known_only hole near zero/noise.
- The old `infer_x0` branch was disabled in config and, even if enabled, was wrapped in `torch.no_grad()`, so it only logged metrics and gave no gradient.
- Therefore x8 trained on teacher-forced hole states but was tested on blank/noisy hole states, which reproduces the old white-mask failure.

Fix implemented in code:

- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
  - Added a real gradient path for the inference-like blank-hole x0 loss.
  - It uses microbatch + interval to avoid the previous VRAM doubling:
    - `infer_x0_microbatch`
    - `infer_x0_loss_interval`
    - `infer_x0_grad`
  - Added `require_infer_x0_grad_for_known_only`; x8 config now fails fast if known_only is used without active infer_x0 gradient.
  - Added training diagnostics:
    - `stats_train_state_hole_mean`
    - `stats_train_target_hole_mean`
    - `stats_train_condition_hole_mean`
    - `stats_train_state_hole_white_ratio`
    - `stats_train_target_hole_white_ratio`
    - `stats_train_state_to_target_hole`
    - `stats_train_state_to_condition_hole`
    - `stats_infer_x0_grad_enabled`
    - `stats_infer_x0_grad_active`
    - `stats_infer_x0_interval`
    - `stats_infer_x0_microbatch`
- `texture/config/inpainting/train.py`
  - Added TensorBoard scalar mappings for the new diagnostics.
- `texture/config/inpainting/options/train/ir-sde-brushnet-ft-x8-knownonly.yml`
  - `infer_x0_loss_weight: 0.01`
  - `infer_x0_grad: true`
  - `infer_x0_loss_interval: 4`
  - `infer_x0_microbatch: 2`
  - `require_infer_x0_grad_for_known_only: true`
  - `x0_recon_loss_weight: 0.01`
  - `lut_delta_gain: 4.5`
  - `logger.print_freq: 20`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - Added `target_like_gt_l1` to `[Target Debug]`.
  - Added `[WhiteMask Alert]` when final hole becomes too white/bright, so bad inference is obvious from the log.
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x8-knownonly-current-domain.yml`
  - `lut_delta_gain: 4.5` to match x8 train config.

Expected early train log after this fix:

- `[Model] train.sde_mu_hole_mode=known_only infer_x0_loss_weight=0.01 infer_x0_grad=True`
- `[Model] inference-like blank-hole x0 loss enabled: ... grad=True, interval=4, microbatch=2`
- No `[X8Guard]` exception.
- At the first printed iteration that is divisible by 4 (with print_freq=20, iter 20):
  - `loss_infer_x0` must be non-zero.
  - `stats_infer_x0_grad_enabled: 1.0000e+00`
  - `stats_infer_x0_grad_active: 1.0000e+00`
  - `stats_infer_x0_microbatch: 2.0000e+00`
  - `stats_sde_mu_hole_mean` should remain near 0 for known_only.

If `loss_infer_x0` is still 0 at iter 20/40/60, stop immediately; that means the old broken no-gradient path is still running.

Expected inference log after retraining with this fix:

- `[RouteCheck] ... sde_mu_hole_mode=known_only ... x_init_hole(mean=0.0000...)`
- `[Target Debug] ... final_white_ratio_hole=... target_like_gt_l1=...`
- No `[WhiteMask Alert]` for normal samples.

## 2026-04-26 x9-clean: 回归原版 StrDiffusion 训练语义

### 问题根因

x8 的 `infer_x0` + `x0_recon` 辅助 loss 额外做了一次 `sde.noise_fn()` 前向+反向（VRAM 翻倍），
且 teacher-forcing 导致模型在 hole 区只见过含 `B(t)*target` 的分布，推理时 hole 从 0 开始则偏白。
`lut_delta_gain=4.5` 导致全局颜色偏黄。

### x9-clean 改动

| 文件 | 改动 |
|------|------|
| `denoising_model.py` (训练) | 删除 `_estimate_x0_from_noise`、`x0_recon`、`infer_x0` 全部参数和 loss 分支；简化 `optimize_parameters` 为原版单 loss + MuDenoiser |
| `denoising_model.py` (训练) | `_build_lut_transformed` 改为自适应 fade-degree LUT |
| `train.py` | `condition_mu = training_target * mask_for_sde`（hole=0），删除 `mu_hole_mode` 分支 |
| `denoising_model.py` (推理) | LUT 逻辑同步为 fade-degree-aware；`condition_mu = known_source * mask_known` |
| 新增 `ir-sde-brushnet-ft-x9-clean.yml` | 训练配置，从 `best_G.pth` 初始化 |
| 新增 `ir-sde-brushnet-x9-clean-current-domain.yml` | 推理配置 |

### 首 100 步必须验证的指标

| 指标 | 正常范围 | 含义 |
|------|----------|------|
| `stats_sde_mu_hole_mean` | ≈ 0.0 | SDE mu 在 hole 区必须为零 |
| `stats_train_target_hole_mean` | 0.3~0.6 | target 是正常壁画颜色 |
| `stats_training_target_delta` | 0.02~0.04 | LUT 变色不过激 |
| `stats_noise_std` | 0.5~2.0 | 模型输出噪声正常 |

### 命令

```bash
# 训练
cd /home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting
python train.py -opt options/train/ir-sde-brushnet-ft-x9-clean.yml

# 推理
cd /home/610-wws/Impainting/StrDiffusion/test/texture-1/config/inpainting
python test.py -opt options/test/ir-sde-brushnet-x9-clean-current-domain.yml
```




## 2026-04-27 x10-consistent verdict: trunk still drifts on hard samples, stop this line

### Key evidence

1. `test_ir-sde-brushnet-x10-consistent-current-domain_260427-181445.log`
   - Loaded `12000_G.pth`.
   - Training log already showed `[Freeze] unfroze pretrained trunk at iter 10000` on `2026-04-27 17:00:42`.
   - Therefore `12000_G` is already **2000 steps after trunk unfreeze**.

2. `test_ir-sde-x10-12000-no-extra-no-structure_260427-185212.log`
   - `pure_no_extra_route=True`
   - `brushnet=False`
   - `texture_core=False`
   - `mu_denoiser=False`
   - `restore_S_guidance=False`
   - This means **the diffusion trunk alone is still bad even after all extra branches are removed**.

3. Hard samples are still clearly worse than the prior:
   - `000098_center: final_gt_l1=0.263132, prior_gt_l1=0.204685`
   - `000098_left:   final_gt_l1=0.322030, prior_gt_l1=0.217610`
   - `000098_right:  final_gt_l1=0.098969, prior_gt_l1=0.092670`
   - `000098_bottom: final_gt_l1=0.072646, prior_gt_l1=0.079317`

### Conclusion

- `x10-consistent` proved that train/infer condition alignment alone is not enough.
- Directly forcing the main diffusion trunk to learn the current LUT target domain still pushes hard samples toward a bright average-looking solution.
- Do **not** continue this line to `14000/16000/20000`.

### Errors already ruled out (do not repeat)

1. **Not a BrushNet-only failure**: results were still bad after `no-extra`.
2. **Not a structure-guidance-only failure**: results were still bad after `no-extra + no-structure`.
3. **Not simply because trunk was still frozen**: `12000_G` was already after unfreeze.
4. **Not a wrong checkpoint / wrong route mix-up**: logs explicitly showed `12000_G.pth` and the correct route flags.

## 2026-04-27 x11-officialinit: keep innovation, but move BrushNet closer to official training semantics

Goal: **preserve the innovation branch, but stop an unreliable color prior from dominating the main reverse trajectory.**

### Official BrushNet reference

Reference repo:
- `https://github.com/TencentARC/BrushNet`

The official pattern is:
- `BrushNetModel.from_unet(...)`
- freeze the original UNet trunk
- train the BrushNet branch as a safe residual conditioner

### Changes made in this round

#### 1. Initialize BrushNet from the pretrained trunk instead of random init

Files:
- `D:\code\kyihua\Impainting\StrDiffusion+e00	rain-3	exture\config\inpainting\models\pixel_brushnet.py`
- `D:\code\kyihua\Impainting\StrDiffusion	est	exture-1\config\inpainting\models\pixel_brushnet.py`

Added:
- `bootstrap_from_main_unet(main_unet, reset_zero_convs=True)`

Meaning:
- BrushNet encoder / mid layers copy weights from the baseline trunk.
- Zero-conv outputs remain zero-initialized, so the branch starts as a safe residual path.

#### 2. Feed BrushNet with the observed image known area, not just `xt-cond`

Files:
- `D:\code\kyihua\Impainting\StrDiffusion+e00	rain-3	exture\config\inpainting\modelsrushnet_wrapper.py`
- `D:\code\kyihua\Impainting\StrDiffusion	est	exture-1\config\inpainting\modelsrushnet_wrapper.py`
- `D:\code\kyihua\Impainting\StrDiffusion+e00	rain-3	exture\config\inpainting\models
etworks.py`
- `D:\code\kyihua\Impainting\StrDiffusion	est	exture-1\config\inpainting\models
etworks.py`

Added config:
- `brushnet.input_source`

Supported values:
- `residual`
- `xt`
- `observed_raw`
- `observed_known`

Current recommendation:
- `input_source: observed_known`

Meaning:
- BrushNet now uses the real observed image in the known area as its primary anchor.
- This is much closer to the official BrushNet masked-image conditioning style.
- The real known area becomes the main reference, while the color prior becomes only auxiliary guidance.

#### 3. Keep color prior and confidence as weak auxiliary inputs

Current policy:
- `color_prior` and `confidence` are still fed into BrushNet.
- Confidence is **not** used as a hard output gate by default.
- `feature_scale` stays small.
- `prior_dropout_prob` is set to `0.10` so BrushNet cannot overfit to the color prior.

This keeps the innovation branch, but weakens the prior so that it does not dominate the repair direction.

#### 4. Pass `observed_degraded` explicitly in both training and inference

Files:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`

Added:
- train: `brushnet_kwargs['observed_degraded'] = self.original_degraded`
- test: `observed_degraded=self.original_degraded`

This closes the remaining gap so that `observed_known` is not only supported by the wrapper, but also really receives the observed image from the upper call chain.

#### 5. x11 config updated

Files:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x11-officialinit.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x11-officialinit-current-domain.yml`

Updated values:
- `brushnet.input_source: observed_known`
- `brushnet.prior_dropout_prob: 0.10`

### Current judgement on MGLC / MuCleaner

#### MGLC

- It is **not** just a harmless denoiser.
- It directly changes mid / decoder feature trajectories inside the main network.
- Therefore it should stay off until BrushNet-only is proven stable.

#### MuCleaner / MuDenoiser

- It is mild only when used as an analysis or auxiliary denoising module.
- Once it is used to build `condition_mu` or `known_source`, it directly changes the SDE condition distribution.
- So it is **not** a pure denoising post-filter; it can change the main train/infer structure.

Recommended re-introduction order:
1. prove BrushNet-only is stable first
2. then try MGLC
3. only after that, try MuCleaner in the `condition_mu` path

### Cleanup

Deleted unused local reference files:
- `D:/code/ky/bihua/Impainting/_refs_brushnet.py`
- `D:/code/ky/bihua/Impainting/_refs_train_brushnet.py`

### Note on mojibake / encoding

- New comments added in this round are kept in UTF-8-friendly English to avoid more mojibake.
- Some historical comments in old files are still garbled, but this does **not** affect runtime logic.
- If needed later, do a separate comment-only cleanup pass instead of mixing it into model logic changes.


## 2026-04-28 x11-officialinit verdict: official BrushNet-style injection alone did not fix the failure

### Evidence from logs

Training log:
- `C:/Users/admin/Desktop/train_ir-sde-brushnet-ft-x11-officialinit_260428-002114.log`
- `[BrushNetInit] initialized BrushNet encoder from pretrained trunk`
- `[Freeze] frozen 215 pretrained trunk params until iter 999999`
- `loaded 231/326 tensors ... missing=95`
- `stats_sde_mu_hole_mean` stayed `0.0`
- `loss_main` decreased to about `2.0e-03`

Inference log:
- `C:/Users/admin/Desktop/test_ir-sde-brushnet-x11-officialinit-current-domain_260428-094728.log`
- Loaded checkpoint: `11000_G.pth`
- `brushnet.enabled(config/runtime)=True/True`
- `brushnet.input_source: observed_known`
- `texture_core=False`, `mu_denoiser=False`

Hard sample results still failed:
- `000098_center: final_gt_l1=0.258537, prior_gt_l1=0.204685`
- `000098_left:   final_gt_l1=0.351338, prior_gt_l1=0.217610`
- `[WhiteMask Alert]` still triggered with `final_hole_mean?0.90`

### What x11 already ruled out

1. Not a random-init BrushNet problem.
2. Not a wrong BrushNet input-source problem (`observed_known` was active).
3. Not an MGLC problem (`texture_core=False`).
4. Not a MuCleaner problem (`mu_denoiser=False`).
5. Not a trunk-drift problem in this specific line (trunk stayed frozen).

### Updated root-cause judgement

The remaining unstable part is the **main diffusion supervision target itself**.

Current mural training still does this in:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/train.py`

Current logic:
- `x0/GT/reverse target = LUT(denoised(degraded_full))`

This means even with official-style BrushNet conditioning, the diffusion objective is still being asked to model the LUT-shifted target domain directly. x11 shows that this is the wrong place to force the color-domain innovation.

### Recommended next direction (x12)

Do **not** abandon the innovation branch. Instead, separate the roles:

1. Main diffusion target returns to the **stable raw mural domain**.
2. BrushNet still uses:
   - `observed_known`
   - `color_prior`
   - `confidence`
3. LUT target is kept only as a **small hole-only color auxiliary objective**, not the main diffusion target.
4. MGLC stays off first.
5. MuCleaner stays off first.

Important dataset note:
- In this mural dataset, `Y_GT` is already a generated target-like image, not the raw degraded mural.
- Therefore x12 must not simply switch `training_target = Y_GT`.
- The correct stable raw-domain anchor is `Y_degraded_full`.

This keeps the innovation inside training/inference, but stops the score field from collapsing toward the bright LUT domain.


## 2026-04-28 x12-rawtarget-coloraux: move the innovation away from the main score target

### Why x12 is different from x11

x11 proved that:
- official BrushNet-style initialization was working,
- trunk freeze was working,
- MGLC and MuCleaner were not the direct cause,
- but the model still failed on hard samples.

The remaining problem was the place where the LUT-domain innovation was applied:
- x11 still made the **main diffusion target** live in the LUT-shifted domain.

x12 changes only that role assignment:
- the **main diffusion target** goes back to the stable raw mural domain (`Y_degraded_full`);
- the LUT target stays in training as a **small hole-only color auxiliary loss**;
- BrushNet remains the innovation carrier inside the network.

### x12 code changes

Files:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/train.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x12-rawtarget-coloraux.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x12-rawtarget-coloraux-current-domain.yml`

#### Train-side role split

In mural mode:
- `training_target_lut = LUT(denoised(Y_degraded_full))`
- if `datasets.train.main_target_domain == raw`:
  - `training_target = Y_degraded_full`
  - `condition_mu_source = Y_degraded_full`
- else:
  - keep the older LUT-domain path

This means the main SDE now learns the stable raw mural domain again, while the LUT branch is preserved separately.

#### Auxiliary color loss

Added in `models/denoising_model.py`:
- `color_aux_loss_weight`
- `color_aux_loss_start_iter`
- `color_aux_blur_kernel`
- `color_aux_clamp_b_min`

Mechanism:
- estimate `x0_hat` from the current noisy state and predicted noise;
- blur both `x0_hat` and `training_target_lut`;
- apply a **hole-only L1 loss** with a small weight.

Purpose:
- let the model learn the main repair in the stable raw domain;
- let the LUT-domain innovation act only as a weak color preference.

#### Inference-side alignment

x12 test config uses:
- `condition_known_source: degraded`
- `structure_source: degraded`

This keeps inference aligned with the raw-domain main SDE target while still allowing BrushNet to see color-prior inputs.

Additional x12 inference code fix:
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
- full-mode final composite no longer hard-forces `lut_transformed` on known pixels
- it now uses `prepared["known_source"]`, so the final protected known area follows `inference.condition_known_source`

This removes one remaining train/infer inconsistency that would otherwise keep x12 partially tied to the old LUT-domain composite path even after switching the main SDE route back to the raw domain.

### x12 intended behavior

Expected:
1. no more pressure for the main score field to collapse toward a bright LUT-domain average;
2. BrushNet remains active as an internal innovation branch;
3. LUT-domain supervision still exists, but only as a weak hole-only color guidance term;
4. the baseline trunk stays protected because x12 still uses the x11-style frozen-trunk BrushNet training setup.

### Errors already ruled out before x12

Do not revisit these as primary hypotheses unless new evidence appears:

1. **Random BrushNet initialization**  
   Ruled out by x11 official trunk bootstrap logs.

2. **Wrong BrushNet image input source**  
   Ruled out by x11 `observed_known` route verification.

3. **MGLC as main cause**  
   Ruled out because x11/x12 first-stage runs keep `texture_core=False`.

4. **MuCleaner / MuDenoiser as main cause**  
   Ruled out because x11/x12 first-stage runs keep `mu_denoiser=False`.

5. **Trunk drift caused by long fine-tuning**  
   Ruled out for x11/x12 style runs because the pretrained trunk is frozen.

6. **Main failure caused only by structure guidance**  
   Earlier no-extra/no-structure checks still failed, so structure guidance alone is not the root cause.

### 2026-04-28 x12 first-run postmortem: two concrete implementation bugs

Evidence:
- `C:/Users/admin/Desktop/train_ir-sde-brushnet-ft-x12-rawtarget-coloraux_260428-113026.log`
- `C:/Users/admin/Desktop/test_ir-sde-brushnet-x12-rawtarget-coloraux-current-domain_260428-134655.log`

Observed failure:
- the run went back to an almost fully white hole result
- example `000098_center`:
  - `raw_hole(mean=0.9774, white=0.9660)`
  - `final_hole(mean=0.9779, white=0.9696)`

Root causes found:

1. **The x12 color auxiliary loss was configured but not actually added into `optimize_parameters()`**
   - symptom: training log showed `loss_total == loss_main`
   - symptom: no `loss_color_aux` field appeared in the training log
   - consequence: the tested x12 checkpoint was effectively a raw-target-only run, not the intended raw-target + weak color-aux run

2. **`inference.structure_source=degraded` was wrong for mural inference**
   - in test-time mural inference, `degraded` refers to the observed white-hole canvas, not the full degraded mural image used as the raw-domain anchor during x12 training
   - consequence: structure guidance consumed a white-hole source image and pushed the reverse trajectory back toward the old white-mask failure

Fixes applied after this finding:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
  - wired the x12 hole-only LUT color auxiliary loss into `optimize_parameters()`
  - added `loss_color_aux`, `loss_color_aux_weighted`, and `stats_training_target_main_to_lut` logs
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/denoising_model.py`
  - added `structure_source=prefill` support using `prepared["denoised_original"]`
  - when `condition_known_source=degraded`, full-mode compose now prefers a prefilled source for feather blending instead of leaking the observed white-hole canvas
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x12-rawtarget-coloraux-current-domain.yml`
  - changed `structure_source: degraded` -> `structure_source: prefill`

Conclusion:
- the first x12 test result must **not** be used to judge the corrected x12 idea
- it was affected by two real implementation bugs and therefore was not a fair evaluation of the intended design

### 2026-04-28 x12 corrected run verdict

Evidence:
- `C:/Users/admin/Desktop/train_ir-sde-brushnet-ft-x12-rawtarget-coloraux_260428-152422.log`
- `C:/Users/admin/Desktop/test_ir-sde-brushnet-x12-rawtarget-coloraux-current-domain_260428-181409.log`

What is now confirmed to be correct:
- `loss_color_aux` is active in training
- `loss_total > loss_main`
- `stats_training_target_main_to_lut` is logged at about `0.03`
- inference uses `structure_source=prefill`
- inference no longer shows the old `0.97~0.99` white ratio collapse on the hard center sample

Representative improvement:
- earlier broken x12 center run: `final_white_ratio_hole ≈ 0.79`
- corrected x12 center run: `final_white_ratio_hole ≈ 0.22`

So the pipeline is no longer suffering from the same pure implementation bug as before.

### Remaining problem after corrected x12

Hard samples are still too bright and still worse than `prior_gt_l1`:
- center: `final_gt_l1=0.266505`, `prior_gt_l1=0.204685`
- left:   `final_gt_l1=0.386703`, `prior_gt_l1=0.217610`

Updated interpretation:
- the remaining issue is **not** YAML misconfiguration
- the remaining issue is the **training strategy mismatch**

Reason:
- x12 changed the main target domain to the raw mural domain
- but the pretrained baseline trunk was still kept frozen for the whole run
- that trunk was originally trained with a different target-domain prior
- BrushNet + weak color-aux can reduce the collapse, but cannot fully retarget the frozen score field on hard samples

### Next strategy: x12 stage-2 unfreeze

Do not restart from scratch again.

Instead:
1. continue from the corrected x12 checkpoint (`6000_G.pth`)
2. unfreeze the pretrained trunk with a **very small** main LR
3. keep the current x12 semantics:
   - raw main target
   - weak hole-only LUT color auxiliary
   - `condition_known_source=degraded`
   - `structure_source=prefill`
4. disable `restore_S_guidance` during the first stage-2 evaluation to avoid extra confounding while checking whether the trunk brightness prior is being corrected

Prepared configs:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x12-stage2-unfreeze.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x12-stage2-current-domain.yml`

Key stage-2 settings:
- `pretrain_model_G: .../ir-sde-brushnet-ft-x12-rawtarget-coloraux/models/6000_G.pth`
- `freeze_pretrained_until_iter: 0`
- `lr_G: 1e-6`
- `lr_new: 5e-6`
- test-side `restore_S_guidance: false`

### 2026-04-28 x12 stage-2 unfreeze verdict

Evidence:
- `C:/Users/admin/Desktop/train_ir-sde-brushnet-ft-x12-stage2-unfreeze_260428-184414.log`
- `C:/Users/admin/Desktop/test_ir-sde-brushnet-x12-stage2-current-domain_260428-201716.log`

What stage-2 confirms:
- the x12 training path is now genuinely active (`loss_color_aux` appears and `loss_total > loss_main`)
- inference is genuinely running with `restore_S_guidance=False` and `structure_source=prefill`
- the old pure white collapse is no longer the dominant failure mode (`final_white_ratio_hole` on hard samples dropped to very small values such as `0.0026` and `0.0064`)

Representative hard-sample metrics:
- center: `final_gt_l1=0.266677`, `prior_gt_l1=0.204685`, `final_lut_l1=0.132357`, `final_hole_mean=0.8605`
- left:   `final_gt_l1=0.349788`, `prior_gt_l1=0.217610`, `final_lut_l1=0.204826`, `final_hole_mean=0.8682`

Interpretation:
- the main issue is no longer a YAML / route / compose bug
- the remaining issue is an **over-bright / LUT-biased attractor**
- hard samples are still being pulled too far toward the color path even after the trunk is lightly unfrozen
- numerically the outputs are often closer to LUT-space than to the original prior / GT-space on difficult holes

Conclusion:
- do not keep chasing the old "all white due to broken route" hypothesis for stage-2
- the remaining correction must weaken the color pull rather than add more structure or more modules

### Next strategy: x12 stage-3 weak-color continuation

Prepared configs:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x12-stage3-weakcolor.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x12-stage3-weakcolor-current-domain.yml`

Stage-3 changes:
- continue from `ir-sde-brushnet-ft-x12-stage2-unfreeze/models/best_G.pth`
- reduce `brushnet.feature_scale: 0.03 -> 0.01`
- increase `brushnet.prior_dropout_prob: 0.10 -> 0.20`
- reduce `color_aux_loss_weight: 0.02 -> 0.005`
- delay color auxiliary start to `color_aux_loss_start_iter: 800`
- increase blur kernel for color auxiliary to `11` so the color hint stays lower-frequency
- keep `restore_S_guidance: false`
- keep `condition_known_source=degraded` and `structure_source=prefill`

Goal:
- preserve the innovation path
- stop the model from being over-dominated by the LUT / color branch on hard samples
- let the lightly unfrozen trunk settle back toward the stable raw-domain solution before the color hint is reintroduced
