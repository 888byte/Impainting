# StrDiffusion 鎺掗敊璺熻釜锛歯o-retrain 鍘熺増璺緞娑堣瀺

鏇存柊鏃堕棿锛?026-04-22
宸ヤ綔鍖猴細`D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3`

## 褰撳墠缁撹

- 杩欒疆涓嶉噸鏂拌缁冿紝鍙仛鎺ㄧ悊鏈熸秷铻嶏細淇濇寔 `ConditionalUNetWithBrushNet` wrapper 浠ュ吋瀹瑰綋鍓?checkpoint key锛屼絾鍏抽棴鏂板妯″潡銆?
- 淇濈暀 `restore_S_guidance=true`锛屽洜涓哄畠鏄師鐗?StrDiffusion 鐨勭粨鏋勫紩瀵艰矾寰勶紝涓嶈鍏?BrushNet/MGLC/Mu-Denoiser 鏂板鍒嗘敮銆?
- `infer_x0` 灞炰簬璁粌渚ц緟鍔?loss/鐩戞帶锛涙湰杞帹鐞嗕笉浣跨敤銆傝缁冮厤缃凡鎶?`train.infer_x0_loss_weight` 缃负 `0.0`锛屽苟鍥哄畾 `infer_x0_grad=false`銆?
- 鍚庣画鍙淮鎶や富鎺ㄧ悊鏍戯細`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting`锛涗笉鍐嶇户缁鐞?`_texture1_patch`銆?

## 宸叉帓闄?涓嶅啀浼樺厛杩界殑鏂瑰悜

- 涓嶅啀鎶婇棶棰樺崟鐙綊鍥犱簬 infer_x0 鍙嶄紶锛歚infer_x0_weighted` 鍙嶄紶浼氭槑鏄惧鍔犳樉瀛橈紝褰撳墠鍏堝叧闂€?
- 涓嶅啀缁х画鐩茬洰鍔犲己 MGLC 鎴栨墿澶?BrushNet 娉ㄥ叆锛氬厛璇佹槑鍘绘帀鏂板鍒嗘敮鍚庝富骞叉槸鍚﹁兘鎭㈠銆?
- 涓嶅啀浠呯湅鏈€缁堝浘鍒ゆ柇锛涘繀椤诲悓鏃剁湅鏃ュ織涓殑 route銆乧heckpoint銆乧ondition/mu銆乭ole 鍖虹粺璁″拰 `state_*` 杞ㄨ抗銆?
- 涓嶇洿鎺ュ垏鍥炲師濮?`ConditionalUNet` 绫伙紱杩欐牱鍙兘瀵艰嚧褰撳墠澧炲己 checkpoint 鐨?key 澶ч噺涓嶅尮閰嶃€傚綋鍓嶇敤 wrapper 浣?bypass 鏂板妯″潡銆?

## 鏈疆鏂板 no-retrain 閰嶇疆

涓绘帹鐞嗗叆鍙ｏ細
`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\test.py`

鍙洿鎺ヤ娇鐢ㄧ殑閰嶇疆锛?

```powershell
cd D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting

# 鎺ㄨ崘鍏堣窇锛氬叧闂?BrushNet/MGLC/Mu-Denoiser锛屼絾淇濇寔褰撳墠 x7 璁粌鍩?condition_lut
python test.py -opt options/test/ir-sde-no-extra-current-domain.yml

# 瀵圭収锛氭洿鎺ヨ繎鍘熺増 hole 渚?known_only 璇箟锛屼絾鍙兘鍜屽綋鍓?x7 璁粌鍩熶笉涓€鑷?
python test.py -opt options/test/ir-sde-no-extra-original-semantics.yml
```

濡傛灉 YAML 閲岀殑 `path.pretrain_model_G` 涓嶆槸浣犺娴嬬殑褰撳墠 checkpoint锛岀敤鍛戒护琛岃鐩栵紝涓嶉渶瑕佹敼鏂囦欢锛?

```powershell
python test.py -opt options/test/ir-sde-no-extra-current-domain.yml `
  --set path.pretrain_model_G=/path/to/current_G.pth
```

## 姣忔杩愯蹇呴』妫€鏌ョ殑鏃ュ織

鏌ユ壘杩欎簺鏃ュ織鏍囩锛?

- `[RouteCheck] network_G=ConditionalUNetWithBrushNet model_class=ConditionalUNetWithBrushNet no_extra_route=True`
- `[RouteCheck] brushnet.enabled(config/runtime)=False/False`
- `[RouteCheck] texture_core.enabled(config/runtime)=False/False`
- `[RouteCheck] mu_denoiser.enabled(config/available/runtime/has_weights)=False/.../False/False`
- `[RouteCheck] restore_S_guidance=True`
- `[RouteCheck] sde_mu_hole_mode=condition_lut` 鎴?`known_only`
- `[RouteCheck] pretrain_model_G=...`锛岀‘璁や笉鏄鐢ㄤ簡鏃?闅忔満 checkpoint
- `[LoadCheck] loaded ... missing=... unexpected=...`锛岀‘璁?checkpoint 鐪熷疄鍔犺浇鎯呭喌
- `[NoExtraRoute] BrushNet/MGLC/Mu-Denoiser are bypassed...`
- `[Inference Debug] ... cond_hole ... raw_hole ... final_hole ...`锛岃瀵?hole 鍖烘槸鍚﹀揩閫熺伆/榛?鐧藉缂?

## 鍒ゆ柇鍒嗘敮

1. 濡傛灉 `no_extra_current_domain` 鏄庢樉鏀瑰杽锛氫紭鍏堝洖鏌?BrushNet prior銆丮GLC 娉ㄥ叆銆丮u-Denoiser 鐨勪换涓€鍒嗘敮鏄惁鐮村潖涓诲共銆?
2. 濡傛灉 `no_extra_current_domain` 浠嶅樊锛屼絾 `known_only` 鏀瑰杽锛氫紭鍏堝洖鏌?`condition_lut` hole anchor 涓庡綋鍓?checkpoint 璁粌鐩爣鏄惁涓嶄竴鑷淬€?
3. 濡傛灉涓や釜 no-extra 閮藉樊锛氫紭鍏堝洖鏌ュ師鐗堜笉鍙橀噺鏄惁浠嶈鐮村潖锛歚training_target/GT/x0`銆乣condition_mu`銆乣sde.set_mu()`銆乣reverse_optimum_step` target 蹇呴』鍚屽煙涓€鑷淬€?
4. 濡傛灉鏃ュ織閲?`no_extra_route=False`锛氬厛涓嶈鐪嬪浘锛岃鏄庨厤缃病鏈夌湡姝ｅ叧闂柊澧炴ā鍧椼€?

## 涓嬫缁х画鎺掓煡鐨勬渶鐭矾寰?

- 鍏堣创瀵瑰簲 run 鐨?`test_*.log` 涓?`[RouteCheck]`銆乣[LoadCheck]`銆乣[Inference Debug]` 鍑犺銆?
- 鍚屾椂璐村悓涓€鏍锋湰鐨?`x_init.png`銆乣condition_mu.png`銆乣state_1/state_10/state_25/state_50/state_100/final.png`銆?
- 鍙湪 no-extra 璺緞澶嶇幇鍚庯紝鍐嶅喅瀹氭槸鍚﹂€愪釜鎵撳紑 BrushNet銆丮GLC銆丮u-Denoiser 鍋氫簩鍒嗗畾浣嶃€?

## 2026-04-22 21:20 鏃ュ織澶嶇洏锛歯o-extra 浠嶅彂鐧?

宸叉煡鐪嬩袱浠芥棩蹇楋細

- `C:\Users\admin\Desktop\test_ir-sde-no-extra-current-domain_260422-204651.log`
- `C:\Users\admin\Desktop\test_ir-sde-no-extra-original-semantics_260422-205916.log`

鍏抽敭浜嬪疄锛?

- `no_extra_route=True`锛岃鏄?BrushNet/MGLC/Mu-Denoiser 宸茬粡鐪熸鍏抽棴銆?
- 褰撳墠鏉冮噸 `32000_G.pth` 涓诲共鍔犺浇瀹屾暣锛歚loaded 231/231`锛屾柊澧炲垎鏀负 `unexpected=147`锛岀鍚堝叧闂柊澧炴ā鍧楀悗鐨勯鏈熴€?
- 浣嗕袱娆¤繍琛屼粛鏄?`discriminator_guidance=True` 涓?`deterministic_reverse=False`銆?
- 鍥犳杩欎袱娆¤繕涓嶆槸鈥滅函 no-extra sampler鈥濊瘖鏂紱浠嶆贩鍏ュ垽鍒櫒鍊欓€夐€夋嫨鍜岄殢鏈哄弽鎺ㄥ櫔澹般€?
- `condition_lut` 鐗堟湰涓紝`cond_hole` 鍜?`prior_hole` 閮戒笉鏄櫧鑹诧紝浣?`raw_hole/final_hole` 鐧借壊姣斾緥寰堥珮锛岃鏄庡彂鐧藉彂鐢熷湪 reverse sampler/score 杞ㄨ抗涓紝鑰屼笉鏄緭鍏ュ厛楠岀洿鎺ユ槸鐧借壊銆?

宸叉洿鏂颁袱涓?no-extra YAML锛?

- `inference.deterministic_reverse: true`
- `inference.discriminator_guidance.enabled: false`

涓嬩竴姝ュ厛閲嶈窇 `ir-sde-no-extra-current-domain.yml`锛屾棩蹇楀簲鍑虹幇锛?

- `no_extra_route=True pure_no_extra_route=True`
- `discriminator_guidance=False`
- `deterministic_reverse=True`

濡傛灉 pure no-extra 浠嶇櫧锛屽啀鍥炴煡涓诲共 score / S guidance / checkpoint 涓诲共璁粌鍒嗗竷锛涘鏋?pure no-extra 涓嶇櫧锛岄棶棰樹紭鍏堝綊鍥犱簬鍒ゅ埆鍣ㄥ紩瀵兼垨闅忔満 reverse noise銆?

## 2026-04-22 21:50 缁х画澶嶇洏锛氱函 no-extra 浠嶅彂鐧藉悗鐨勪笅涓€姝?

濡傛灉鏂?run 宸茬‘璁わ細

- `no_extra_route=True`
- `pure_no_extra_route=True`
- `discriminator_guidance=False`
- `deterministic_reverse=True`

浣?mask 鍖轰粛鐒跺彂鐧斤紝鍒欏彲浠ユ帓闄?BrushNet / MGLC / Mu-Denoiser / 鍒ゅ埆鍣?/ 闅忔満 reverse noise 鏄富鍥犮€?

涓嬩竴姝ヤ紭鍏堥殧绂?`restore_S_guidance` / SPADE 缁撴瀯璺緞锛氭柊澧炰袱涓瘖鏂厤缃細

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-no-structure-current-domain.yml`
- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-no-structure-known-only.yml`

瀹冧滑淇濇寔锛?

- BrushNet=false
- MGLC=false
- Mu-Denoiser=false
- D guidance=false
- deterministic_reverse=true
- 浣?`restore_S_guidance=false`

鍚屾椂鍦?`utils/sde_utils.py` 澧炲姞 `[Trajectory Debug]`锛屼細鍦ㄨ嫢骞插叧閿璁板綍 hole 鐨?mean/min/max/white 鍜?score_abs_mean锛岀敤鏉ョ‘璁ゆ槸鍝釜闃舵寮€濮嬫帹鐧姐€?

鍒ゆ柇锛?

1. 鍏虫帀 `restore_S_guidance` 鍚庢槑鏄句笉鐧斤細浼樺厛淇粨鏋?S/edge/SPADE 璺緞銆?
2. 鍏虫帀 `restore_S_guidance` 鍚庝粛鐧斤細浼樺厛鏌ュ綋鍓?32000_G 涓诲共 score 鏄惁宸茶璁粌鍒嗗竷甯﹀亸锛屾垨鑰呭綋鍓嶆帹鐞?condition/x0 涓庤缁冨煙浠嶄笉涓€鑷淬€?

## 2026-04-22 23:00 鍘诲櫔璧风偣淇

鐢ㄦ埛鎸囧嚭鎺ㄧ悊鍘诲櫔璧风偣涓嶅锛氬師鐗?StrDiffusion inpainting 鐨?clean start/mu 搴旇鏄?`known_pixels * mask_known`锛宧ole 鍖哄湪鍔犲櫔鍓嶆槸榛戣壊锛沗noise_state()` 涔嬪悗 hole 鎵嶄細鏈夊皬骞?Gaussian noise銆?

鏃ュ織纭涔嬪墠 `no-structure-current-domain` 浣跨敤浜嗭細

- `sde_mu_hole_mode=condition_lut`
- `cond_hole(mean鈮?.69~0.81)`

杩欒鏄?hole 琚?LUT 鍐呭棰勫～浜嗭紝纭疄涓嶆槸鍘熺増榛戞礊璧风偣锛屼篃浼氬鑷村幓鍣繃绋嬬湅璧锋潵鍙槸杞诲井璋冩暣銆?

宸蹭慨鏀癸細

- `ir-sde-no-extra-current-domain.yml` 鏀逛负 `sde_mu_hole_mode: known_only`
- `ir-sde-no-extra-no-structure-current-domain.yml` 鏀逛负 `sde_mu_hole_mode: known_only`
- 鎺ㄧ悊鏃ュ織鏂板 `x_init_hole(...)` 鍜?`noisy_start_hole(...)`
- 涓棿鍥炬柊澧?`x_start_noisy.png`

涓嬩竴娆″簲浼樺厛璺戝甫缁撴瀯鐨勫師鐗堣矾寰勶細

`D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-current-domain.yml`

鏈熸湜鏃ュ織锛?

- `sde_mu_hole_mode=known_only`
- `x_init_hole(mean鈮?,min鈮?,max鈮?,white=0)`
- `noisy_start_hole` 涓烘帴杩?0 鐨勫皬鍣０锛岃€屼笉鏄?LUT 鐏?鐧藉潡

濡傛灉杩欎釜璧风偣姝ｇ‘鍚庝粛娌℃湁鏈夋晥鍘诲櫔锛屽啀鏌ヤ富骞?score/璁粌鍒嗗竷銆?

## 2026-04-22 23:35 榛戞礊璧风偣姝ｇ‘锛屼絾绾?mean sampler 鏃犵汗鐞?

鏂版棩蹇?`test_ir-sde-no-extra-current-domain_260422-232202.log` 璇存槑锛?

- `x_init_hole(mean=0,min=0,max=0)`锛屽師鐗堥粦娲炶捣鐐瑰凡缁忔纭€?
- `noisy_start_hole` 鍙湁灏忓櫔澹帮紝璧风偣闂宸蹭慨姝ｃ€?
- 鍦ㄧ函璇婃柇璺緞涓細`deterministic_reverse=True` 涓?`discriminator_guidance=False`锛宧ole 浠庨粦鑹查€愭笎鍙樻垚娴呰壊/鐧借壊鍧囧€煎潡锛屼絾娌℃湁绾圭悊銆?

杩欒鏄庡綋鍓嶉棶棰樹笉鍐嶆槸璧风偣閿欒锛岃€屾槸锛氬叧闂墍鏈夎緟鍔╁垎鏀苟浣跨敤 deterministic mean sampler 鏃讹紝褰撳墠涓诲共鍙粰鍑哄潎鍊?棰滆壊瓒嬪娍锛屼笉鑳界敓鎴愮汗鐞嗙粏鑺傘€?

鏂板涓や釜涓嬩竴姝ラ厤缃細

1. `ir-sde-no-extra-original-sampler-known-only.yml`
   - BrushNet/MGLC/Mu-Denoiser 浠嶅叧闂?
   - `known_only` 榛戞礊璧风偣
   - 鎭㈠鍘熺増 sampler 椋庢牸锛歚deterministic_reverse=false` + `discriminator_guidance=true`
   - 鐢ㄦ潵纭鍘熺増闅忔満+D sampler 鏄惁鑳界粰褰撳墠涓诲共甯﹀洖绾圭悊銆?

2. `ir-sde-brushnet-only-known-start.yml`
   - 榛戞礊璧风偣 + 鍘熺増缁撴瀯寮曞
   - 鍙墦寮€ BrushNet锛屽叧闂?MGLC/Mu-Denoiser/D guidance
   - 鐢ㄦ潵楠岃瘉鈥滃彧娉ㄥ叆棰滆壊鍏堥獙鍥撅紝涓嶆敼涓诲共缁撴瀯鈥濇槸鍚﹁兘鎻愪緵绾圭悊/棰滆壊鍙傝€冦€?

鍒ゆ柇锛?

- 濡傛灉 original-sampler 鏈夌汗鐞嗭細涔嬪墠绾?deterministic 璇婃柇杩囦簬淇濆畧锛屾渶缁堣矾寰勯渶瑕佷繚鐣欏師鐗?sampler銆?
- 濡傛灉 original-sampler 浠嶆棤绾圭悊锛屼絾 brushnet-only 鏈夌汗鐞嗭細璇存槑褰撳墠涓诲共鍗曠嫭涓嶅锛孊rushNet prior 鏄繀瑕佹潯浠躲€?
- 濡傛灉涓よ€呴兘鏃犵汗鐞嗭細浼樺厛鏌?BrushNet 杈撳叆/鐗瑰緛娉ㄥ叆寮哄害锛屾垨褰撳墠 32000_G 涓诲共璁粌鍒嗗竷宸茬粡鍋忔垚鍧囧€煎～鍏呫€?

## 2026-04-23 00:25 original-sampler / BrushNet-only 浠嶅け璐ュ悗鐨勭粨璁?

鏂版棩蹇楋細

- `test_ir-sde-no-extra-original-sampler-known-only_260422-235949.log`
  - no-extra 璺敱鎴愮珛锛欱rushNet=false, MGLC=false, Mu-Denoiser=false銆?
  - 榛戞礊璧风偣鎴愮珛锛歚x_init_hole(mean=0,min=0,max=0)`銆?
  - 鎭㈠闅忔満 reverse + D guidance 鍚庯紝hole 浠嶄富瑕佸彉鎴愭祬鐏?鐧藉潡锛屾病鏈夌汗鐞嗐€?
- `test_ir-sde-brushnet-only-known-start_260423-000710.log`
  - BrushNet 鏉冮噸纭疄鍔犺浇骞跺弬涓庯細`loaded 326/326`锛孊rushNet runtime=true銆?
  - 璧风偣浠嶆纭紝浣嗚緭鍑哄彧鍑虹幇榛?娣辫壊鍧楋紝涓嶆槸鏈夋晥绾圭悊淇銆?

鍥犳宸叉帓闄わ細

1. 鈥滃彧鏄捣鐐逛笉鏄粦娲炩€?鈥斺€?宸蹭慨姝ｏ紝浠嶅け璐ャ€?
2. 鈥滃彧鏄?deterministic mean sampler 澶繚瀹堚€?鈥斺€?鎭㈠闅忔満+D 鍚庝粛澶辫触銆?
3. 鈥淏rushNet 娌″姞杞?娌＄敓鏁堚€?鈥斺€?BrushNet-only 鏄庢樉鏀瑰彉杞ㄨ抗锛屼絾鏂瑰悜閿欒銆?
4. 鈥滃叧鎺夋柊澧炴ā鍧楀氨鑳芥仮澶嶅師鐗堣兘鍔涒€?鈥斺€?褰撳墠 32000_G 鐨?no-extra 涓诲共璺緞浠嶄笉鑳芥仮澶嶇汗鐞嗐€?

涓嬩竴姝ヤ笉鍐嶇户缁洸璋?x7 鎺ㄧ悊鍙傛暟锛岃€屾槸鍋?**鍘熺増 StrDiffusion sampler parity**锛氬綋鍓?enhanced 璺緞铏界劧妯℃嫙浜嗗師鐗堥噰鏍凤紝浣嗕粛涓嶆槸閫愯鍘熺増 `reverse_sde` 鍒嗘敮銆傚凡鏂板涓や釜閰嶇疆锛岀敤鍚屼竴涓?wrapper/checkpoint 鍋氭棤閲嶈瀵圭収锛?

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-legacy-reverse-current-domain.yml`
  - no BrushNet/MGLC/Mu
  - `force_legacy_reverse=true`
  - `condition_known_source=lut`, `structure_source=lut`
  - 鐩殑锛氶獙璇佹槸鍚︽槸 enhanced reverse 鍒嗘敮鏈韩鍋忕鍘熺増銆?

- `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-no-extra-legacy-reverse-gt-parity.yml`
  - no BrushNet/MGLC/Mu
  - `force_legacy_reverse=true`
  - `condition_known_source=gt_if_available`, `structure_source=gt_if_available`
  - 鐩殑锛氬湪璁粌闆?鏈?GT 鐨勬牱鏈笂灏介噺璐磋繎鍘熺増 StrDiffusion 娴嬭瘯璇箟锛屽垽鏂綋鍓?x7 涓诲共鏄惁杩樹繚鐣欏師鐗堜慨澶嶈兘鍔涖€?

鏂板鏃ュ織搴斿嚭鐜帮細

- `force_legacy_reverse=True`
- `[LegacyReverseRoute] ... enhanced_inference=false`
- `[StructureRoute] ... resolved=lut` 鎴?`resolved=gt`

鍒ゆ柇锛?

- current-domain legacy 鑳芥敼鍠勶細涔嬪墠闂闆嗕腑鍦?enhanced reverse/composite 閫昏緫銆?
- current-domain 浠嶅樊浣?gt-parity 鏀瑰杽锛氶棶棰橀泦涓湪褰撳墠鎺ㄧ悊 condition/structure 鏋勯€犲拰鍘熺増璁粌/娴嬭瘯璇箟涓嶄竴鑷淬€?
- 涓よ€呬粛宸細褰撳墠 x7 checkpoint 鐨勪富骞?score 宸茬粡琚悗缁缁?鏂板妯″潡甯﹀亸锛涙棤閲嶈鍏抽棴妯″潡鏃犳硶鎭㈠鍘熺増鑳藉姏锛岄渶瑕佸洖鍒板師鐗堟敹鏁?checkpoint 鎴栧仛涓诲共鍐荤粨/灏忓涔犵巼鎭㈠璁粌銆?

## 2026-04-23 10:50 legacy reverse + GT parity 浠嶅け璐?

鏂版棩蹇楋細

- `test_ir-sde-no-extra-legacy-reverse-current-domain_260423-103607.log`
- `test_ir-sde-no-extra-legacy-reverse-gt-parity_260423-104051.log`

纭淇℃伅锛?

- `force_legacy_reverse=True`锛岀‘瀹炶蛋浜嗗師鐗?`reverse_sde` 鍒嗘敮锛坄enhanced_inference=false`锛夈€?
- no-extra 璺敱鎴愮珛锛欱rushNet=false, MGLC=false, Mu-Denoiser=false銆?
- `loaded 231/231 tensors into ConditionalUNetWithBrushNet`锛屽綋鍓?x7 checkpoint 鐨勫師鐗堜富骞叉潈閲嶅叏閮ㄥ姞杞姐€?
- 璧风偣浠嶆纭細`x_init_hole(mean=0,min=0,max=0)`銆?
- GT parity 涓粨鏋勬潵婧愮‘瀹炰负 GT锛歚[StructureRoute] ... resolved=gt has_gt=True`銆?

缁撹锛?

鍗充娇浣跨敤鍘熺増 sampler 鍒嗘敮锛屽苟涓斿湪璁粌闆?鏈?GT 鎯呭喌涓嬫妸 condition/structure 灏介噺璐磋繎鍘熺増 StrDiffusion锛屽綋鍓?x7 鐨?`32000_G.pth` 涓诲共浠嶄笉鑳戒慨澶嶃€傛鏃堕棶棰樺熀鏈笉鍦ㄦ帹鐞嗗垎鏀€佽捣鐐广€丏 guidance銆丅rushNet/MGLC/Mu 寮€鍏筹紝鑰屾槸褰撳墠 checkpoint 鐨勪富骞?score 宸茬粡鍋忕鍘熺増鍙慨澶嶈В銆?

涓嬩竴姝ラ獙璇佷笉鍐嶇户缁敤 x7 鐩茶皟锛岃€屾槸鍋氫袱浠朵簨锛?

1. **鍘熺増 baseline checkpoint parity**
   - 鏂板閰嶇疆锛?
     `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-baseline-original-checkpoint-gt-parity.yml`
   - 榛樿鍔犺浇锛?
     `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
   - 鑻ュ畠鑳芥甯镐慨澶嶏紝鍒欒鏄庡綋鍓嶆帹鐞嗕唬鐮佸凡缁忚冻澶熸帴杩戝師鐗堬紝闂闆嗕腑鍦?x7 checkpoint 涓诲共琚缁冨甫鍋忋€?
   - 鑻ュ畠涔熶笉鑳戒慨澶嶏紝鍒欒繕闇€瑕佺户缁榻愬綋鍓?`texture-1` 娴嬭瘯鏍戜笌鍘熺増 `StrDiffusion/test/texture` 鐨勬暟鎹?缁撴瀯鐢熸垚銆?

2. **checkpoint 涓诲共婕傜Щ瀹¤**
   - 鏂板鑴氭湰锛?
     `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\tools\checkpoint_trunk_audit.py`
   - 姣旇緝鍘熺増 baseline 鍜?x7 涔嬮棿鍏变韩鐨?ConditionalUNet 涓诲共鏉冮噸婕傜Щ銆?
   - 杈撳嚭锛?
     `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\STRDIFFUSION_CHECKPOINT_DRIFT.md`

濡傛灉 baseline parity 姝ｅ父涓?drift 寰堝ぇ锛屽悗缁棤閲嶈鍙€夋柟妗堝彧鏈夛細

- 鐩存帴浣跨敤鍘熺増 baseline checkpoint 鎺ㄧ悊锛涙垨
- 鍋?checkpoint surgery锛氭妸 x7 checkpoint 涓師鐗堜富骞叉潈閲嶆浛鎹㈠洖 baseline锛屽彧淇濈暀鏂板妯″潡鏉冮噸锛屽啀鍋?no-extra/BrushNet-only 鎺ㄧ悊娑堣瀺銆?

## 2026-04-23 12:10 baseline checkpoint parity 鏈夋晥鍚庣殑涓嬩竴姝?

鐢ㄦ埛鍙嶉 `ir-sde-baseline-original-checkpoint-gt-parity`锛?

- 鍘熺増 baseline checkpoint 宸茬粡鑳戒慨锛岃竟缂?缁熶竴棰滆壊鍖哄煙鏄庢樉鎭㈠銆?
- 浠嶅瓨鍦ㄥ眬閮ㄩ粦/娣辫壊鏂戝潡锛屽鏉傜汗鐞嗗尯鍩熺粏鑺備笉瓒炽€?

缁撳悎 `STRDIFFUSION_CHECKPOINT_DRIFT.md`锛?

- baseline vs x7 鍏变韩涓诲共 `231` 涓?tensor銆?
- x7 澶氬嚭 `189` 涓柊澧炴ā鍧?tensor銆?
- 涓诲共 global `relative_rms鈮?.060467`锛屾紓绉绘渶澶у湪 `mid_block*`銆乣ups.*` 鍜屾棭鏈?`downs.0`銆?

缁撹锛?

1. 褰撳墠 `texture-1` 鎺ㄧ悊閾捐矾宸茬粡瓒冲鎺ヨ繎鍘熺増锛屽師鐗?baseline checkpoint 鍙互鎭㈠鍩烘湰淇鑳藉姏銆?
2. x7 鍏抽棴鏂板妯″潡浠嶅け璐ワ紝璇存槑 x7 鐨勫師鐗堜富骞茶鍚庣画璁粌甯﹀亸锛岃€屼笉鏄帹鐞嗚矾寰勬湰韬潖銆?
3. baseline 鐨勯粦鏂戞洿澶氬儚鍘熺増闅忔満+D adaptive sampler 鐨勫€欓€夐€夋嫨浼奖锛屽挨鍏跺湪鍙傝€冨厛楠?GT/LUT 鐨?hole 鍖烘病鏈夋殫鑹插唴瀹规椂锛孌 浠嶅彲鑳介€夊埌灞€閮ㄨ繃鏆?proposal銆?

宸叉柊澧?淇敼锛?

- 澧炲己鐗?D guard锛歚D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\utils\sde_utils.py`
  - 澧炲姞 dark-ratio 妫€鏌ャ€?
  - enhanced guarded sampler 浣跨敤 color_prior/GT/LUT 浣滀负瀹夊叏鍙傝€冿紝鑰屼笉鏄粦娲?`mu`銆?
  - 鏃ュ織鏂板锛歚[DiscriminatorGuard] rejected_candidates=...`銆?

- 鏂板 baseline sampler 瀵圭収閰嶇疆锛?
  - `ir-sde-baseline-original-checkpoint-guarded-sampler-gt-parity.yml`
    - baseline G锛宔nhanced guarded stochastic + D銆?
    - 鐩爣锛氫繚鐣欑汗鐞嗗悓鏃舵姂鍒堕粦鏂戙€?
  - `ir-sde-baseline-original-checkpoint-deterministic-gt-parity.yml`
    - baseline G锛宒eterministic + no D銆?
    - 鐩爣锛氶獙璇侀粦鏂戞槸鍚︾敱 stochastic/D 閫犳垚锛涢鏈熸洿骞虫粦銆佺汗鐞嗘洿灏戙€?

- 鏂板 checkpoint surgery 鑴氭湰锛?
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\tools\make_baseline_trunk_hybrid.py`
  - 鐢熸垚锛歜aseline trunk + x7 added modules銆?
  - 榛樿杈撳嚭锛?
    `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x7/models/32000_G.baseline_trunk_x7_extra.pth`

- 鏂板 hybrid BrushNet-only 瀵圭収閰嶇疆锛?
  - `ir-sde-hybrid-baseline-trunk-brushnet-only-guarded-gt-parity.yml`
  - `ir-sde-hybrid-baseline-trunk-brushnet-only-deterministic-gt-parity.yml`

涓嬩竴姝ラ『搴忥細

1. 鍏堣窇 baseline guarded sampler锛岀湅榛戞枒鏄惁娑堝け鍚屾椂淇濈暀绾圭悊銆?
2. 鍐嶈窇 baseline deterministic锛岀‘璁ら粦鏂戞槸鍚︾敱 D/stochastic 寮曞叆銆?
3. 濡傛灉 baseline guarded 杈冨ソ锛屽啀鐢熸垚 hybrid checkpoint 骞惰窇 hybrid BrushNet-only锛岄獙璇?x7 鐨?BrushNet 鑳藉惁鍦?baseline 涓诲共涓婃彁渚涢鑹?绾圭悊鍏堥獙銆?

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
  - Therefore the remaining issue is likely **not** 鈥渨hich inference switch to toggle鈥? but that the x7 checkpoint has learned to denoise around a filled hole anchor instead of learning robust hole generation from noise.

Secondary suspicion confirmed by code inspection:

- The current color transform path is intentionally conservative:
  - `_build_lut_transformed()` blends LUT output back with the original image using `effective_weight = lut_confidence * lut_strength`.
  - `ColorPriorGenerator.generate_quality()` heavily smooths Lab deltas (multi-scale + guided/bilateral filtering).
  - So even `lut_strength: 1.0` does **not** imply a strong visible domain shift; the actual transform can still be weak.
- This matches the user鈥檚 feeling that 鈥滃彉鑹蹭簡鍜屾病鍙樿壊宸笉澶氣€?

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

- The two x8 YAML files had their Linux Chinese dataset directory accidentally saved as `"""""`.
- Restored it to UTF-8 `瑁佸壀鐨勫浘鐗嘸 in:
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
- Search check for corrupted `"""""` dataset paths:
  - no remaining `"""""` in train/test option YAML files.

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
- UTF-8 Chinese dataset path values survived; no `"` replacement in dataset path lines.
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

## 2026-04-26 x9-clean: 鍥炲綊鍘熺増 StrDiffusion 璁粌璇箟

### 闂鏍瑰洜

x8 鐨?`infer_x0` + `x0_recon` 杈呭姪 loss 棰濆鍋氫簡涓€娆?`sde.noise_fn()` 鍓嶅悜+鍙嶅悜锛圴RAM 缈诲€嶏級锛?
涓?teacher-forcing 瀵艰嚧妯″瀷鍦?hole 鍖哄彧瑙佽繃鍚?`B(t)*target` 鐨勫垎甯冿紝鎺ㄧ悊鏃?hole 浠?0 寮€濮嬪垯鍋忕櫧銆?
`lut_delta_gain=4.5` 瀵艰嚧鍏ㄥ眬棰滆壊鍋忛粍銆?

### x9-clean 鏀瑰姩

| 鏂囦欢 | 鏀瑰姩 |
|------|------|
| `denoising_model.py` (璁粌) | 鍒犻櫎 `_estimate_x0_from_noise`銆乣x0_recon`銆乣infer_x0` 鍏ㄩ儴鍙傛暟鍜?loss 鍒嗘敮锛涚畝鍖?`optimize_parameters` 涓哄師鐗堝崟 loss + MuDenoiser |
| `denoising_model.py` (璁粌) | `_build_lut_transformed` 鏀逛负鑷€傚簲 fade-degree LUT |
| `train.py` | `condition_mu = training_target * mask_for_sde`锛坔ole=0锛夛紝鍒犻櫎 `mu_hole_mode` 鍒嗘敮 |
| `denoising_model.py` (鎺ㄧ悊) | LUT 閫昏緫鍚屾涓?fade-degree-aware锛沗condition_mu = known_source * mask_known` |
| 鏂板 `ir-sde-brushnet-ft-x9-clean.yml` | 璁粌閰嶇疆锛屼粠 `best_G.pth` 鍒濆鍖?|
| 鏂板 `ir-sde-brushnet-x9-clean-current-domain.yml` | 鎺ㄧ悊閰嶇疆 |

### 棣?100 姝ュ繀椤婚獙璇佺殑鎸囨爣

| 鎸囨爣 | 姝ｅ父鑼冨洿 | 鍚箟 |
|------|----------|------|
| `stats_sde_mu_hole_mean` | 鈮?0.0 | SDE mu 鍦?hole 鍖哄繀椤讳负闆?|
| `stats_train_target_hole_mean` | 0.3~0.6 | target 鏄甯稿鐢婚鑹?|
| `stats_training_target_delta` | 0.02~0.04 | LUT 鍙樿壊涓嶈繃婵€ |
| `stats_noise_std` | 0.5~2.0 | 妯″瀷杈撳嚭鍣０姝ｅ父 |

### 鍛戒护

```bash
# 璁粌
cd /home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting
python train.py -opt options/train/ir-sde-brushnet-ft-x9-clean.yml

# 鎺ㄧ悊
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
- `[WhiteMask Alert]` still triggered with `final_hole_mean"0.90`

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
- earlier broken x12 center run: `final_white_ratio_hole 鈮?0.79`
- corrected x12 center run: `final_white_ratio_hole 鈮?0.22`

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

### 2026-04-29 decisive structural findings (evidence-backed, not a parameter guess)

After re-checking the original StrDiffusion code paths, two deeper structural issues were confirmed.

#### Finding A: the original "baseline works" result is not solving the same inference task

Evidence from the original repository:
- Training: `D:/code/ky/bihua/Impainting/StrDiffusion/train/texture/config/inpainting/train.py:243-244`
  - `timesteps, states = sde.generate_random_states(x0=Y_GT, mu=Y_GT*mask)`
  - `model.feed_data(states, Y_GT*mask, Y_GT, ...)`
- Testing: `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture/config/inpainting/test.py:170-172`
  - `noisy_state = sde.noise_state(Y_GT * mask)`
  - `model.feed_data(noisy_state, Y_GT * mask, Y_GT, ...)`

This means the original baseline is trained **and evaluated** with a target-domain known-region condition (`Y_GT * mask`).
It is **not** evaluated with the current-domain degraded known-region condition (`degraded * mask_known`).

Therefore, the historical statement "the original baseline already works" cannot be used as proof that the current degraded-known inference route is already solved by the baseline task.
It only proves that the original target-domain-known task is solvable.

#### Finding B: the current train/test generator definitions were not identical

Evidence from stage-3 inference log:
- `C:/Users/admin/Desktop/test_ir-sde-brushnet-x12-stage3-weakcolor-current-domain_260428-234259.log`
- generator load line showed: `loaded 246/246 tensors ... unexpected=80`

This is a real structural mismatch: inference was discarding 80 generator tensors from the training checkpoint.
A direct file comparison confirmed that the following train/test module files differed:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/modules/DenoisingUNet_arch.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/modules/DenoisingUNet_arch.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/modules/__init__.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/modules/__init__.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/modules/loss.py`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/modules/loss.py`

Action taken:
- synced the three module files from the training tree to the `texture-1` inference tree so inference uses the same UNet module definitions as training.

Implication:
- any inference conclusion obtained before this sync is not fully trustworthy, because the trained generator was not being reconstructed exactly at test time.

#### Current highest-confidence diagnosis

The remaining issue is no longer best explained by a small hyperparameter mistake.
The strongest evidence now points to a **task-definition mismatch**:
1. the original baseline solved a target-domain-known task (`Y_GT * mask`), not the current degraded-known task;
2. the x12/stage2/stage3 line changed the main diffusion target to `Y_degraded_full`, which teaches a faded/raw mural target rather than the desired restoration target;
3. until the train/test generator definitions were synced, inference was also structurally inconsistent.

What is ruled out now:
- simple YAML typo / route typo as the primary cause
- old pure white collapse caused only by compose/structure leakage
- BrushNet feature scale alone being the primary cause
- color auxiliary weight alone being the primary cause

### 2026-04-29 cleanup / restore policy for the next line

To avoid inheriting ineffective changes:

- do **not** continue from `x12-stage2` / `x12-stage3` checkpoints
- do **not** use `main_target_domain: raw` for the next clean line
- do **not** rely on color-aux as a required mechanism for the next clean line

Active clean direction:
- main diffusion target should return to dataset `Y_GT`
- condition/mu should stay in the degraded/current-domain branch
- BrushNet stays enabled as a weak auxiliary branch
- `restore_S_guidance` stays off during the first clean verification

Prepared clean config pair:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x13-gtcurrent-weakbrush.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x13-gtcurrent-weakbrush-current-domain.yml`

### 2026-04-29 follow-up fix: texture-1 inference module export mismatch

Issue:
- x13 test startup failed with:
  - `AttributeError: module 'models.modules' has no attribute 'ConditionalUNets'`

Root cause:
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/modules/__init__.py`
  exported only `ConditionalUNet`
- but `texture-1` inference `models/networks.py` expects both:
  - `ConditionalUNetWithBrushNet`
  - `ConditionalUNets` (for `network_Gs`)

Fix:
- export `ConditionalUNets` again in both:
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/models/modules/__init__.py`
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/modules/__init__.py`

This is a startup / module-export bug, not a model-quality diagnosis.

### 2026-04-29 decisive finding: x13 still trains a full-image LUT target because `gt_mode=full`

Hard evidence from code:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/data/mural_inpainting_dataset.py`
  - `_generate_gt(...)` returns `self.color_prior_gen.build_target(degraded_img, mask, mode=mode, feather_radius=7)`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/color_prior_generator.py`
  - `build_target(..., mode="full")` returns `lut_only`
  - `build_target(..., mode="partial")` returns a feathered blend that keeps known pixels in the original image domain

Implication:
- in the mural dataset, `Y_GT` is not an external clean reference image;
- when `gt_mode=full`, `Y_GT` means the entire image is mapped into the LUT target domain;
- therefore x13 (`main_target_domain: gt` + `gt_mode: full`) still trains the main diffusion objective toward a full-image LUT style target.

Why this matters structurally:
- inference uses `condition_known_source: degraded` and `known_area_projection=True`, so known pixels are preserved in the degraded/current domain;
- training with `gt_mode=full` tells the model that known pixels should move toward the LUT domain;
- this is a direct task-definition mismatch, not a small hyperparameter error.

Observed symptom explained by this mismatch:
- persistent bright / pale hole fillings even after white-collapse bugs were mitigated;
- low `white_ratio` but high `final_hole_mean` (~0.96 in x13 test);
- outputs remain closer to the LUT-style bright basin than to the degraded-known continuity the user expects.

Ruled out by this finding:
- x13 failure is not best explained by BrushNet scale alone;
- x13 failure is not best explained by color-aux weight (it is 0 in x13);
- x13 failure is not best explained by a remaining route typo.

New clean direction prepared:
- switch mural `gt_mode` from `full` to `partial` while keeping:
  - `main_target_domain: gt`
  - `condition_mu_domain: degraded`
  - weak BrushNet guidance
  - `restore_S_guidance: false`

Prepared config pair:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x14-gtcurrent-partialbrush.yml`
- `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x14-gtcurrent-partialbrush-current-domain.yml`

### 2026-04-29 x14 / stage3 follow-up: the remaining failure is not a route typo; it is the missing inference-like blank-hole supervision

Hard evidence:
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
  logs this warning whenever `known_only` is active and no infer-like blank-hole branch is enabled:
  - `[X8Guard] known_only removes target/color content from hole during inference; training must enable a real inference-like blank-hole loss.`
- `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/train.py`
  still samples training states with:
  - `timesteps, states = sde.generate_random_states(x0=training_target, mu=condition_mu)`
  which means the hole pixels of training `x_t` still contain forward-process target content.
- inference logs for x14 show the actual reverse chain starts from:
  - `x_init_hole(mean=0.0000...)`
  - `cond_hole(mean=0.0000...)`
  i.e. a true blank-hole / known-only start.

Implication:
- even after `gt_mode=partial`, `condition_known_source=degraded`, `structure_source=prefill`, and `restore_S_guidance=false` were aligned,
  the model is still being trained mostly on target-content hole states but is evaluated from blank-hole states.
- this is a structural train/inference mismatch in the optimization target, not a simple remaining weight tweak.

Additional confirmation:
- x14 training logs show `loss_color_aux = 0`, so the remaining bright attractor is not caused by color auxiliary loss.
- x14 test logs still show `final_hole_mean 鈮?0.97` and large `final_gt_l1 > prior_gt_l1`, even though white-ratio collapse is much lower than the earlier pure-white failures.

Action taken:
- reintroduced a **small, explicit inference-like blank-hole x0 supervision** branch in:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/models/denoising_model.py`
- added new config pair that keeps the x14 task definition but enables the missing blank-hole supervision:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x15-partial-blankhole.yml`
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x15-partial-blankhole-current-domain.yml`
- explicitly pinned the route on both sides to avoid hidden defaults:
  - train: `train.sde_mu_hole_mode: known_only`
  - test: `inference.sde_mu_hole_mode: known_only`
  - test: `inference.expected_train_sde_mu_hole_mode: known_only`

Clean-up / restore note:
- x15 does **not** continue the x12 raw-target line.
- x15 keeps the x14 structural fixes (`gt_mode=partial`, current-domain known input, prefill structure prep, no Mu/MGLC) and adds only the missing blank-hole supervision branch.
- 2026-04-29 / x15 result conclusion:
  - `x15` already proved the missing blank-hole branch was real and is now active:
    - train log has non-zero `loss_infer_x0` / `loss_infer_x0_weighted`
    - test route is clean: `condition_known_source=degraded`, `structure_source=prefill`, `restore_S_guidance=false`, `unexpected=0`
  - Yet `x15` still loads from `ir-sde-brushnet-ft-x14-gtcurrent-partialbrush/models/best_G.pth`, i.e. from the already-bright basin.
  - Therefore `x15` does **not** isolate the task-definition fix from the bad initialization path.
  - Next clean ablation must keep x15 settings unchanged and only switch initialization back to the original baseline trunk.

- 2026-04-30 / x16 clean-init blank-hole line:
  - New train config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x16-cleaninit-blankhole.yml`
  - New test config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x16-cleaninit-blankhole-current-domain.yml`
  - This line changes **only one variable** relative to x15:
    - `pretrain_model_G` is reset from the original baseline trunk
    - no longer inherits `x14` / `x15` bright basin
  - Everything else intentionally stays the same as x15:
    - `gt_mode=partial`
    - `main_target_domain=gt`
    - `condition_mu_domain=degraded`
    - `sde_mu_hole_mode=known_only`
    - blank-hole `infer_x0` supervision enabled
    - weak BrushNet enabled
    - `restore_S_guidance=false`
    - `structure_source=prefill`
  - This is the current active clean line. Older x12/x13/x14/x15 checkpoints should not be used as warm starts for it.

- 2026-04-30 / x16 result conclusion:
  - x16 clean-init **falsified** the remaining 鈥渂ad warm start鈥?hypothesis.
  - Evidence:
    - train log loads the original baseline trunk:
      - `pretrain_model_G=/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
    - x15 blank-hole auxiliary is active from the start:
      - non-zero `loss_infer_x0`
      - non-zero `loss_infer_x0_weighted`
    - test route is clean:
      - `unexpected=0`
      - `condition_known_source=degraded`
      - `structure_source=prefill`
      - `restore_S_guidance=false`
  - Yet x16 still converges to the same bright solution:
    - `final_hole_mean 鈮?0.97`
    - hard samples remain much worse than `prior_gt_l1`
  - Therefore the remaining failure is **not** best explained by:
    - a route typo,
    - a leftover x14/x15 bright checkpoint,
    - BrushNet being on/off,
    - color auxiliary weight.

- Structural finding after x16:
  - The dominant mismatch is still in the **main training state distribution**.
  - `train.py` still forms the main texture loss on:
    - `timesteps, states = sde.generate_random_states(x0=training_target, mu=condition_mu)`
  - `str_utils/sde_utils.py` shows this means:
    - `state_mean = mu_bar(x0, timesteps)`
    - where `mu_bar(x0, t) = mu + (x0 - mu) * exp(-...)`
  - So for low / mid timesteps, hole states still contain strong target leakage.
  - The x15/x16 blank-hole branch is only a small auxiliary correction on top of this:
    - `loss_infer_x0_weighted` is typically `1e-4 ~ 6e-4`
    - while `loss_main` remains `~2e-3 ~ 4e-3`
  - This explains why x15/x16 still collapse into the bright basin even after route cleanup.

- New active clean line after x16: x17 high-t-only main loss
  - New train config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x17-hight-only.yml`
  - New test config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x17-hight-only-current-domain.yml`
  - Code change:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py`
      now supports `train.main_t_min_ratio` / `train.main_t_max_ratio`.
      When set, the **main** texture SDE states are sampled with explicit high timesteps via
      `generate_random_states_texture(...)` instead of the unrestricted `generate_random_states(...)`.
  - x17 keeps the x16 route exactly the same and changes only this structural variable:
    - the dominant main-loss states are now restricted to `t in [0.65T, 1.0T]`
    - this removes low / mid-t target leakage from the main loss instead of only adding a small auxiliary correction.
  - Follow-up fix:
    - the first x17 YAML draft accidentally contained garbled Chinese dataset path literals.
    - fixed active files:
      - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x17-hight-only.yml`
      - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x17-hight-only-current-domain.yml`
    - corrected path segment:
      - `/home/610-wws/Impainting/dataset/瑁佸壀鐨勫浘鐗?...`

- 2026-04-30 / x17 result conclusion:
  - x17 confirmed that the previous diagnosis about the main state distribution was real:
    - training uses `train.main_t_range=[0.65, 1.0]`
    - `stats_timestep_high_ratio=1.0000`
    - `stats_train_state_hole_mean` drops to roughly `0.02 ~ 0.03`, much closer to inference blank-hole states
    - the failure mode changes from near-white collapse (`final_hole_mean ~ 0.97`) to a flatter pale fill (`final_hole_mean ~ 0.87`)
  - However, hard samples still remain clearly worse than `prior_gt_l1`.
  - This means the remaining dominant problem is not route mismatch, warm-start contamination, BrushNet on/off, or color-aux weight.

- Structural finding after x17:
  - The mural training dataset still does not provide a real paired GT target to the diffusion main loss.
  - In `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\data\mural_inpainting_dataset.py`:
    - line ~534 loads a single image:
      - `degraded_img = self._load_image(img_path)`
    - line ~557 builds the observed input from that same image:
      - `observed_degraded = self._build_observed_input(degraded_img, mask)`
    - line ~591 builds `gt` from that same image:
      - `gt = self._generate_gt(degraded_img, mask, current_mode)`
    - line ~484 shows `_generate_gt(...)` is only:
      - `self.color_prior_gen.build_target(degraded_img, mask, mode=mode, feather_radius=7)`
  - Therefore the training main target is still a synthetic target-like image generated from the same source image, not a true paired restoration GT.
  - In contrast, the inference dataset does load a separate evaluation GT from `dataroot_GT`.
  - This is now the primary structural explanation for why x13/x14/x15/x16/x17 can improve the failure mode but still cannot match the real GT on hard holes.

- Actionable consequence:
  - Do not continue stacking more route or loss tweaks on top of `mural_inpainting_dataset.py`.
  - The next meaningful line must first decide whether a real paired training GT exists:
    - if yes: rewrite the mural training dataset to load `degraded_full/current-domain` and `GT_true` separately, and use `GT_true` as the diffusion main target
    - if no: accept that the current synthetic-target training setup cannot supervise the model toward the real test GT, so current expectations must be reduced or the supervision source must be redesigned

## 2026-04-30 x18 diagnosis -> x19 refine-mask line

- x18 is the first line that clearly improves coarse color on hard samples under true paired supervision:
  - `000098_center final_gt_l1=0.101340` vs `prior_gt_l1=0.204685`
  - `000098_left final_gt_l1=0.135976` vs `prior_gt_l1=0.217610`
- Remaining visible failure shifts from global white collapse to:
  1. white halo / border around the hole;
  2. weak texture.
- Direct log evidence for the halo root cause:
  - `cond_known(...white=0.0207)` on center and `0.0280` on left in x18 test logs.
  - This means the current-domain `mask_merge` image still contains near-white pixels inside the region treated as known.
  - `raw_gt_l1` is slightly lower than `final_gt_l1`, so final compose / known-area preservation is introducing some of the visible border.
- Conclusion:
  - The next structural fix should refine the binary hole mask using observed white boundary pixels, consistently in both train and test.
  - This is not another generic parameter tweak; it addresses a concrete train/test data mismatch at the hole boundary.
- New active line: `x19-refinemask`
  - train dataset: `mural_paired_inpainting_dataset.py` now supports `refine_mask_from_observed_white`.
  - test dataset: `mural_inference_dataset.py` now supports the same refinement.
  - x19 configs enable:
    - `refine_mask_from_observed_white: true`
    - `mask_white_refine_threshold: 0.95`
    - `mask_white_refine_dilate: 6`
    - `mask_white_refine_expand: 0`
  - x19 warm-starts from x18 best on purpose because x18 already established the correct paired supervision and coarse-color recovery.
- Reminder from user confirmed and active test configs corrected: structure checkpoint path must remain
  `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  (with the trailing `s` in `StrDiffusion+e00s`).
  Fixed in active test configs:
  - `ir-sde-brushnet-x18-pairedstem-hight-current-domain.yml`
  - `ir-sde-brushnet-x19-refinemask-current-domain.yml`
  - `ir-sde-brushnet-x20-strongmask-cleancompose-current-domain.yml`


## 2026-05-01 x20 stable baseline -> x21 weak texture line

- x20 trained with paired supervision + blank-hole + refined mask + clean compose is now the first stable coarse-color baseline.
- Evidence from `test_ir-sde-brushnet-x20-strongmask-cleancompose-current-domain_260501-111501.log`:
  - white-border failure is effectively solved (`final_white_ratio_hole=0` on the tracked samples);
  - final compose no longer degrades the raw prediction (`raw_gt_l1 == final_gt_l1` on tracked samples);
  - coarse color / region reconstruction is much better than prior:
    - center `0.092132` vs prior `0.196733`
    - left `0.090027` vs prior `0.203064`
    - top `0.107236` vs prior `0.206919`
- Remaining issue after x20 is no longer route or boundary correctness. It is mostly:
  1. texture is still flat / washed;
  2. fine details are missing even when coarse colors are acceptable.
- Current interpretation:
  - x20 is a valid base model for the current-domain task;
  - the next step should preserve x20's paired supervision, blank-hole semantics, refined mask, and clean compose;
  - only a weak texture refinement branch should be added.
- New active line: `x21-weaktexture`
  - train config:
    - `D:\code\kyihua\Impainting\StrDiffusion+e00	rain-3	exture\config\inpainting\options	rain\ir-sde-brushnet-ft-x21-weaktexture.yml`
  - test config:
    - `D:\code\kyihua\Impainting\StrDiffusion	est	exture-1\config\inpainting\options	est\ir-sde-brushnet-x21-weaktexture-current-domain.yml`
- x21 keeps the x20 route unchanged and only adds a conservative texture core:
  - warm start from x20 best
  - `texture_core.enabled: true`
  - `insert_mid: true`
  - `insert_dec: false`
  - `backend: sem_lite`
  - `use_mask_gate: true`
  - `gate_hidden: 8`
  - `boundary_width: 2`
  - `zero_init_last: true`
  - short pretrained freeze (`freeze_pretrained_until_iter: 800`) so the new zero-init texture branch learns first without disturbing x20 coarse colors
  - smaller trunk lr and larger new-module lr:
    - `lr_G=2e-7`
    - `lr_new=3e-6`
- The intent of x21 is not to re-solve color or boundary issues. It is specifically to test whether a weak, zero-init residual texture refinement path can add detail on top of the now-stable x20 base.

- 2026-05-01 x21 follow-up fix: the first x21 test YAML accidentally kept the old x18 checkpoint path. Active test config now points to `/home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x21-weaktexture/models/best_G.pth`.

## 2026-05-01 x21 result -> x22 explicit high-frequency texture supervision

- x21 is structurally healthy but not sufficient to recover texture.
- Evidence from `train_ir-sde-brushnet-ft-x21-weaktexture_260501-132307.log`:
  - `loaded 246/272 ... missing=26, unexpected=0`; the only new parameters are the `mglc_mid` texture weights.
  - `freeze_pretrained_until_iter=800` behaved as expected.
  - the training stays numerically stable, but no new instability or strong texture-learning signal appears beyond the existing blank-hole loss.
- Evidence from `test_ir-sde-brushnet-x21-weaktexture-current-domain_260501-224041.log`:
  - `texture_core.enabled(config/runtime)=True/True`
  - `restore_S_guidance=False`
  - `final_white_ratio_hole=0.000000`
  - tracked-sample metrics stay almost identical to x20:
    - bottom `0.065632` vs x20 `0.066680`
    - center `0.092090` vs x20 `0.092132`
    - left `0.089848` vs x20 `0.090027`
    - right `0.088087` vs x20 `0.086613`
    - top `0.108010` vs x20 `0.107236`
- Conclusion:
  - x21 proves that simply enabling a weak zero-init texture branch is not enough to create meaningful hole texture on top of x20.
  - This is no longer a route bug or a mask/compose bug; it is a missing supervision signal for texture detail.
- New active line: `x22-hftexture`
  - Keep x20/x21's already validated route:
    - paired current-domain supervision
    - refined mask
    - clean compose
    - `known_only`
    - high-t main loss
    - blank-hole `infer_x0`
  - Add explicit hole-only high-frequency supervision on the inference-like blank-hole prediction:
    - compute a luminance high-pass of `x0_hat_infer`
    - compute a luminance high-pass of `training_target`
    - apply hole-only L1 with a small weight
  - Active train config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x22-hftexture.yml`
  - Active test config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x22-hftexture-current-domain.yml`
  - Initial settings:
    - warm start from x21 best
    - `texture_hf_loss_weight: 0.02`
    - `texture_hf_loss_start_iter: 0`
    - `texture_hf_blur_kernel: 11`
- YAML path fix recorded:
  - the first x22 YAML drafts contained garbled Chinese dataset roots; active x22 train/test YAMLs now use the correct `/home/610-wws/Impainting/dataset/瑁佸壀鐨勫浘鐗?...` paths.

## 2026-05-02 x22 result -> x23 direct trunk high-frequency supervision

- x22 confirms a key distinction:
  - the current route is stable;
  - white-border / compose failures are no longer the main issue;
  - but explicit infer-branch HF loss plus weak texture_core still does not create convincing texture.
- Evidence from `train_ir-sde-brushnet-ft-x22-hftexture_260502-005345.log`:
  - `loss_texture_hf` / `loss_texture_hf_weighted` are active and non-zero;
  - blank-hole `loss_infer_x0` remains active;
  - training is numerically stable.
- Evidence from `test_ir-sde-brushnet-x22-hftexture-current-domain_260502-103757.log`:
  - `texture_core.enabled(config/runtime)=True/True`
  - `restore_S_guidance=False`
  - `final_white_ratio_hole=0.000000`
  - metrics remain very close to x21/x20 on the tracked samples; improvements are too small and inconsistent to count as real texture recovery.
- Interpretation:
  - original StrDiffusion texture came primarily from the **main diffusion trunk**;
  - in x20/x21/x22, we deliberately protected the trunk to stabilize current-domain supervision and pushed most 鈥渢exture pressure鈥?onto auxiliary paths (`texture_core`, infer-branch HF loss);
  - this keeps coarse color stable, but it also explains why visible texture does not come back.
- New active line: `x23-trunkhftexture`
  - Keep x20鈥檚 validated current-domain route:
    - paired supervision
    - refined mask
    - clean compose
    - `known_only`
    - high-t main loss
    - blank-hole `infer_x0`
  - Turn **off** `texture_core` again.
  - Add **direct hole-only HF supervision on the main branch prediction** (`x0_hat_main`) so the base diffusion trunk itself relearns texture under the stable route.
  - Active train config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x23-trunkhftexture.yml`
  - Active test config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x23-trunkhftexture-current-domain.yml`
  - Initial settings:
    - warm start from x20 best
    - `texture_core.enabled: false`
    - `texture_hf_source: main`
    - `texture_hf_loss_weight: 0.01`
    - `texture_hf_blur_kernel: 11`
    - `lr_G: 1e-6`
    - `lr_new: 1e-6`
    - `freeze_pretrained_until_iter: 0`
- x23-specific fixes recorded:
  - the first x23 YAML drafts inherited garbled Chinese dataset roots; active x23 train/test YAMLs have been corrected to `/home/610-wws/Impainting/dataset/瑁佸壀鐨勫浘鐗?...`
  - active x23 test YAML keeps the correct structure checkpoint path:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`


## 2026-05-02 x24 mixed-t trunk texture line
- Observation from x20/x21/x22/x23: route is stable and white-border issue is largely solved, but texture remains flat.
- Evidence: x23 enabled direct trunk HF supervision, yet `stats_timestep_high_ratio` stayed at `1.0000`, meaning the main trunk still only saw high-t states.
- Conclusion: the remaining bottleneck is structural. High-t-only main supervision preserves coarse color/shape but suppresses the original trunk's mid/low-t texture-learning regime.
- x24 fix: keep the stable high-t main loss and high-t blank-hole branch, but add a second inference-like blank-hole branch at mid-t (`infer_x0_mid_*`) plus an optional mid-t HF loss (`texture_hf_mid_*`) so the trunk can relearn local continuity and texture.
- Warm start for x24 should come from x20 stable base, not x23, because x23 did not provide stable texture gains.


## 2026-05-02 x24 code/config sync check + first log result

- Sync check for the active x24 line is complete:
  - train code: `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
    - added config parsing + logging for `infer_x0_mid_*`
    - added actual mid-t blank-hole branch execution through `_run_blankhole_x0_branch(...)`
    - added actual mid-t HF trunk loss on `x0_hat_mid_for_texture`
    - added `loss_infer_x0_mid`, `loss_infer_x0_mid_weighted`, `loss_texture_hf_mid`, `loss_texture_hf_mid_weighted` to `log_dict`
  - train loop / TB sync: `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py`
    - still keeps `main_t_range=[0.65, 1.0]` on the dominant main branch
    - tensorboard scalar map now includes `train/loss_infer_x0_mid` and `train/loss_texture_hf_mid`
  - active train YAML:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x24-mixedt-trunktexture.yml`
  - active test YAML:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x24-mixedt-trunktexture-current-domain.yml`
  - route-critical items are aligned on both sides:
    - paired current-domain supervision
    - refined mask
    - clean compose
    - `sde_mu_hole_mode=known_only`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=false`
    - structure checkpoint path remains `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- Evidence from training log `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x24-mixedt-trunktexture_260502-183242.log`:
  - warm start is correct: x24 loads x20 best with `loaded 246/246`, `missing=0`, `unexpected=0`
  - new x24 losses are really active from the start:
    - iter 20: `loss_infer_x0_mid=4.5537e-01`, `loss_infer_x0_mid_weighted=6.8306e-04`
    - iter 20: `loss_texture_hf_mid=4.4365e-02`, `loss_texture_hf_mid_weighted=2.2182e-04`
    - later they stay non-zero (e.g. iter 2660 still has `loss_infer_x0_mid=7.7031e-02`, `loss_texture_hf_mid=1.5153e-02`)
  - however, the dominant main branch distribution is unchanged:
    - `train.main_t_range=[0.65, 1.0]`
    - training log keeps reporting `stats_timestep_high_ratio=1.0000`
  - this means x24 adds a real mid-t auxiliary branch, but it does **not** change the main diffusion branch away from the x17~x23 high-t-only regime.

- Evidence from test log `C:\Users\admin\Desktop\test_ir-sde-brushnet-x24-mixedt-trunktexture-current-domain_260502-205825.log`:
  - load/route are clean:
    - `loaded 246/246` for G, `151/151` for structure Gs, `38/38` for D
    - `brushnet.enabled(config/runtime)=True/True`
    - `texture_core.enabled(config/runtime)=False/False`
    - `restore_S_guidance=False`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `deterministic_reverse=True`, `discriminator_guidance=False`
    - blank-hole start is still correct: `x_init_hole(mean=0.0000 ... white=0.0000)`
  - but the first x24 inference run is **not** a stable improvement over x20. The old white failure reappears on hard samples even though the route is correct.
  - direct hard-sample evidence:
    - `000098_bottom`: `final_gt_l1=0.156462` vs `prior_gt_l1=0.077797`, `final_white_ratio_hole=0.661004`
    - `000098_right`: `final_gt_l1=0.211414` vs `prior_gt_l1=0.096517`, `final_white_ratio_hole=0.676488`
    - `000098_center`: still better than prior in L1 (`0.133461` vs `0.196733`) but already shows large white ratio `0.352206`
  - some easier / moderate samples still improve:
    - `000098_left`: `0.082424` vs `0.203064`
    - `000098_top`: `0.095890` vs `0.206919`
    - `000180_left`: `0.069776` vs `0.095763`
  - but this is not enough to call x24 stable, because the failure mode on the harder holes is qualitatively the old one again.
  - additional confidence-slice evidence on failed samples shows the whitening is strongest in the low-confidence hole subset, while `prior_hole` / `lut_hole` themselves are not white. So the regression happens during the reverse trajectory, not from a white input prior.

- Important note about the available x24 test log:
  - this log is partial; it stops during `000180_right`, so it is not a full 250-image sweep.
  - still, the route checks are decisive and the early hard-sample failures are strong enough to reject x24 as a new stable base.

- Current x24 conclusion:
  - x24 code/config sync is correct.
  - x24 did activate the intended mid-t auxiliary supervision.
  - but x24 did **not** solve the dominant structural issue, because the main branch still trains entirely on high-t states.
  - first inference evidence shows regression back toward the old white-mask failure on hard samples.
  - Therefore x24 should be treated as a failed auxiliary-only fix, not as the next stable line after x20.


## 2026-05-02 x24 white failure: not best explained by "just train longer"

- User question addressed against:
  - train log `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x24-mixedt-trunktexture_260502-183242.log`
  - test log `C:\Users\admin\Desktop\test_ir-sde-brushnet-x24-mixedt-trunktexture-current-domain_260502-205825.log`
- Important nuance:
  - x24 is **not fully finished training yet** in the available log; it reaches only about `iter 2720 / 6000`, so we cannot call it fully converged in the optimization sense.
  - But the visible white failure on hard samples is **not best explained** as a simple remaining-convergence issue.

- Reasons this is not just "undertrained":
  1. route / input side is already correct:
     - blank-hole init is correct: `x_init_hole(mean=0.0000 ... white=0.0000)`
     - `condition_known_source=degraded`, `structure_source=prefill`, `restore_S_guidance=false`
     - `deterministic_reverse=true`, `discriminator_guidance=false`
  2. whiteness is generated by the model trajectory itself, not by final compose:
     - failed samples have `raw_hole == final_hole`
     - failed samples have `raw_gt_l1 == final_gt_l1`
     - so compose is not introducing the white region
  3. the prior inputs are not white on the failed samples:
     - e.g. `000098_bottom/right` both have non-white `prior_hole` / `lut_hole`
     - yet `raw_hole/final_hole` become heavily white
     - therefore whitening happens during reverse prediction, not because the input prior is already white
  4. the failure is selective and confidence-structured, not a global "model still blurry everywhere" symptom:
     - some samples improve strongly
     - some hard samples regress catastrophically to white
     - confidence-slice logs show low-confidence hole subsets whiten more severely
  5. the core structural issue remains in training:
     - `stats_timestep_high_ratio=1.0000` throughout x24 training
     - so the **main branch** still only trains on high-t states
     - x24 mid-t branch is real, but still auxiliary; it does not replace the dominant main-branch state distribution

- Additional practical note:
  - test YAML loads `best_G.pth`, i.e. the train-side `best-texture` checkpoint.
  - In current training code, `best_G` is selected by EMA-smoothed `loss_main`, **not** by white-ratio, hard-sample GT quality, or inference-time tracked metrics.
  - So even if later checkpoints look better visually on hard holes, the current best-checkpoint rule does not specifically optimize for that failure mode.

- Current interpretation:
  - continuing x24 training may still move metrics somewhat, because the run is only at `2720/6000`.
  - however, it is **not justified** to expect that simply training longer on the same x24 objective will reliably remove the hard-sample white failure.
  - the white issue is better explained by the remaining structural mismatch: mid-t supervision is still auxiliary while the dominant trunk remains high-t-only.


## 2026-05-02 x25 main-distribution fix line (no network-structure change)

- User constraint reinforced:
  - time is limited; prefer the shortest path toward the final structural fix
  - avoid any train/inference network mismatch
  - every change must be tracked here
  - original StrDiffusion reference tree consulted:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\train\texture\config\inpainting`

- x25 design decision:
  - Do **not** change the network structure.
  - Do **not** add another auxiliary branch.
  - Change the **main branch training-state distribution** directly.
- Reason:
  - x24 already proved that mid-t auxiliary supervision can be wired correctly, but hard-sample whitening still reappears because the dominant main branch remains `high-t-only`.
  - Since time is limited, the lowest-risk high-value move is to keep train/test architecture identical and only change what the main loss actually sees.

- Core x25 idea:
  - keep most of the main batch on the stable x17/x20 high-t forward states
  - replace a controlled subset of the main batch with **mid-t hybrid blank-hole states**:
    - known area: still normal forward state from `(training_target, condition_mu, t_mid)`
    - hole area: switched to inference-like blank-hole state `condition_mu + sigma(t_mid) * noise`
  - this makes the **main loss itself** see more inference-like hole states, instead of relying on a weak side branch

- Why this is safer than changing the model structure:
  - the same `ConditionalUNetWithBrushNet` is used for both train and test
  - no checkpoint-key or runtime-graph divergence is introduced
  - test config only needs a new checkpoint path; inference route semantics remain unchanged

- Code changes:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py`
    - added `_sample_timestep_range(...)`
    - added `_build_main_states_hybrid_mid_blank_hole(...)`
    - added `train.main_state_mode`
    - when `main_state_mode=hybrid_mid_blank_hole`, the main batch is built as:
      - base high-t forward states for the whole batch
      - then overwrite a subset with x25 hybrid mid-t blank-hole states
    - added per-step debug stats passed to the model:
      - `stats_main_state_mode_hybrid_mid_blank`
      - `stats_main_mid_blank_ratio`
      - `stats_main_mid_blank_count`
      - `stats_main_mid_blank_t_mean`
      - `stats_main_high_forward_ratio`
      - `stats_main_high_forward_count`
    - TB scalar map now includes:
      - `stats/main_mid_blank_ratio`
      - `stats/main_mid_blank_t_mean`
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
    - `optimize_parameters(...)` now logs the x25 `main_state_debug` diagnostics when present

- New active x25 configs:
  - train:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x25-mainmixed-blankhole.yml`
  - test:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x25-mainmixed-blankhole-current-domain.yml`

- x25 config policy:
  - warm start still from x20 stable base
  - inference route unchanged from the stable current-domain line:
    - paired current-domain supervision
    - refined mask
    - clean compose
    - `known_only`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=false`
    - structure checkpoint path remains `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  - x24 auxiliary mid-t branches are intentionally turned off in x25:
    - `infer_x0_mid_loss_weight: 0.0`
    - `texture_hf_mid_loss_weight: 0.0`
  - x25 changes only the dominant main branch distribution, not the model graph

- Initial x25 training settings:
  - `main_state_mode: hybrid_mid_blank_hole`
  - `main_t_min_ratio: 0.65`
  - `main_t_max_ratio: 1.0`
  - `main_mid_blank_ratio: 0.25`
  - `main_mid_t_min_ratio: 0.20`
  - `main_mid_t_max_ratio: 0.55`
  - keep high-t inference-like blank-hole branch from the stable line:
    - `infer_x0_loss_weight: 0.002`

- Expected diagnostic signature once x25 starts training:
  - `stats_timestep_high_ratio` should drop below `1.0000`
  - `stats_main_mid_blank_ratio` should be around `0.25`
  - `stats_main_mid_blank_t_mean` should land near the configured mid-t band
  - if the hypothesis is right, hard-sample whitening should reduce **without** needing a network-route change

- Static validation completed after the code edit:
  - `train.py` passes `py_compile`
  - `models/denoising_model.py` passes `py_compile`
  - x25 train/test YAMLs point to the intended experiment names and checkpoint paths

## 2026-05-03 x25 first result: actual runtime mismatch + stronger white collapse

- User-provided evidence checked against:
  - train log: `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x25-mainmixed-blankhole_260502-221736.log`
  - test log: `C:\Users\admin\Desktop\test_ir-sde-brushnet-x25-mainmixed-blankhole-current-domain_260503-101230.log`
- High-confidence conclusion: the tested x25 run is **worse than x24/x20 on the hard white-failure samples**.

- First critical finding: the **actual executed x25 training run was not the intended warm start**.
  - Current local x25 YAML in workspace points to x20 stable base:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x25-mainmixed-blankhole.yml`
    - `path.pretrain_model_G = /home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x20-strongmask-cleancompose/models/best_G.pth`
  - But the actual x25 train log says the run loaded:
    - `pretrain_model_G: /home/610-wws/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/log/ir-sde-brushnet-ft-x24-mixedt-trunktexture/models/best_total_G.pth`
    - `[LoadCheck] loaded 246/246 tensors ... missing=0 unexpected=0`
  - Therefore the tested x25 result is **not a clean x25-from-x20 experiment**. It is effectively an **x25-from-x24-best_total** run.
  - This matters because x24 already had the hard-sample white-attractor issue. x25 then changed the dominant main-branch state distribution on top of that unstable base.

- Second critical finding: x25 main-distribution change was real and strong, not a fake/no-op change.
  - Train log keeps showing:
    - `stats_timestep_high_ratio=0.7500`
    - `stats_main_mid_blank_ratio=0.2500`
    - `stats_main_state_mode_hybrid_mid_blank=1.0000`
  - So this x25 run truly replaced 25% of the main batch with the hybrid mid-t blank-hole states.
  - That means x25 is **not** failing because the code path was disconnected; it is failing because this specific main-distribution move is too aggressive / destabilizing in the actually executed setup.

- Test-route check remains clean in x25 inference:
  - `brushnet.enabled(config/runtime)=True/True`
  - `texture_core.enabled(config/runtime)=False/False`
  - `restore_S_guidance=False`
  - `condition_known_source=degraded`
  - `structure_source=prefill`
  - `sde_mu_hole_mode=known_only`
  - `deterministic_reverse=True`
  - structure checkpoint still correctly uses:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
- So the regression is again **not** from route mismatch or compose mismatch; it is generated during the reverse trajectory.

- Hard-sample regression vs prior is severe in x25:
  - `000098_bottom`:
    - `final_gt_l1=0.342377` vs `prior_gt_l1=0.077797`
    - `final_white_ratio_hole=0.934181`
  - `000098_right`:
    - `final_gt_l1=0.417303` vs `prior_gt_l1=0.096517`
    - `final_white_ratio_hole=0.931165`
  - `000098_center`:
    - `final_gt_l1=0.208188` vs `prior_gt_l1=0.196733`
    - `final_white_ratio_hole=0.451539`
  - Even the easier improving samples got worse than x24:
    - `000098_left`: `final_gt_l1=0.103389` (x24 had `0.082424`)
    - `000098_top`: `final_gt_l1=0.125634` (x24 had `0.095890`)
- Direct visual/log interpretation:
  - x25 re-enters the old white attractor **earlier and harder**.
  - Example `000098_bottom` trajectory:
    - `t=100` hole mean `0.7965`, white `0.0000`
    - `t=40` hole mean `1.0992`, white `0.8258`
    - `t=20` hole mean `1.1615`, white `0.8942`
    - `t=4` hole mean `1.1861`, white `0.9209`
  - Example `000098_right` trajectory:
    - `t=100` hole mean `0.8062`, white `0.0000`
    - `t=40` hole mean `1.1343`, white `0.8714`
    - `t=20` hole mean `1.2059`, white `0.9131`
    - `t=4` hole mean `1.2304`, white `0.9250`
  - As in x24, `raw_hole == final_hole` on failed cases, so compose is not the source of the whitening.

- Current interpretation of x25:
  - The first x25 result should be marked as a **failed run**.
  - But the failure has two layers:
    1. experiment hygiene failure: the actual runtime warm start drifted to x24 `best_total_G` instead of the intended x20 `best_G`
    2. optimization failure: even in that actual run, the 25% always-on main-batch swap is too strong and worsens the late reverse collapse on hard samples
  - Therefore we should **not** keep the x25 setting unchanged.


## 2026-05-03 x26 direction: guarded x20 warm start + small ramped main mix + keep x24 mid auxiliaries

- Design goal after x25:
  - keep the final-direction idea (main trunk must eventually see some mid-t / inference-like hole states)
  - but remove the two concrete x25 problems:
    1. no more silent warm-start drift
    2. no more 25% always-on hard swap of the main batch from iter 0

- Code-level safety improvement added in:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py`
- New safeguards / scheduler:
  1. `path.expected_pretrain_model_G`
     - if set, training now hard-checks that runtime `path.pretrain_model_G` exactly matches it
     - if not, training raises `[ConfigGuard]` immediately instead of silently starting from the wrong checkpoint
  2. `train.main_mid_blank_ratio_start`
  3. `train.main_mid_blank_ratio_warmup_iter`
     - main hybrid ratio can now ramp from a safe starting value to the target ratio over time
     - this lets x20 stability dominate the early phase instead of shocking the trunk from iter 0
  4. extra logging:
     - `stats_main_mid_blank_ratio_requested`
     - existing `stats_main_mid_blank_ratio` remains the actual per-batch realized ratio after rounding by batch size

- Added new train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x26-ramped-smallmainmix.yml`
- Added new test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x26-ramped-smallmainmix-current-domain.yml`

- x26 policy:
  - warm start is forced back to x20 stable base
  - x24 mid auxiliary branch is kept (because x25 main-only replacement was worse):
    - `infer_x0_mid_loss_weight: 0.0015`
    - `texture_hf_mid_loss_weight: 0.005`
  - x25-style main-distribution shift is retained only in a **small / ramped** form:
    - `main_state_mode: hybrid_mid_blank_hole`
    - `main_mid_blank_ratio_start: 0.0`
    - `main_mid_blank_ratio: 0.10`
    - `main_mid_blank_ratio_warmup_iter: 1200`
    - `main_mid_t_min_ratio: 0.35`
    - `main_mid_t_max_ratio: 0.60`
  - inference route remains unchanged and stable:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=false`
    - structure checkpoint path remains `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- Expected x26 diagnostics:
  - first verify the guard works:
    - runtime log must show x20 `best_G.pth` as pretrain
    - if a stale remote YAML still points somewhere else, training should now fail immediately instead of wasting a run
  - during training:
    - `stats_main_mid_blank_ratio_requested` should ramp from `0.0` toward `0.10`
    - `stats_main_mid_blank_ratio` should stay small early and only later become non-zero on more batches
    - `stats_timestep_high_ratio` should remain much closer to x20/x24 than x25 did
  - during test:
    - the first hard samples should be checked again in this order:
      - `000098_bottom`
      - `000098_right`
      - `000098_center`
    - if x26 is on the right track, these should stop exploding to `~0.93` white ratio at hole level

- Static validation after the new code edit:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py` passes `py_compile`
- Also added `expected_pretrain_model_G` to the local x24/x25 train YAMLs so those lines will fail fast too if the runtime warm start drifts again.

## 2026-05-03 x26 result: route / warm-start corrected, white reduced, but split failure remains (white on hard low-confidence holes, warm/pink hue on stable holes)

- Evidence checked against:
  - train log: `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x26-ramped-smallmainmix_260503-103112.log`
  - test log: `C:\Users\admin\Desktop\test_ir-sde-brushnet-x26-ramped-smallmainmix-current-domain_260503-134440.log`
- High-level outcome:
  - x26 is **better than x25** and directionally correct.
  - It fixed the x25 runtime hygiene issue and reduced the catastrophic white collapse.
  - But the failure mode is now split into two distinct classes:
    1. some hard samples still show late-stage white overshoot
    2. some non-white samples are visually acceptable in structure but keep a warm/pink hue bias

- x26 training / route hygiene is correct this time:
  - warm start guard succeeded; actual runtime log shows:
    - `pretrain_model_G = /home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x20-strongmask-cleancompose/models/best_G.pth`
    - `expected_pretrain_model_G = ...same path...`
  - train-side x26 losses are all active as intended:
    - high-t blank-hole branch enabled
    - mid-t blank-hole branch enabled
    - mid-t HF trunk loss enabled
  - inference route remains clean and unchanged:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=false`
    - structure checkpoint stays `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- x26 main-mix scheduler behavior from the train log:
  - early stage behaves conservatively as designed:
    - iter 20: `stats_main_mid_blank_ratio_requested=0.001667`, actual `stats_main_mid_blank_ratio=0.0000`
    - iter 40: requested `0.003333`, actual `0.0000`
    - iter 100: requested `0.008333`, actual `0.0000`
  - around the 1000-iter checkpoint used for best selection, x26 has only **1 / 16** hybrid-mid sample in the main batch on many steps:
    - e.g. iter 900: requested `0.075`, actual `0.0625`
  - after warmup, actual full-ratio operation is batch-rounded to **2 / 16 = 0.125**:
    - e.g. later logs around iter 4320 show requested `0.10`, actual `0.125`

- Important checkpoint-selection consequence:
  - x26 test loads `best_G.pth`
  - x26 best checkpointing starts at iter `1000`
  - train log shows many `[best-texture]` / `[best-total]` updates immediately after iter `1000` while the main hybrid ratio is still below the final intended regime
  - so the tested x26 `best_G` is still relatively early in the scheduled main-mix rollout, i.e. before the final x26 regime has fully dominated

- White-failure status in x26: clearly improved vs x25, but not solved.
  - hard examples from the same `000098_*` group:
    - `000098_bottom`:
      - x25: `final_white_ratio_hole=0.934181`, `final_gt_l1=0.342377`
      - x26: `final_white_ratio_hole=0.438620`, `final_gt_l1=0.096073`
      - interpretation: huge recovery, but still a real white overshoot remains in the hard low-confidence subset
    - `000098_center`:
      - x25: `final_white_ratio_hole=0.451539`, `final_gt_l1=0.208188`
      - x26: `final_white_ratio_hole=0.231778`, `final_gt_l1=0.105230`
      - interpretation: same pattern, much better but not clean
    - `000098_right`:
      - x25: `final_white_ratio_hole=0.931165`, `final_gt_l1=0.417303`
      - x26: `final_white_ratio_hole=0.575777`, `final_gt_l1=0.166833`
      - interpretation: still the hardest one among the first group
  - confidence-slice evidence still points to the old mechanism:
    - `000098_bottom`: low-confidence subregion `final_low white=0.7575` vs high-confidence `0.1976`
    - `000098_right`: low-confidence subregion `final_low white=0.8839` vs high-confidence `0.3393`
  - conclusion: residual whitening is still concentrated in the low-reliability hole subset during the late reverse trajectory, not from input prior whiteness and not from compose

- Stable non-white samples in x26 are now quite good:
  - `000098_left`: `final_gt_l1=0.082415`, `final_white_ratio_hole=0.000000`
  - `000098_top`: `final_gt_l1=0.095655`, `final_white_ratio_hole=0.000000`
  - `000180_bottom`: `final_gt_l1=0.072303`, `final_white_ratio_hole=0.000000`
- These samples show x26 is not globally broken; the trunk can stay stable when the hole is easier or reliability is more favorable.

- New observation from user screenshots + log evidence: some samples are no longer white, but remain visually warm / pink.
  - Example `000257_bottom`:
    - `final_gt_l1=0.090196` vs `prior_gt_l1=0.114658` vs `lut_gt_l1=0.124258`
    - `final_white_ratio_hole=0.027040`
    - `confidence_hole_mean=0.418768`
    - `rawprior_to_safeprior_hole=0.050748`
    - interpretation: this is **not** a white-collapse sample. Trajectory stays stable; the remaining issue is hue bias / warm cast.
  - Example `000348_right`:
    - `final_gt_l1=0.122570` vs `prior_gt_l1=0.160261` vs `lut_gt_l1=0.161100`
    - `final_white_ratio_hole=0.000000`
    - `confidence_hole_mean=0.465463`
    - `rawprior_to_safeprior_hole=0.055634`
    - interpretation: again not a white-collapse case. The final image is better than prior/LUT in GT error, but visually still inherits a warm/pink chroma bias.

- Current interpretation of the pink samples:
  - they are **not** generated by the same failure mode as the white ones
  - their trajectories stay stable and non-white all the way to `t=4`
  - the remaining problem is that the final prediction stays too close to the warm color manifold induced by the color-prior / LUT path, while the diffusion objective alone does not fully neutralize that low-frequency hue bias
  - evidence for this interpretation:
    - pink examples have `final_white_ratio_hole ~ 0`
    - but `final_prior_l1` / `final_lut_l1` remain fairly small, indicating the result still sits near the prior/LUT chroma manifold


## 2026-05-03 x27 direction: keep x26 anti-white route, add weak GT low-frequency color anchor, and delay best-checkpoint selection

- Goal after x26:
  - preserve x26's anti-white gains
  - reduce warm/pink hue bias on the already-stable samples
  - avoid selecting `best_G` too early, before the scheduled main-mix regime is fully active

- Minimal code change added in:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
- New training option:
  - `train.color_aux_target_domain: lut | gt`
- Behavior:
  - existing x12 `color_aux_loss_weight` path is now configurable
  - `lut` keeps the old behavior (blurred x0 vs `training_target_lut`)
  - `gt` changes it to a weak hole-only low-frequency color anchor against blurred paired GT (`training_target`)
- Reason:
  - x26 showed that some samples are structurally fine and no longer white, but still too warm/pink
  - this is exactly the regime where a weak GT low-frequency color anchor is appropriate; it addresses hue bias without changing the inference graph or route

- Added new train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x27-ramped-colorfix.yml`
- Added new test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x27-ramped-colorfix-current-domain.yml`

- x27 policy:
  - warm start still strictly from x20 stable base (`expected_pretrain_model_G` guard kept)
  - keep x26 anti-white route unchanged:
    - paired current-domain
    - refined mask
    - clean compose
    - `known_only`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=false`
  - keep x26 mid-t auxiliaries:
    - `infer_x0_mid_loss_weight: 0.0015`
    - `texture_hf_mid_loss_weight: 0.005`
  - keep ramped small main-mix:
    - `main_mid_blank_ratio_start: 0.0`
    - `main_mid_blank_ratio: 0.10`
    - `main_mid_blank_ratio_warmup_iter: 1200`
  - new color-fix term:
    - `color_aux_loss_weight: 0.01`
    - `color_aux_target_domain: gt`
  - delay best checkpoint start:
    - `best_save_start_iter: 1500`
    - this prevents `best_G` from being captured too early, before x26/x27 scheduled main-mix has fully entered its intended operating regime

- Expected x27 diagnostic signature:
  1. train log should still confirm x20 warm start exactly
  2. `stats_main_mid_blank_ratio_requested` should ramp as before
  3. first selected `best_G` should now occur only after iter `1500`
  4. if the hypothesis is correct:
     - white-heavy hard samples should stay at least as good as x26
     - stable-but-pink samples like `000257_bottom` / `000348_right` should move closer to GT in low-frequency chroma while keeping `final_white_ratio_hole` near zero

- Static validation after x27 code/config edits:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py` passes `py_compile`
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\train.py` passes `py_compile`


## 2026-05-03 x26 checkpoint comparison: `best_G` vs `best_total_G` vs `6000_G`

- Compared logs:
  - x26 earlier `best_G` probe (already recorded above)
  - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x26-ramped-smallmainmix-current-domain_260503-142825.log`
    - runtime `pretrain_model_G=/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x26-ramped-smallmainmix/models/6000_G.pth`
  - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x26-ramped-smallmainmix-current-domain_260503-143254.log`
    - runtime `pretrain_model_G=/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x26-ramped-smallmainmix/models/best_total_G.pth`
- Both new tests still keep the correct stable inference route:
  - `condition_known_source=degraded`
  - `structure_source=prefill`
  - `restore_S_guidance=False`
  - structure checkpoint path remains `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- Key result: `best_total_G` is effectively the same as `best_G`, while `6000_G` is clearly worse on the hard white-failure probes.

- Hard-sample comparison on the first `000098_*` probe set:
  - `000098_bottom`
    - x26 `best_G`: `final_gt_l1=0.096073`, `final_white_ratio_hole=0.438620`
    - x26 `best_total_G`: `final_gt_l1=0.097280`, `final_white_ratio_hole=0.444682`
    - x26 `6000_G`: `final_gt_l1=0.179629`, `final_white_ratio_hole=0.722810`
    - interpretation: `best_total_G ~= best_G`; `6000_G` falls back toward the old white basin
  - `000098_center`
    - x26 `best_G`: `final_gt_l1=0.105230`, `final_white_ratio_hole=0.231778`
    - x26 `best_total_G`: `final_gt_l1=0.107294`, `final_white_ratio_hole=0.231740`
    - x26 `6000_G`: `final_gt_l1=0.151337`, `final_white_ratio_hole=0.407880`
    - interpretation: `best_total_G ~= best_G`; `6000_G` is materially worse
  - `000098_left`
    - x26 `best_G`: `final_gt_l1=0.082415`, `final_white_ratio_hole=0.000000`
    - x26 `best_total_G`: `final_gt_l1=0.081114`, `final_white_ratio_hole=0.000000`
    - x26 `6000_G`: `final_gt_l1=0.084867`, `final_white_ratio_hole=0.071522`
    - interpretation: even on an easier sample, `6000_G` reintroduces residual white
  - `000098_right`
    - x26 `best_G`: `final_gt_l1=0.166833`, `final_white_ratio_hole=0.575777`
    - x26 `best_total_G`: `final_gt_l1=0.165594`, `final_white_ratio_hole=0.572551`
    - x26 `6000_G`: `final_gt_l1=0.253087`, `final_white_ratio_hole=0.806923`
    - interpretation: same pattern; `best_total_G ~= best_G`, `6000_G` is clearly worse
  - `000098_top`
    - x26 `best_G`: `final_gt_l1=0.095655`, `final_white_ratio_hole=0.000000`
    - x26 `best_total_G`: `final_gt_l1=0.095574`, `final_white_ratio_hole=0.000000`
    - x26 `6000_G`: `final_gt_l1=0.100269`, `final_white_ratio_hole=0.096392`
    - interpretation: `best_total_G` and `best_G` are functionally identical; `6000_G` again degrades

- Operational conclusion for x26:
  - checkpoint fishing inside x26 does **not** change the core diagnosis
  - `best_total_G` does not provide a new regime beyond `best_G`
  - the late `6000_G` checkpoint is not a better final model; it re-whitens hard samples and even reintroduces white on easier ones
  - therefore, do **not** spend more time expecting a later x26 checkpoint to rescue the remaining white/pink issues
  - if time is limited, the correct next move is to leave x26 checkpoint selection as solved and move to the next minimal-change branch (`x27`) rather than continue probing more x26 snapshots


## 2026-05-03 x27 regression verdict: the GT color-aux branch makes the white failure worse; kill x27 and return to x26 as the final white-stable base

- Compared logs:
  - train:
    - `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x27-ramped-colorfix_260503-150518.log`
  - test:
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x27-ramped-colorfix-current-domain_260503-190251.log` (`best_G`)
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x27-ramped-colorfix-current-domain_260503-190559.log` (`best_total_G`)
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x27-ramped-colorfix-current-domain_260503-190848.log` (`latest_G`)

- x27 runtime hygiene is correct:
  - warm start is exactly from x20 stable base:
    - `pretrain_model_G=/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x20-strongmask-cleancompose/models/best_G.pth`
  - test route is still the stable one:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=False`
  - structure checkpoint path remains `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  - therefore x27's regression is **not** a route mismatch, **not** a wrong structure checkpoint, and **not** a wrong warm start

- x27 code/config difference vs x26 is minimal but decisive:
  - x26:
    - `color_aux_loss_weight: 0.0`
  - x27:
    - `color_aux_loss_weight: 0.01`
    - `color_aux_target_domain: gt`
    - `best_save_start_iter: 1500`
  - since x27 `latest_G` is also worse than x26, this is **not** just a best-checkpoint selection problem
  - the only new training signal with explanatory power is the x27 GT low-frequency color auxiliary

- Direct hard-sample comparison, x26 `best_G` vs x27 `best_G`:
  - `000098_bottom`
    - x26: `final_gt_l1=0.096073`, `final_white_ratio_hole=0.438620`
    - x27: `final_gt_l1=0.135102`, `final_white_ratio_hole=0.609356`
  - `000098_center`
    - x26: `final_gt_l1=0.105230`, `final_white_ratio_hole=0.231778`
    - x27: `final_gt_l1=0.122997`, `final_white_ratio_hole=0.375447`
  - `000098_left`
    - x26: `final_gt_l1=0.082415`, `final_white_ratio_hole=0.000000`
    - x27: `final_gt_l1=0.079915`, `final_white_ratio_hole=0.042519`
  - `000098_right`
    - x26: `final_gt_l1=0.166833`, `final_white_ratio_hole=0.575777`
    - x27: `final_gt_l1=0.178829`, `final_white_ratio_hole=0.656248`
  - `000098_top`
    - x26: `final_gt_l1=0.095655`, `final_white_ratio_hole=0.000000`
    - x27: `final_gt_l1=0.095174`, `final_white_ratio_hole=0.061422`

- x27 checkpoint comparison shows the whole line is unstable, not just one checkpoint:
  - `best_G` and `best_total_G` are almost the same bad line:
    - `000098_bottom`: `0.609356` vs `0.610808`
    - `000098_center`: `0.375447` vs `0.361924`
    - `000098_right`: `0.656248` vs `0.673848`
  - `latest_G` is even worse:
    - `000098_bottom`: `final_white_ratio_hole=0.757555`
    - `000098_center`: `final_white_ratio_hole=0.402042`
    - `000098_right`: `final_white_ratio_hole=0.795515`
  - conclusion: x27 is a dead branch for anti-white purposes; do not continue it

- Interpretation:
  - x26 had already shown that the remaining white problem lives in the low-confidence hole subset during the late reverse trajectory
  - x27's GT low-frequency color auxiliary pushes the trunk toward a brighter coarse hole solution and re-strengthens the old white attractor
  - because the regression appears in `best_G`, `best_total_G`, and `latest_G`, x27 should be treated as a full objective-level regression, not a checkpoint-picking accident


## 2026-05-03 x28 final white-fix base: resume from x26 `best_G`, remove x27 regression term, keep the x26 anti-white objective unchanged

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x28-x26resume-whitefixfinal.yml`
- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x28-x26resume-whitefixfinal-current-domain.yml`

- x28 policy:
  - treat x26 `best_G` as the current best anti-white base and continue from it directly
  - warm start path is hard-guarded to:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x26-ramped-smallmainmix/models/best_G.pth`
  - keep the x26 anti-white objective intact:
    - `color_aux_loss_weight: 0.0`
    - `sde_mu_hole_mode: known_only`
    - `main_state_mode: hybrid_mid_blank_hole`
    - `main_mid_blank_ratio: 0.10`
    - `infer_x0_mid_loss_weight: 0.0015`
    - `texture_hf_mid_loss_weight: 0.005`
  - remove x27's GT color-anchor regression signal entirely

- Why x28 resumes from x26 instead of restarting from x20:
  - x26 is the best white-stable checkpoint line currently observed
  - x27 proved that restarting from x20 and injecting extra objectives can easily drag the trunk back into the white basin
  - with time limited, the least-risk final move is to continue from the known-good x26 anti-white base rather than relearn it

- Why x28 also changes the fine-tuning schedule:
  - x26 `6000_G` was worse than x26 `best_G`, so long continuation at the old schedule can drift back toward the bad basin
  - x28 therefore uses a conservative refinement schedule:
    - lower LR: `lr_G=2e-7`
    - shorter run: `niter=2500`
    - faster checkpoint cadence: `save_checkpoint_freq=500`
    - early best selection: `best_save_start_iter=200`
  - x28 also removes the old ramp on resume:
    - `main_mid_blank_ratio_start=0.10`
    - `main_mid_blank_ratio_warmup_iter=0`
  - this keeps the resumed x26 objective in its intended steady regime from iter 0, instead of replaying the cold-start ramp designed for x20

- Expected x28 diagnostic signature:
  1. train log must show x26 `best_G.pth` as the actual warm start
  2. `stats_main_mid_blank_ratio_requested` should stay at `0.10` from the start
  3. first probe should compare against x26 `best_G` immediately after the first saved checkpoint (500 or early `best_G`)
  4. success criterion is simple: the `000098_bottom/center/right` white ratios must not exceed the x26 `best_G` baseline again


## 2026-05-03 x28 result: current best white-stable base; residual white remains only on the hard low-confidence holes

- Logs:
  - train: `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x28-x26resume-whitefixfinal_260503-205142.log`
  - test: `C:\Users\admin\Desktop\test_ir-sde-brushnet-x28-x26resume-whitefixfinal-current-domain_260503-230610.log`

- Training hygiene is correct:
  - actual warm start:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x26-ramped-smallmainmix/models/best_G.pth`
  - route objective is the intended steady x26 regime from iter 0:
    - `stats_main_mid_blank_ratio_requested=0.1000`
    - actual batch-rounded `stats_main_mid_blank_ratio=0.1250` (`2 / 16`)
  - no x27 regression term:
    - `loss_color_aux=0`
  - early best checkpoint appears at iter `618`

- x28 is better than x26 on most of the hard probe set:
  - `000098_bottom`
    - x26: `final_gt_l1=0.096073`, `final_white_ratio_hole=0.438620`
    - x28: `final_gt_l1=0.077664`, `final_white_ratio_hole=0.345569`
  - `000098_center`
    - x26: `final_gt_l1=0.105230`, `final_white_ratio_hole=0.231778`
    - x28: `final_gt_l1=0.102192`, `final_white_ratio_hole=0.237353`
    - interpretation: essentially tied on white, slightly better on GT error
  - `000098_left`
    - x26: `final_gt_l1=0.082415`, `final_white_ratio_hole=0.000000`
    - x28: `final_gt_l1=0.077488`, `final_white_ratio_hole=0.000000`
  - `000098_right`
    - x26: `final_gt_l1=0.166833`, `final_white_ratio_hole=0.575777`
    - x28: `final_gt_l1=0.129230`, `final_white_ratio_hole=0.472720`
  - `000098_top`
    - x26: `final_gt_l1=0.095655`, `final_white_ratio_hole=0.000000`
    - x28: `final_gt_l1=0.092468`, `final_white_ratio_hole=0.000000`

- Residual white is still the same old mechanism:
  - the input priors are not white, but late reverse still grows white in the hard hole subset
  - confidence slices still show the failure is concentrated in low-reliability regions:
    - `000098_bottom`: `final_low white=0.6260` vs `final_high white=0.1336`
    - `000098_center`: `final_low white=0.4001` vs `final_high white=0.1394`
    - `000098_right`: `final_low white=0.7754` vs `final_high white=0.2404`
  - conclusion: x28 is the best anti-white line so far, but it still does not completely solve the hard low-confidence white basin

- Practical decision under time pressure:
  - fully eliminating the remaining white would likely require another dedicated white-focused line
  - given current time constraints, x28 should be treated as the final white-stable base and further work should move sideways (texture/detail modules) rather than spending more iterations trying to zero out the last hard white cases


## 2026-05-03 x29 direction: stop chasing residual white directly; add `texture_core` on top of the x28 base with train/test symmetry

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x29-x28resume-texturecore.yml`
- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x29-x28resume-texturecore-current-domain.yml`

- Why this branch:
  - user time is limited, and x28 already gives the best white-stable base so far
  - residual white remains only in hard low-confidence holes
  - the next most practical move is to add a sidecar texture/detail module on top of the stable base rather than keep perturbing the trunk objective

- x29 policy:
  - warm start from x28 `best_G`:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x28-x26resume-whitefixfinal/models/best_G.pth`
  - keep the x28 anti-white trunk objective unchanged:
    - `color_aux_loss_weight: 0.0`
    - `sde_mu_hole_mode: known_only`
    - `main_mid_blank_ratio: 0.10`
    - `infer_x0_mid_loss_weight: 0.0015`
    - `texture_hf_mid_loss_weight: 0.005`
  - enable `texture_core` in both training and test configs with the previously-tested light x21-style settings:
    - `enabled: true`
    - `insert_mid: true`
    - `insert_dec: false`
    - `gate_hidden: 8`
    - `boundary_width: 2`
    - `zero_init_last: true`
  - keep train/test network structure symmetric to avoid route mismatch

- x29 optimisation policy:
  - trunk remains conservative:
    - `lr_G=2e-7`
  - new texture-core parameters learn faster:
    - `lr_new=1e-6`
  - initial trunk freeze to preserve x28 white stability while the new texture branch warms up:
    - `freeze_pretrained_until_iter=400`
    - `freeze_loaded_pretrained_only=true`
  - short run with frequent checkpoints:
    - `niter=3000`
    - `save_checkpoint_freq=500`
    - `best_save_start_iter=500`

- Expected x29 behavior:
  - do **not** expect it to magically solve the hard white problem
  - expect it to preserve x28's white behavior as much as possible while trying to improve texture/detail on the already-stable samples


## 2026-05-03 overnight decision: do not re-enable `restore_S_guidance`; scale up the sidecar texture branch instead

- User requested a larger overnight run instead of another small few-thousand-step tweak.
- Code/history evidence says **do not** re-enable the original structure-guidance path (`restore_S_guidance`) for this overnight shot:
  - early historical configs (`x8`-`x12`) kept `restore_S_guidance=true`
  - tracking already records that turning off `restore_S_guidance` was one of the key steps toward removing the old white-failure route
  - every stable current-domain line from `x13` through `x29` keeps:
    - `restore_S_guidance=false`
  - x28, the current best white-stable base, also keeps:
    - `restore_S_guidance=false`
- Therefore the original structure-guidance path is not a safe 鈥渂igger overnight鈥?add-on under the current degraded-known route; it is a historically high-risk switch for white regression.


## 2026-05-03 x30 overnight branch: keep x28 white-stable route, but scale `texture_core` to the stronger original-enhanced setting

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore.yml`
- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x30-x28resume-overnight-texturecore-current-domain.yml`

- x30 policy:
  - warm start from x28 `best_G`:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x28-x26resume-whitefixfinal/models/best_G.pth`
  - keep the whole x28 anti-white route unchanged:
    - `restore_S_guidance=false`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `sde_mu_hole_mode=known_only`
    - `main_mid_blank_ratio=0.10`
    - `infer_x0_mid_loss_weight=0.0015`
    - `texture_hf_mid_loss_weight=0.005`
  - do **not** re-enable `mu_denoiser`
  - enlarge the sidecar texture branch to the stronger original-enhanced setting instead of the lighter x21-style setting:
    - `texture_core.enabled=true`
    - `insert_mid=true`
    - `insert_dec=true`
    - `gate_hidden=16`
    - `boundary_width=3`
    - `zero_init_last=true`

- Why x30 is the right overnight compromise:
  - It is a genuinely bigger change than x29
  - It uses the already-best x28 white-stable trunk as the base
  - It keeps train/test network structure strictly symmetric
  - It avoids the historically risky `restore_S_guidance` switch

- x30 optimisation schedule is intentionally overnight-sized:
  - `niter=12000`
  - `save_checkpoint_freq=1000`
  - `best_save_start_iter=1000`
  - trunk stays conservative:
    - `lr_G=2e-7`
  - new texture branch learns faster:
    - `lr_new=2e-6`
  - initial trunk freeze while new branch warms up:
    - `freeze_pretrained_until_iter=800`

- Expected x30 behavior:
  - it is still **not** designed to fully eliminate the remaining hard white cases
  - it is designed to spend the overnight budget on texture/detail improvement while preserving as much of x28 white stability as possible


## 2026-05-04 x30 result: residual white is strongly brightness-sensitive; light low-confidence holes still blow out, dark holes are mostly stable

- Logs:
  - train: `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore_260504-001142.log`
  - test: `C:\Users\admin\Desktop\test_ir-sde-brushnet-x30-x28resume-overnight-texturecore-current-domain_260504-103622.log`

- Training hygiene is correct:
  - warm start is exactly from x28 best:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x28-x26resume-whitefixfinal/models/best_G.pth`
  - train/test structure stays symmetric:
    - `texture_core.enabled=True`
    - `restore_S_guidance=False`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
  - therefore the overnight result can be interpreted as a real model-behavior result, not a route mismatch

- x30 confirms a more specific pattern than just 鈥渟ome samples still white鈥?
  - **light / bright holes with low confidence remain the unstable subset**
  - **dark holes are much more stable even when confidence is not especially high**

- Representative bright-hole failures:
  - `000098_bottom`
    - `prior_hole(mean)=0.7957`
    - `confidence_hole_mean=0.3974`
    - `final_hole(mean)=0.9791`
    - `final_white_ratio_hole=0.6312`
    - low-confidence slice: `final_low white=0.9568`
  - `000098_right`
    - `prior_hole(mean)=0.7989`
    - `confidence_hole_mean=0.3970`
    - `final_hole(mean)=0.9917`
    - `final_white_ratio_hole=0.6766`
    - low-confidence slice: `final_low white=0.9546`
  - `000098_center`
    - `prior_hole(mean)=0.7068`
    - `confidence_hole_mean=0.4177`
    - `final_hole(mean)=0.7454`
    - `final_white_ratio_hole=0.3661`
    - low-confidence slice: `final_low white=0.5308`

- Representative darker / lower-luminance holes:
  - `000180_bottom`
    - `prior_hole(mean)=0.3594`
    - `confidence_hole_mean=0.4093`
    - `final_hole(mean)=0.3911`
    - `final_white_ratio_hole=0.0000`
  - `000180_right`
    - `prior_hole(mean)=0.4029`
    - `confidence_hole_mean=0.3827`
    - `final_hole(mean)=0.3922`
    - `final_white_ratio_hole=0.0784`
  - `000257_bottom`
    - `prior_hole(mean)=0.6116`
    - `confidence_hole_mean=0.4188`
    - `final_hole(mean)=0.6561`
    - `final_white_ratio_hole=0.0626`

- Key interpretation:
  - the failure is **not** simply 鈥渓ow confidence => white鈥?
  - confidence matters, but **hole luminance / brightness prior matters too**
  - more accurate statement:
    - the remaining white failure happens mainly on **bright, smooth, low-confidence holes**
    - dark holes are comparatively robust even when confidence is mediocre

- Consequence for next-step planning:
  - adding `texture_core` alone does **not** fix the bright-hole whitening mechanism
  - this means the residual issue is now more about **luminance calibration / brightness overshoot** in the trunk reverse process than about missing texture modules
  - so:
    1. if the project goal is 鈥渙verall best practical result under time pressure鈥? x30/x28 can be accepted as the current base and work can continue on other modules
    2. if the goal is specifically to eliminate the remaining bright-hole whitening, that requires a dedicated brightness-targeted fix rather than just adding more generic structure/detail branches


## 2026-05-04 x31 overnight full-module branch: keep x30 current-domain route, but intentionally re-enable original structure guidance and Mu-Denoiser

- User decision:
  - x30 logs/local bad cases now show a very specific residual failure: **bright / smooth / low-confidence holes still blow out to white**, while darker holes are mostly stable.
  - User no longer wants to keep spending the overnight budget on small anti-white-only tweaks, and explicitly asked to **add the other modules back in, including structure guidance**, then run a longer training stage.

- Evidence-based risk statement before enabling it:
  - tracking already records that `restore_S_guidance=false` was part of the stable x20/x26/x28/x30 current-domain route
  - therefore x31 is **not** treated as a low-risk safe extension
  - it is an intentional, higher-risk "full-module overnight" branch because the user prefers broader capability gain over continuing to isolate the white issue first

- Code support added to make this branch technically sound:
  - file:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
  - new behavior:
    - primary load still uses `path.pretrain_model_G`
    - optional secondary init path now supports:
      - `path.pretrain_model_G_fallback`
      - `path.pretrain_model_G_fallback_only_missing`
    - when the primary checkpoint misses tensors (the main x31 case is the newly re-enabled `restore_S_guidance` SPADE blocks), the loader can pull **only missing keys** from a fallback checkpoint
  - why this is needed:
    - x31 resumes from x30 `best_G`, but x30 was trained with `restore_S_guidance=false`
    - turning `restore_S_guidance=true` adds the legacy SPADE structure-guidance tensors back into `network_G`
    - without fallback init, those tensors would be random
    - with fallback init, x31 can reuse the original StrDiffusion-compatible SPADE weights exactly as the wrapper layout was designed for

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x31-x30resume-fullmodules-restores.yml`
- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x31-x30resume-fullmodules-restores-current-domain.yml`

- x31 policy:
  - base checkpoint:
    - primary warm start from x30 `best_G`
      - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
    - hard guard:
      - `expected_pretrain_model_G` points to the same x30 `best_G`
  - restore the original structure-guidance path:
    - `restore_S_guidance=true`
  - keep current active sidecar modules on:
    - `texture_core.enabled=true`
    - `insert_mid=true`
    - `insert_dec=true`
  - re-enable Mu-Denoiser for the long run:
    - `mu_denoiser.enabled=true`
    - `use_for_condition_mu=false`
  - keep the current-domain route semantics unchanged:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `sde_mu_hole_mode=known_only`
  - structure-network checkpoint path remains fixed to the required path with trailing `s`:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- x31 fallback init source for the newly restored SPADE path:
  - `path.pretrain_model_G_fallback`:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
  - `path.pretrain_model_G_fallback_only_missing=true`
  - intended effect:
    - reuse the original StrDiffusion-compatible SPADE tensors only for the keys that x30 `best_G` does not contain
    - keep the rest of the x30 trunk / BrushNet / texture-core weights untouched

- x31 optimisation schedule:
  - long overnight run size stays at:
    - `niter=12000`
  - checkpoint cadence:
    - `save_checkpoint_freq=1000`
    - `best_save_start_iter=1500`
  - trunk stays conservative:
    - `lr_G=2e-7`
  - newly activated / newly missing-filled modules learn faster but not as aggressively as x30 texture-only overnight:
    - `lr_new=1e-6`
  - initial freeze remains:
    - `freeze_pretrained_until_iter=800`
    - `freeze_loaded_pretrained_only=true`

- Train/test symmetry check for x31:
  - both train and test configs now agree on:
    - `texture_core.enabled=true`
    - `restore_S_guidance=true`
    - `mu_denoiser.enabled=true`
  - this avoids another train/test network mismatch while running the full-module branch

- Validation:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
    - `py_compile` passed
  - x31 train/test YAML:
    - `yaml.safe_load` passed

- Important interpretation:
  - x31 is **not** a claim that the bright-hole white issue has been solved
  - x31 is a deliberate, user-driven switch from "continue isolating anti-white fixes" to "accept current residual white and spend the overnight budget on the broader original-enhanced full stack"

## 2026-05-04 x31 full-module branch result: not a simple unconverged case, should stop and adjust direction

- Logs analysed:
  - train:
    - `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x31-x30resume-fullmodules-restores_260504-114419.log`
  - test:
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x31-x30resume-fullmodules-restores-current-domain_260504-165341.log` (`best_G`)
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x31-x30resume-fullmodules-restores-current-domain_260504-170255.log` (`best_total_G`)

- Training-side facts:
  - primary warm start is correct:
    - `pretrain_model_G = .../ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
  - fallback for restored SPADE / structure-guidance tensors is active:
    - first load: `274/378`, `missing=104`
    - fallback attempt from original texture checkpoint
    - fallback actually fills only `64` tensors
  - optimizer grouping is notable:
    - `[Model] Param groups: pretrained=362 (lr=2.00e-07), new=0 (lr=1.00e-06)`
    - this means the x31 full-module branch is **not** really running with a visible `new`-param bucket in the main G optimizer
  - freeze is also active at the start:
    - `[Freeze] frozen 362 pretrained trunk params until iter 800`
  - Mu-Denoiser clearly trains and its loss drops fast:
    - early `loss_mu_total ~ 0.40`
    - around `iter 1600`, `loss_mu_total ~ 0.065`
  - total training loss keeps improving numerically:
    - e.g. `iter 1589 best-total loss_total = 6.3693e-02`
    - later still improves again around `iter 1733`, `loss_total = 6.1061e-02`

- Test-route check:
  - route is clean in both best and best_total tests:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `restore_S_guidance=True`
    - `texture_core.enabled=True`
    - `mu_denoiser.enabled=True`
  - structure checkpoint path is still the required one:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  - so x31 visual failures are **real model behaviour**, not another train/test mismatch

- Hard-sample outcome (`best_G`, source-of-record = logged white-ratio / hole stats / screenshots):
  - `000098_bottom`
    - `prior_hole(mean)=0.7957`
    - `final_hole(mean)=1.1022`
    - `final_white_ratio_hole=0.9369`
    - `final_low white=1.0000`, `final_high white=0.8892`
  - `000098_center`
    - `prior_hole(mean)=0.7068`
    - `final_hole(mean)=0.8160`
    - `final_white_ratio_hole=0.5385`
    - `final_low white=0.6517`, `final_high white=0.4704`
  - `000098_left`
    - `prior_hole(mean)=0.5439`
    - `final_hole(mean)=0.6202`
    - `final_white_ratio_hole=0.1369`
  - `000098_right`
    - `prior_hole(mean)=0.7989`
    - `final_hole(mean)=1.1177`
    - `final_white_ratio_hole=0.9346`
    - `final_low white=0.9995`, `final_high white=0.8848`
  - `000098_top`
    - `prior_hole(mean)=0.5189`
    - `final_hole(mean)=0.6017`
    - `final_white_ratio_hole=0.1707`
  - `000180_bottom`
    - `prior_hole(mean)=0.3594`
    - `final_hole(mean)=0.4337`
    - `final_white_ratio_hole=0.0000`

- Hard-sample outcome (`best_total_G`, source-of-record = logged white-ratio / hole stats / screenshots):
  - `000098_bottom`
    - `final_hole(mean)=1.0898`
    - `final_white_ratio_hole=0.9251`
    - `final_low white=1.0000`, `final_high white=0.8685`
  - `000098_center`
    - `final_hole(mean)=0.7926`
    - `final_white_ratio_hole=0.5065`
    - `final_low white=0.6209`, `final_high white=0.4377`
  - `000098_left`
    - `final_hole(mean)=0.6299`
    - `final_white_ratio_hole=0.1475`
  - `000098_right`
    - `final_hole(mean)=1.0994`
    - `final_white_ratio_hole=0.9223`
    - `final_low white=0.9965`, `final_high white=0.8653`
  - `000098_top`
    - `final_hole(mean)=0.5996`
    - `final_white_ratio_hole=0.1611`
  - `000180_bottom`
    - `final_hole(mean)=0.4372`
    - `final_white_ratio_hole=0.0000`

- Interpretation:
  - x31 is **not** a mild undertrained / waiting-to-converge branch
  - `best_G` and `best_total_G` are both bad in the same bright-hole way
  - the catastrophic white failure on `000098_bottom` / `000098_right` is already close to old collapse-style behaviour (`white ~0.93`)
  - darker holes remain stable, so this is still the same brightness-sensitive failure mode, but x31 makes it much worse than x28/x30
  - because training loss keeps improving while the bright-hole visual failure remains catastrophic, this should be judged as an **objective / direction regression**, not as a simple convergence issue

- Decision:
  - **do not spend more time continuing x31 as-is waiting for convergence**
  - treat x31 as a wrong-direction full-module branch for the current degraded-known current-domain route
  - if further work is needed, roll back to the x28/x30 family and re-add modules much more selectively instead of keeping `restore_S_guidance=True` and hoping longer training will fix it

## 2026-05-04 x32 plan: keep texture_core + Mu-Denoiser + restore_S_guidance, but make restored SPADE guidance truly train as a selective "new branch"

- User requirement:
  - keep the already useful modules:
    - `texture_core`
    - `mu_denoiser`
  - **structure guidance must also stay on**
  - train/test network must remain strictly consistent

- x31-specific issue identified from code + logs:
  - x31 turned `restore_S_guidance=true`, but the optimizer grouping showed:
    - `[Model] Param groups: pretrained=362, new=0`
  - this means the restored SPADE guidance branch did **not** end up in a visible higher-LR / new-module bucket
  - combined with:
    - fallback only filling part of the missing tensors (`64 / 104`)
    - initial freeze of loaded pretrained params
  - the restored structure-guidance path in x31 was not being trained in the intended "new-side-branch warmup" way

- Code change to support a selective structure-on branch:
  - file:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\denoising_model.py`
  - new behaviour:
    - added `_get_new_module_prefixes()`
    - `train.force_new_param_prefixes` can now explicitly mark loaded parameters as **new**, even when `freeze_loaded_pretrained_only=true`
    - `_resolve_pretrained_param_names(...)` now excludes any forced-new prefixes from the loaded/pretrained set
    - `_apply_pretrained_trunk_freeze()` now uses the same merged new-prefix set, so these forced-new params will not be frozen with the pretrained trunk
  - intended effect:
    - keep the x30 trunk / BrushNet / texture-core weights stable
    - but let the restored SPADE structure-guidance tensors actually adapt as a selective high-LR branch

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x32-x30resume-restoreSselective.yml`
- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x32-x30resume-restoreSselective-current-domain.yml`

- x32 route policy:
  - base warm start stays on x30:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
  - fallback for restored structure-guidance tensors still uses:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
  - keep:
    - `texture_core.enabled=true`
    - `mu_denoiser.enabled=true`
    - `restore_S_guidance=true`
  - keep stable current-domain route semantics:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `sde_mu_hole_mode=known_only`
  - structure checkpoint path remains exactly:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- x32 selective-new prefixes:
  - `downs.0.4.`
  - `downs.1.4.`
  - `downs.2.4.`
  - `downs.3.4.`
  - rationale:
    - in `brushnet_wrapper.py`, when `restore_S_guidance=true`, each down block appends `SPADEBlock` at `downs[i][4]`
    - these are exactly the restored legacy structure-guidance tensors that need to adapt, instead of being silently treated as frozen/loaded trunk

- x32 optimisation schedule:
  - `lr_G = 2e-7`
  - `lr_new = 1e-6`
  - `freeze_pretrained_until_iter = 800`
  - `freeze_loaded_pretrained_only = true`
  - `best_save_start_iter = 2000`
  - `niter = 12000`

- Train/test consistency:
  - x32 train and test both enable:
    - `texture_core`
    - `mu_denoiser`
    - `restore_S_guidance`
  - this keeps the x32 branch structurally symmetric and avoids another train/test mismatch

- Validation:
  - `denoising_model.py`
    - `py_compile` passed
  - x32 train/test YAML
    - `yaml.safe_load` passed

## 2026-05-04 x32 early result: structure-on selective branch is learning now, but still only a partial recovery

- Logs analysed:
  - train:
    - `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x32-x30resume-restoreSselective_260504-190712.log`
  - test:
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x32-x30resume-restoreSselective-current-domain_260504-204139.log`

- Good news first:
  - x32 fixes the x31 optimiser-grouping issue:
    - `[Model] Param groups: pretrained=151 (lr=2.00e-07), new=211 (lr=1.00e-06)`
  - this means the restored `restore_S_guidance` SPADE path is no longer silently trapped in the pure pretrained/frozen bucket
  - train/test route is also still fully consistent:
    - `texture_core.enabled=True`
    - `mu_denoiser.enabled=True`
    - `restore_S_guidance=True`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - structure checkpoint path still uses required `e00s` path

- Training-side status:
  - x32 is still early relative to the planned `niter=12000`
  - best tracking only starts after `iter >= 2000`
  - after the first valid best stage, total loss improves quickly:
    - `iter 2000 best-total loss_total = 8.2369e-02`
    - `iter 2106 best-total loss_total = 6.3586e-02`
  - current observed tail around `iter 2260~2360` is roughly:
    - `loss_total ~ 7.2e-02`
  - so x32 is **not** dead-on-arrival; the selective structure branch is genuinely learning now

- Visual/result status versus x31:
  - the hard bright-hole white failure is still present, but **slightly better** than x31
  - `000098_bottom`
    - x31 best_G: `final_white_ratio_hole=0.9369`
    - x32 current best_G: `final_white_ratio_hole=0.8743`
  - `000098_right`
    - x31 best_G: `0.9346`
    - x32 current best_G: `0.8724`
  - `000098_center`
    - x31 best_G: `0.5385`
    - x32 current best_G: `0.5334`
  - lighter/easier cases also improve slightly:
    - `000098_left`: `0.1369 -> 0.1282`
    - `000098_top`: `0.1707 -> 0.1503`
  - dark cases remain stable:
    - `000180_bottom`: `final_white_ratio_hole=0.0013`

- Interpretation:
  - x32 is **better than x31**, so the selective-new structure fix did move the branch in the right direction
  - but the improvement is still only partial; the bright-hole white failure remains severe on the same hard samples
  - therefore x32 should **not** be treated as already solved, but it also should **not** be discarded as quickly as x31

- Decision recommendation:
  - if the constraint is **structure must stay on**, x32 is the first branch worth giving a bit more time
  - recommended next step is:
    - continue x32 only to a short next checkpoint window (roughly `iter 4000~5000`)
    - retest the same hard samples before committing to the full 12k run
  - stop/adjust immediately if the hard bright-hole metrics still do not move materially:
    - `000098_bottom` / `000098_right` still above about `0.80` white
    - `000098_center` still around `0.50+`
  - in other words:
    - **continue a bit, but with a stop-loss**
    - do **not** just assume blind long training to 12k will automatically solve it

## 2026-05-04 x32 follow-up at ~6k: stop-loss hit, continued training regressed bright-hole white failure

- Logs analysed:
  - train:
    - `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x32-x30resume-restoreSselective_260504-190712.log`
  - test:
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x32-x30resume-restoreSselective-current-domain_260504-233151.log`

- Train-side status up to ~6k:
  - x32 continued training normally and still kept the corrected optimiser grouping:
    - `[Model] Param groups: pretrained=151 (lr=2.00e-07), new=211 (lr=1.00e-06)`
  - the trunk unfreeze happened as expected at `iter 800`
  - best-total kept improving after the early 2k stage:
    - `iter 2916 best-total loss_total = 6.2348e-02`
    - `iter 3403 best-total loss_total = 6.1367e-02`
    - `iter 3609 best-total loss_total = 6.2014e-02`
  - but by the time training reached `iter 4000~6000`, the online loss no longer showed a clean monotonic gain toward a visibly better anti-white solution:
    - `iter 4000 loss_total = 6.7172e-02`
    - `iter 5900 loss_total = 6.6536e-02`
    - `iter 6000 loss_total = 8.0857e-02`

- Test-side result:
  - the tested route is still structurally correct and symmetric:
    - `texture_core.enabled=True`
    - `mu_denoiser.enabled=True`
    - `restore_S_guidance=True`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - structure checkpoint path still uses the required:
      - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  - however, the actual bright-hole white failure got **worse** than the earlier x32 test

- x32 early test (`204139`) -> later x32 test (`233151`) on the same hard cases:
  - `000098_bottom`
    - `final_white_ratio_hole: 0.8743 -> 0.9724`
  - `000098_center`
    - `0.5334 -> 0.6137`
  - `000098_left`
    - `0.1282 -> 0.2655`
  - `000098_right`
    - `0.8724 -> 0.9553`
  - `000098_top`
    - `0.1503 -> 0.2159`
  - the darker reference case still remains stable:
    - `000180_bottom` continues to avoid this bright-white collapse pattern

- Additional hard evidence from the later x32 test:
  - `000098_bottom`
    - `final_hole_mean=1.1478`
    - `final_low white=1.0000`
    - `final_high white=0.9516`
  - `000098_right`
    - `final_hole_mean=1.1667`
    - `final_low white=0.9993`
    - `final_high white=0.9215`
  - this is no longer just "partial residual white"; this is a strong return toward the old bright-hole white attractor

- Final interpretation for x32:
  - the selective-new structure fix **did** solve the x31 optimiser-grouping bug
  - but after giving x32 the requested extra time, the stop-loss condition was hit:
    - the hard bright-hole white metrics did not improve into a safe range
    - and by the later test they clearly regressed
  - therefore x32 should **not** be continued further as-is

- Decision:
  - **stop x32**
  - do **not** keep waiting for more convergence on this branch
  - the next step should be a direction adjustment, not more blind training time on the same x32 setup

## 2026-05-04 x33 restart version: gated restore-S guidance instead of full-strength SPADE overwrite

- User question after x32:
  - whether the newly re-enabled modules have a structural problem
  - request a clean restart version rather than continuing the regressed x32 branch

- Code-level diagnosis:
  - in `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\brushnet_wrapper.py`
  - the restored structure-guidance path was previously applied as:
    - `x = blocks[4](x, S)`
  - unlike:
    - BrushNet (`feature_scale`)
    - texture_core (`zero_init_last`)
  - the restored SPADE path had **no strength gate at all**
  - this matches the observed failure pattern:
    - bright / shallow / low-confidence holes overexpose badly
    - darker holes remain comparatively stable
  - therefore the issue is not a train/test mismatch; it is more likely a **full-strength restore-S injection problem** when mixing current-domain x30 trunk + restored legacy SPADE guidance

- Structural fix added:
  - file:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\models\brushnet_wrapper.py`
  - new option:
    - `restore_S_guidance_scale`
  - new behaviour:
    - if `restore_S_guidance=true`, the SPADE-guided feature is blended instead of hard-overwriting:
      - `x = x + scale * (guided - x)`
    - `scale=1.0` keeps old behaviour
    - `scale<1.0` makes structure guidance softer and reduces the risk of bright-hole overdrive

- Restart branch created:
  - train config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x33-x30resume-restoreSgated.yml`
  - test config:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x33-x30resume-restoreSgated-current-domain.yml`

- x33 route policy:
  - restart from the safer x30 base again:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
  - keep fallback fill for missing structure-guidance tensors from:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
  - keep modules enabled in both train and test:
    - `texture_core=true`
    - `mu_denoiser=true`
    - `restore_S_guidance=true`
  - keep current-domain route semantics:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `sde_mu_hole_mode=known_only`
  - structure network checkpoint path remains exactly:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- x33 stabilisation choices:
  - `restore_S_guidance_scale: 0.2`
  - `lr_new: 5e-7` (down from x32 `1e-6`)
  - `freeze_pretrained_until_iter: 1200`
  - `best_save_start_iter: 2500`
  - keep selective-new prefixes for restored SPADE blocks:
    - `downs.0.4.`
    - `downs.1.4.`
    - `downs.2.4.`
    - `downs.3.4.`

- Validation:
  - `brushnet_wrapper.py`
    - `py_compile` passed
  - x33 train/test YAML
    - `yaml.safe_load` passed

- Expected purpose of x33:
  - not to remove all white failure instantly
  - but to test whether the main regression with structure-on came from **ungated restore-S guidance amplitude**
  - if x33 works, bright hard holes should stop racing toward near-1.0 hole means as aggressively as x32/x31

## 2026-05-05 correction after comparing original StrDiffusion: issue is not "original structure guidance is wrong", but our restart baseline was not original-like enough

- User objection is valid:
  - the original StrDiffusion structure-guidance path itself is not obviously broken
  - in the original code:
    - `D:\code\ky\bihua\Impainting\StrDiffusion\train\texture\config\inpainting\models\modules\DenoisingUNet_arch.py`
  - `ConditionalUNet` also uses a full `SPADEBlock` at each down stage, not a weakened gate
  - so the previous x33-style "softened structure amplitude" is **not** a faithful reproduction of the original structure-guidance regime

- More important original-vs-x31/x32 differences found in code/config:
  1. original structure-on route starts from a **structure-on pretrained checkpoint**
     - original enhanced config:
       - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet.yml`
       - `path.pretrain_model_G = /home/610-wws/Impainting/StrDiffusion+e00/train/experiments/inpainting/ir-sde/models/best_G.pth`
     - original finetune config comments explicitly recommend a fully converged structure-guided starting point
  2. original structure-on finetune uses:
     - `sde_mu_hole_mode: condition_lut`
     - not the later white-stable current-domain `known_only` route
  3. x31/x32 instead were enabling `restore_S_guidance=true` on top of an x30 base that had long been trained with:
     - `restore_S_guidance=false`
     - `sde_mu_hole_mode=known_only`

- Revised interpretation:
  - the main problem is more likely:
    - **"turning original-style full structure guidance back on over an x30 no-structure / known-only base"**
  - not simply:
    - **"structure guidance amplitude is inherently too large"**

## 2026-05-05 x34 restart version: original-like full restore-S restart for current-domain

- Goal:
  - give the user a restart version that is **closer to original StrDiffusion logic**
  - keep train/test structure consistent
  - keep current-domain degraded/prefill route
  - but avoid the x31/x32 mistake of using an x30 no-structure main checkpoint as the primary generator source

- New train config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion+e00\train-3\texture\config\inpainting\options\train\ir-sde-brushnet-ft-x34-origrestore-currentdomain.yml`

- New test config:
  - `D:\code\ky\bihua\Impainting\StrDiffusion\test\texture-1\config\inpainting\options\test\ir-sde-brushnet-x34-origrestore-currentdomain.yml`

- x34 key design:
  - keep:
    - `texture_core.enabled=true`
    - `mu_denoiser.enabled=true`
    - `restore_S_guidance=true`
  - do **not** use reduced `restore_S_guidance_scale`
    - x34 goes back to full original-like structure guidance
  - primary generator warm start now uses the original structure-on checkpoint:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train/experiments/inpainting/ir-sde/models/best_G.pth`
  - fallback fill now uses x30 to recover missing current branch tensors:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
  - switch back to the original structure-on hole-mu setting:
    - `sde_mu_hole_mode: condition_lut`
  - in test config, keep train/test consistency:
    - `sde_mu_hole_mode: condition_lut`
    - `expected_train_sde_mu_hole_mode: condition_lut`
  - keep current-domain inference semantics:
    - `condition_known_source=degraded`
    - `structure_source=prefill`
  - structure network checkpoint path remains exactly:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- x34 optimisation simplification:
  - `freeze_pretrained_until_iter: 0`
  - `lr_G: 5e-7`
  - `best_save_start_iter: 2500`
  - remove x32 selective forced-new SPADE prefixes from the config
  - rationale:
    - if the primary checkpoint already contains full structure-guidance weights,
      we should not treat the structure path as a newly attached x30-era addon

- Validation:
  - x34 train/test YAML
    - `yaml.safe_load` passed

- Current recommendation status:
  - if the user insists on an original-like structure-guidance restart,
    - **x34 is the correct restart branch to try next**
  - x33 remains a soft-gated experimental fallback idea,
    - but it is no longer the preferred branch after re-checking the original code/config logic

## 2026-05-05 x34 result: original-like restore-S + condition_lut is catastrophic on current-domain route; do not continue

- Logs analysed:
  - train:
    - `C:\Users\admin\Desktop\train_ir-sde-brushnet-ft-x34-origrestore-currentdomain_260505-005609.log`
  - test:
    - `C:\Users\admin\Desktop\test_ir-sde-brushnet-x34-origrestore-currentdomain_260505-102336.log`

- Train-side status:
  - route started as intended:
    - `freeze_pretrained_until_iter: 0`
    - `[Model] Param groups: pretrained=215 (lr=5.00e-07), new=147 (lr=1.00e-06)`
  - training was numerically alive, and best-total appeared around the late-2.8k stage:
    - `iter 2844 [best-total] loss_total = 6.4531e-02`
    - `iter 2851 [best-total] loss_total = 6.4677e-02`
    - by `iter 2900`, online `loss_total = 6.3129e-02`
  - therefore x34 is **not** a simple dead-on-arrival load failure

- Test-side route check:
  - train/test structure is still consistent:
    - `texture_core.enabled=True`
    - `mu_denoiser.enabled=True`
    - `restore_S_guidance=True`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
    - `sde_mu_hole_mode=condition_lut`
    - `expected_train_sde_mu_hole_mode=condition_lut`
  - structure checkpoint path is still correct:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- But actual x34 inference result is far worse than x32:
  - bright hard cases completely blow up:
    - `000098_bottom`
      - `final_hole_mean=16.1909`
      - `final_white_ratio_hole=1.0000`
    - `000098_center`
      - `final_hole_mean=7.8509`
      - `final_white_ratio_hole=0.9493`
    - `000098_right`
      - `final_hole_mean=21.2888`
      - `final_white_ratio_hole=0.9857`
  - even medium/easier 000098 cases regress badly:
    - `000098_left`
      - `final_white_ratio_hole=0.8343`
    - `000098_top`
      - `final_white_ratio_hole=0.7494`
  - unlike x30/x32, the failure is no longer limited to bright-hole subsets only
  - darker families also collapse:
    - `000180_bottom`
      - `final_white_ratio_hole=0.7926`
    - `000180_center`
      - `0.7479`
    - `000180_left`
      - `0.8914`
    - `000180_right`
      - `0.9323`
    - `000257_bottom`
      - `0.9562`
    - `000257_left`
      - `0.9982`
    - `000348_bottom`
      - `0.7364`
    - `000348_center`
      - `0.5231`

- Strong diagnostic clue:
  - x34 test repeatedly shows huge `MuAnchor Debug` distances under `condition_lut`, e.g.:
    - `000098_bottom final_lut_l1=15.433558`
    - `000098_right final_lut_l1=20.521713`
  - this is not a mild colour drift; it indicates the reverse trajectory is running far away from the intended LUT/colour anchor

- Interpretation:
  - the earlier user correction was right in the narrow sense:
    - original StrDiffusion structure guidance itself is not inherently "wrong"
  - but this x34 result proves:
    - **the original-like full restore-S + condition_lut recipe does not transfer directly to the current-domain degraded/prefill route**
  - this is not a 鈥渘eeds more convergence鈥?situation
  - x34 is a route-level regression and should be stopped immediately

- Decision:
  - **stop x34**
  - do **not** continue training this branch
  - treat `sde_mu_hole_mode=condition_lut` as incompatible with the active current-domain route when full structure guidance is on
  - any next structure-on branch should return to the current-domain-safe hole-mu semantics (i.e. not this x34 recipe)

## 2026-05-05 x35 correct restart: strictly revert to the documented current-domain safe route

- User feedback was correct: by the time x34 was created, the tracking file had already recorded the safe current-domain route clearly enough that we should not have reintroduced the original-like `restore_S_guidance=true` + `sde_mu_hole_mode=condition_lut` recipe.
- The validated current-domain safe route remains:
  - `restore_S_guidance=false`
  - `condition_known_source=degraded`
  - `structure_source=prefill`
  - `sde_mu_hole_mode=known_only`
- Important clarification:
  - **keeping `restore_S_guidance=false` does not mean "structure is off"**
  - the structure network is still used through the prefill route, and the required structure checkpoint path remains:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`
  - what stays off is only the texture trunk's internal original-style restore-S branch, because that branch repeatedly regressed on the current-domain route (`x31` / `x32` / `x34`)

- New train config:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x35-currentsafe-correctrestart.yml`
- New test config:
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x35-currentsafe-correctrestart-current-domain.yml`

- x35 policy:
  - this is intentionally a clean reissue of the x30 current-domain-safe recipe, not a new experimental branch
  - only experiment naming / checkpoint output target change
  - keep:
    - `brushnet.enabled=true`
    - `texture_core.enabled=true`
    - `restore_S_guidance=false`
    - `mu_denoiser.enabled=false`
    - `sde_mu_hole_mode=known_only`
    - `condition_known_source=degraded`
    - `structure_source=prefill`
  - train warm start remains x28 best_G:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x28-x26resume-whitefixfinal/models/best_G.pth`
  - test checkpoint target becomes x35 best_G:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x35-currentsafe-correctrestart/models/best_G.pth`

- Rationale:
  - x30 was the last verified current-domain-safe training route before the later restore-S regressions
  - x31 / x32 / x34 all proved that re-enabling the internal restore-S branch on the active current-domain degraded/prefill route is high-risk and was not justified by the accumulated evidence
  - therefore x35 should be treated as the "correct restart" branch for retraining

- Validation:
  - x35 train YAML parsed with `yaml.safe_load`
  - x35 test YAML parsed with `yaml.safe_load`
  - the required structure checkpoint path still points to:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

\n\n

## 2026-05-05 x36 correct enabled restart: if `restore_S_guidance` and `mu_denoiser` must both stay on, use the x32-style current-domain route rather than x34

- User explicitly requires these two switches to stay on:
  - `restore_S_guidance=true`
  - `mu_denoiser.enabled=true`
- Given that requirement, the technically correct route is **not** x34.
- Code/log evidence says the two major mistakes to avoid are:
  1. **x31 mistake**: structure guidance got turned on but the restored SPADE branch was not treated as a real trainable new branch (`Param groups ... new=0`)
  2. **x34 mistake**: route semantics were changed back to the original-like recipe (`sde_mu_hole_mode=condition_lut`) even though the tracking had already established the active current-domain safe semantics as `known_only` + `degraded` + `prefill`

- Therefore the correct enabled restart branch should keep:
  - `restore_S_guidance=true`
  - `restore_S_guidance_scale=1.0`
  - `mu_denoiser.enabled=true`
  - `texture_core.enabled=true`
  - `condition_known_source=degraded`
  - `structure_source=prefill`
  - `sde_mu_hole_mode=known_only`
- And it must also keep the x32 selective-trainability fix:
  - `train.force_new_param_prefixes:`
    - `downs.0.4.`
    - `downs.1.4.`
    - `downs.2.4.`
    - `downs.3.4.`
  - this is what prevents the x31 `new=0` optimizer-group bug from reappearing

- New train config:
  - `D:/code/ky/bihua/Impainting/StrDiffusion+e00/train-3/texture/config/inpainting/options/train/ir-sde-brushnet-ft-x36-x30resume-restoreSmu-correct.yml`
- New test config:
  - `D:/code/ky/bihua/Impainting/StrDiffusion/test/texture-1/config/inpainting/options/test/ir-sde-brushnet-x36-x30resume-restoreSmu-correct-current-domain.yml`

- x36 is intentionally a clean reissue of the x32 route, not x34:
  - warm start remains x30 best_G:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train-3/experiments/inpainting/ir-sde-brushnet-ft-x30-x28resume-overnight-texturecore/models/best_G.pth`
  - fallback still only fills missing tensors from the original texture checkpoint:
    - `/home/610-wws/Impainting/StrDiffusion+e00/train/texture/config/inpainting/log/ir-sde/models/best_G.pth`
  - test structure checkpoint path still remains exactly:
    - `/home/610-wws/Impainting/StrDiffusion+e00s/train/structure/config/inpainting/log/ir-sde/models/best_G.pth`

- Interpretation:
  - x36 is the **correct enabled version** in the narrow technical sense: it keeps the requested modules on, avoids the documented x31 optimizer bug, and avoids the documented x34 route-semantic mistake.
  - This does **not** mean x36 is already proven white-safe; it only means it is the correct way to reopen training if those two switches must stay on.
