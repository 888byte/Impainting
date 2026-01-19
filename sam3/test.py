import os
import cv2
import numpy as np
import torch
import random
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===================== 核心配置 =====================
BASE_PATH = "/home/lab610/Impainting/dataset/dunhuang"
RAW_IMG_FOLDER = os.path.join(BASE_PATH, "dunhuang")
CROPPED_FOLDER = os.path.join(BASE_PATH, "dunhuang_768x768")
MASK_OUTPUT_FOLDER = os.path.join(BASE_PATH, "dunhuang_mask")
HOLE_OUTPUT_FOLDER = os.path.join(BASE_PATH, "dunhuang_hole")   # 挖空图文件夹
VIS_OUTPUT_FOLDER  = os.path.join(BASE_PATH, "dunhuang_visual") # ✅ 新增：把彩色预览图放这里，别放 mask 文件夹

TARGET_COLORS = [
    {"name": "铅丹", "eng_name": "lead_red",  "lab": [50, 80, 70],  "tolerance": 20, "visual_color": (255, 0, 0)},
    {"name": "铅白", "eng_name": "lead_white","lab": [90, 0, 0],    "tolerance": 15, "visual_color": (255, 255, 255)},
    {"name": "密陀僧", "eng_name": "litharge","lab": [80, -10, 60], "tolerance": 20, "visual_color": (0, 255, 255)}
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.6
MASK_AREA_LIMIT = 0.5

# ===================== 工具函数 =====================
def generate_random_brush_mask(h, w):
    """当没有检测到颜料时，生成随机线条Mask (模拟划痕/破损)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    num_lines = random.randint(3, 8)

    for _ in range(num_lines):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        thickness = random.randint(10, 30)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    for _ in range(random.randint(2, 5)):
        cx, cy = random.randint(0, w), random.randint(0, h)
        radius = random.randint(10, 40)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    return mask > 0  # bool


def rgb2lab_and_find_single_color(img_rgb, color_info):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lower_lab = np.array([c - color_info["tolerance"] for c in color_info["lab"]])
    upper_lab = np.array([c + color_info["tolerance"] for c in color_info["lab"]])
    color_mask = cv2.inRange(img_lab, lower_lab, upper_lab)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        h, w = img_rgb.shape[:2]
        return np.array([w // 4, h // 4, 3 * w // 4, 3 * h // 4])
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)
    return np.array([x, y, x + w, y + h])


def resize_stretch_768(img_rgb):
    return cv2.resize(img_rgb, (768, 768), interpolation=cv2.INTER_LINEAR)


def generate_chinese_readme():
    readme_path = os.path.join(BASE_PATH, "README_DATASET.txt")
    content = f"""
敦煌壁画数据集处理说明文档
==================================================

一、文件夹结构
--------------------------------------------------
1. dunhuang_hole/ : 存放挖空图 input_with_hole_*.png（网络输入 I_deg）
2. dunhuang_mask/ : 存放二值 Mask：
   - mask_[color]_[SAM/RANDOM]_*.png（单色 mask）
   - ✅ mask_union_*.png（三色并集洞 mask，和 input_with_hole 严格对齐，训练建议用这个）
3. dunhuang_768x768/ : Ground Truth 原图（拉伸到 768x768）
4. ✅ dunhuang_visual/ : 存放彩色可视化预览 visual_overlay_*.png（注意：不是训练 mask！）

二、Mask 生成策略（混合模式）
--------------------------------------------------
对每张图，对每种颜色做一次分割，得到 final_mask_bool：

A) SAM 没检测到（全黑） -> 生成随机破损 Mask 补位（RANDOM）
B) 检测到但面积过大 -> 截断
C) 正常 -> 使用 SAM mask

三、训练使用建议
--------------------------------------------------
- 训练 inpainting / BrushNet 时，请使用：
  I_deg = dunhuang_hole/input_with_hole_*.png
  M     = dunhuang_mask/mask_union_*.png  （0/255 黑白，读入后转 0/1）
- dunhuang_visual/ 下的彩色图只用于人工检查，不要当 mask 读入训练。
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 说明文档已更新: {readme_path}")


# ===================== 主处理函数 =====================
def sam3_batch_segment_3colors():
    # 创建所有必要的文件夹
    for folder in [CROPPED_FOLDER, MASK_OUTPUT_FOLDER, HOLE_OUTPUT_FOLDER, VIS_OUTPUT_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    print("初始化SAM3模型...")
    model = build_sam3_image_model(
        device=DEVICE,
        checkpoint_path="./sam3/sam3.pt",
        load_from_HF=False,
        compile=False,
        enable_inst_interactivity=True
    )
    model.to(DEVICE)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)

    img_list = [x for x in os.listdir(RAW_IMG_FOLDER) if x.lower().endswith((".jpg", ".png", ".jpeg"))]
    total_imgs = len(img_list)

    with torch.inference_mode():
        for idx, img_name in enumerate(img_list):
            img_path = os.path.join(RAW_IMG_FOLDER, img_name)
            print(f"[{idx+1}/{total_imgs}] 处理图片：{img_name}")

            try:
                img = Image.open(img_path).convert("RGB")
                img_rgb = np.array(img)
                img_768 = resize_stretch_768(img_rgb)

                # 保存 GT
                cv2.imwrite(
                    os.path.join(CROPPED_FOLDER, img_name),
                    cv2.cvtColor(img_768, cv2.COLOR_RGB2BGR)
                )

                final_visual = img_768.copy()
                img_with_hole = img_768.copy()

                # ✅ 新增：并集洞 mask（和 input_with_hole 对齐）
                union_mask_bool = np.zeros((768, 768), dtype=bool)

                pil_img_768 = Image.fromarray(img_768)
                inference_state = processor.set_image(pil_img_768)

                for color_info in TARGET_COLORS:
                    eng_name = color_info['eng_name']
                    chn_name = color_info['name']

                    # --- 1. SAM 分割 ---
                    auto_box = rgb2lab_and_find_single_color(img_768, color_info)
                    x_c, y_c = (auto_box[0] + auto_box[2]) // 2, (auto_box[1] + auto_box[3]) // 2

                    inference_state = processor.set_text_prompt(
                        state=inference_state,
                        prompt=f"壁画中的{chn_name}颜色区域"
                    )

                    masks, scores, _ = model.predict_inst(
                        inference_state=inference_state,
                        point_coords=np.array([[x_c, y_c]]),
                        point_labels=np.array([1]),
                        multimask_output=True
                    )

                    if isinstance(scores, torch.Tensor):
                        scores_np = scores.detach().cpu().numpy()
                        masks_np = masks.detach().cpu().numpy()
                    else:
                        scores_np = scores
                        masks_np = masks

                    best_idx = scores_np.argmax()
                    best_mask_raw = masks_np[best_idx].squeeze() if masks_np[best_idx].ndim == 3 else masks_np[best_idx]

                    # --- 2. 面积检查与策略选择 ---
                    h, w = best_mask_raw.shape
                    total_pixels = h * w
                    limit_pixels = int(total_pixels * MASK_AREA_LIMIT)

                    current_mask_bool = best_mask_raw > 0
                    current_count = np.count_nonzero(current_mask_bool)

                    is_random_mask = False

                    # 情况A: SAM 没检测到（全黑）
                    if current_count == 0:
                        final_mask_bool = generate_random_brush_mask(h, w)
                        is_random_mask = True

                    # 情况B: 面积太大 -> 截断
                    elif current_count > limit_pixels:
                        flat_scores = best_mask_raw.reshape(-1)
                        sorted_scores = np.sort(flat_scores)[::-1]
                        new_threshold = sorted_scores[limit_pixels]
                        final_mask_bool = best_mask_raw > new_threshold

                    # 情况C: 正常
                    else:
                        final_mask_bool = current_mask_bool

                    # ✅ 并集：无论 SAM 还是 RANDOM，都加入 union（因为洞图也会挖空）
                    union_mask_bool |= final_mask_bool

                    # --- 3. 保存单色 mask（黑白）---
                    tag = "RANDOM" if is_random_mask else "SAM"
                    mask_filename = f"mask_{eng_name}_{tag}_{os.path.splitext(img_name)[0]}.png"
                    mask_gray = (final_mask_bool * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, mask_filename), mask_gray)

                    # --- 4. 可视化叠加（只对 SAM 真实检测到的上色；随机破损不染色）---
                    if (not is_random_mask) and np.any(final_mask_bool):
                        final_visual[final_mask_bool] = (
                            final_visual[final_mask_bool] * 0.6 +
                            np.array(color_info["visual_color"]) * 0.4
                        )

                # --- 5. 保存并集 mask（训练建议用这个）---
                union_filename = f"mask_union_{os.path.splitext(img_name)[0]}.png"
                union_gray = (union_mask_bool * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(MASK_OUTPUT_FOLDER, union_filename), union_gray)

                # --- 6. 保存挖空图（严格按 union mask 挖空）---
                if np.any(union_mask_bool):
                    img_with_hole[union_mask_bool] = [0, 0, 0]
                hole_filename = f"input_with_hole_{os.path.splitext(img_name)[0]}.png"
                cv2.imwrite(os.path.join(HOLE_OUTPUT_FOLDER, hole_filename),
                            cv2.cvtColor(img_with_hole, cv2.COLOR_RGB2BGR))

                # --- 7. 保存可视化图（移到独立文件夹，避免训练读错）---
                visual_filename = f"visual_overlay_{os.path.splitext(img_name)[0]}.png"
                cv2.imwrite(os.path.join(VIS_OUTPUT_FOLDER, visual_filename),
                            cv2.cvtColor(final_visual, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"❌ 处理出错 {img_name}: {e}")
                import traceback
                traceback.print_exc()

    generate_chinese_readme()


# ===================== 运行 =====================
if __name__ == "__main__":
    sam3_batch_segment_3colors()
    print("\n✅ 处理完成！已生成：mask_union + input_with_hole + visual_overlay(独立文件夹)")
