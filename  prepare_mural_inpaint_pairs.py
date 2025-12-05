# prepare_mural_inpaint_pairs.py
"""
用途：
  1. 从你已有的原图目录 & 掩膜目录，整理出统一的 image / mask 对；
  2. 统一 resize 到 target_size（默认 512×512）；
  3. 确保掩膜是二值图（白=255=修复区域，黑=0=保持原样）。

用法：
  python prepare_mural_inpaint_pairs.py
"""

import os
import glob
from PIL import Image

# === 你需要改的三个路径 ===
RAW_IMAGE_DIR = r"./raw_images"   # 原始壁画/样片图像目录
RAW_MASK_DIR  = r"./raw_masks"    # 对应掩膜目录
OUT_ROOT      = r"./datasets/mural_inpaint"

TARGET_SIZE = (512, 512)          # SD/LDM 推荐 512×512

def find_mask_for_image(img_path: str) -> str | None:
    """
    根据 image 文件名去找 mask。
    这里假设 mask 命名为 name_mask.png，你可以按实际情况改。
    """
    name = os.path.splitext(os.path.basename(img_path))[0]
    # 可以按自己数据改，比如 name.png -> name_mask.png 或 name.png -> name.png
    cand1 = os.path.join(RAW_MASK_DIR, name + "_mask.png")
    cand2 = os.path.join(RAW_MASK_DIR, name + ".png")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    return None

def main():
    out_img_dir = os.path.join(OUT_ROOT, "images")
    out_msk_dir = os.path.join(OUT_ROOT, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_msk_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(RAW_IMAGE_DIR, "*.*")))
    print(f"发现 {len(img_paths)} 张原始图像")

    cnt = 0
    for img_path in img_paths:
        mask_path = find_mask_for_image(img_path)
        if mask_path is None:
            print("警告：没有掩膜，跳过", img_path)
            continue

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = img.resize(TARGET_SIZE, Image.BICUBIC)
        mask = mask.resize(TARGET_SIZE, Image.NEAREST)

        # 二值化：>0 为 255（白）
        mask = mask.point(lambda v: 255 if v > 0 else 0)

        name = os.path.splitext(os.path.basename(img_path))[0]

        img.save(os.path.join(out_img_dir, f"{name}.png"))
        mask.save(os.path.join(out_msk_dir, f"{name}_mask.png"))
        cnt += 1

    print(f"整理完成，共生成 {cnt} 对 image/mask。输出目录：{OUT_ROOT}")

if __name__ == "__main__":
    main()
