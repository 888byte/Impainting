# sd_inpaint_mural.py
"""
基于 diffusers 的 Stable Diffusion inpainting 测试脚本。

前提：
  - 已安装 diffusers 等依赖；
  - 已准备好 datasets/mural_inpaint/images & masks。

运行：
  python sd_inpaint_mural.py
"""

import os
import glob
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

DATA_ROOT = "./datasets/mural_inpaint"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR  = os.path.join(DATA_ROOT, "masks")
OUT_DIR   = "./results/sd_mural"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(path, size=(512, 512), mode="RGB"):
    img = Image.open(path).convert(mode)
    if size is not None:
        if mode == "RGB":
            img = img.resize(size, Image.BICUBIC)
        else:
            img = img.resize(size, Image.NEAREST)
    return img

def main():
    print("加载 Stable Diffusion inpainting 模型...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 关闭安全检查（纯学术用）
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"发现 {len(image_paths)} 张图像。")

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(MASK_DIR, f"{name}_mask.png")
        if not os.path.exists(mask_path):
            print("无 mask，跳过：", img_path)
            continue

        image = load_image(img_path, size=(512, 512), mode="RGB")
        mask  = load_image(mask_path, size=(512, 512), mode="L")

        prompt = "ancient mural, fresco, realistic restoration"
        generator = torch.Generator(device=device).manual_seed(42)

        if device == "cuda":
            with torch.autocast(device_type="cuda"):
                result = pipe(
                    prompt=prompt,
                    image=image,
                    mask_image=mask,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]
        else:
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator=generator,
            ).images[0]

        out_path = os.path.join(OUT_DIR, f"{name}.png")
        result.save(out_path)
        print("保存：", out_path)

if __name__ == "__main__":
    main()
