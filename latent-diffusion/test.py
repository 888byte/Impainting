# ldm_inpaint_mural.py
"""
基于 CompVis/latent-diffusion 的 inpainting 测试脚本。

使用方法：
  1. 在 latent-diffusion 仓库根目录放这个文件；
  2. 确保 models/ldm/inpainting_big/config.yaml 和 last.ckpt 存在；
  3. 确保 datasets/mural_inpaint 已准备好；
  4. 运行：python ldm_inpaint_mural.py
"""

import os
import glob
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
import torch.nn.functional as F

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

DATA_ROOT = "./datasets/mural_inpaint"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR  = os.path.join(DATA_ROOT, "masks")
OUT_DIR   = "./results/ldm_mural"

os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_PATH = "models/ldm/inpainting_big/config.yaml"
CKPT_PATH   = "models/ldm/inpainting_big/last.ckpt"

def load_image_mask(image_path: str, mask_path: str, device: torch.device):
    """
    读取一张 image/mask，并打包成 LDM 需要的 batch 格式：
      - image: [1,3,H,W] in [-1,1]
      - mask:  [1,1,H,W] in [-1,1] (1=修复，0=保持)
    """
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    img = img.resize((512, 512), Image.BICUBIC)
    mask = mask.resize((512, 512), Image.NEAREST)

    img = np.array(img).astype(np.float32) / 255.0
    img = img[None].transpose(0, 3, 1, 2)  # NCHW
    img = torch.from_numpy(img)

    mask = np.array(mask).astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = mask[None, None, :, :]  # NCHW
    mask = torch.from_numpy(mask)

    image = img.to(device)
    mask  = mask.to(device)

    # [-1, 1] 范围
    image = image * 2.0 - 1.0
    mask  = mask * 2.0 - 1.0

    batch = {
        "image": image,
        "mask": mask,
    }
    # masked_image：非掩膜区域保持原样，掩膜区域置 0
    batch["masked_image"] = (1.0 - (batch["mask"] + 1.0) / 2.0) * ((batch["image"] + 1.0) / 2.0)
    batch["masked_image"] = batch["masked_image"] * 2.0 - 1.0

    return batch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("加载 LDM 配置和模型权重...")
    config = OmegaConf.load(CONFIG_PATH)
    model  = instantiate_from_config(config.model)
    ckpt   = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    sampler = DDIMSampler(model)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"发现 {len(image_paths)} 张图像。")

    with torch.no_grad():
        with model.ema_scope():
            for img_path in tqdm(image_paths):
                name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(MASK_DIR, f"{name}_mask.png")
                if not os.path.exists(mask_path):
                    print("无 mask，跳过：", img_path)
                    continue

                batch = load_image_mask(img_path, mask_path, device)

                # 条件编码：masked_image + 下采样 mask
                # cond_stage_model.encode 的接口可能随版本略有不同，如果报错请按 README 调整
                c = model.cond_stage_model.encode(batch["masked_image"])
                if isinstance(c, tuple):  # 兼容某些返回 (mu, logvar) 的情况
                    c = c[0]

                mask = (batch["mask"] + 1.0) / 2.0   # [0,1]
                mask_down = F.interpolate(mask, size=c.shape[-2:], mode="nearest")
                c_cat = torch.cat([c, mask_down], dim=1)

                shape = (c_cat.shape[1] - 1, c_cat.shape[2], c_cat.shape[3])

                samples, _ = sampler.sample(
                    S=50,
                    conditioning=c_cat,
                    batch_size=1,
                    shape=shape,
                    verbose=False,
                )

                x_rec = model.decode_first_stage(samples)

                # 组合：非掩膜区域用原图，掩膜区域用预测结果
                image = (batch["image"] + 1.0) / 2.0
                mask_full = (batch["mask"] + 1.0) / 2.0  # [0,1]
                pred  = (x_rec + 1.0) / 2.0

                inpainted = (1.0 - mask_full) * image + mask_full * pred
                inpainted = inpainted.clamp(0.0, 1.0)

                inpainted_np = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255.0
                inpainted_np = inpainted_np.astype("uint8")

                out_path = os.path.join(OUT_DIR, f"{name}.png")
                Image.fromarray(inpainted_np).save(out_path)

if __name__ == "__main__":
    main()
