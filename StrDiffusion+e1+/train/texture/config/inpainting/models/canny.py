"""
模型侧的 Canny 边缘提取工具（输入为 tensor）。
"""

import numpy as np
from PIL import Image

from torchvision import transforms

from skimage.feature import canny
from skimage.color import gray2rgb, rgb2gray


def tensor_to_image():
    """tensor -> PIL.Image"""
    return transforms.ToPILImage()


def image_to_tensor():
    """PIL.Image -> tensor"""
    return transforms.ToTensor()


def gray_to_edge(image, sigma):
    """将灰度 tensor 转为边缘图 tensor。"""

    gray_image = np.array(tensor_to_image()(image))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))

    return edge

