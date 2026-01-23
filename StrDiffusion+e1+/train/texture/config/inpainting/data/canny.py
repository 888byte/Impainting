"""
基于 Canny 的边缘/灰度提取工具。
输入为 tensor 图像，输出边缘图（edge）和灰度图（gray）。
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


def image_to_edge(image, sigma):
    """从 RGB tensor 提取边缘与灰度图。"""

    gray_image = rgb2gray(np.array(tensor_to_image()(image)))
    edge = image_to_tensor()(Image.fromarray(canny(gray_image, sigma=sigma)))
    gray_image = image_to_tensor()(Image.fromarray(gray_image))

    return edge, gray_image

