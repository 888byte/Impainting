"""
模型工厂：根据配置中的 model 字段创建对应模型类。
"""

import logging

logger = logging.getLogger("base")


def create_model(opt):
    """创建并返回模型实例。"""
    model = opt["model"]

    if model == "denoising":
        from .denoising_model import DenoisingModel as M
    else:
        raise NotImplementedError("Model [{:s}] not recognized.".format(model))
    m = M(opt)
    logger.info("Model [{:s}] is created.".format(m.__class__.__name__))
    return m
