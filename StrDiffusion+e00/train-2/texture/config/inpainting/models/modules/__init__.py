from .DenoisingUNet_arch import ConditionalUNet

# BrushNet集成模型（新增，不影响原有代码）
try:
    from ..brushnet_wrapper import ConditionalUNetWithBrushNet
except ImportError:
    # 如果导入失败，提供空占位（保持向后兼容）
    ConditionalUNetWithBrushNet = None