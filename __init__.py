from .nodes.image_watermark_node import ImageWatermarkNode
from .nodes.text_watermark_node import TextWatermarkNode

NODE_CLASS_MAPPINGS = {
    "ImageWatermarkNode": ImageWatermarkNode,
    "TextWatermarkNode": TextWatermarkNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWatermarkNode": "Add Image Watermark",
    "TextWatermarkNode": "Add Text Watermark"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
