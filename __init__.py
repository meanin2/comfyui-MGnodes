from .nodes.image_watermark_node import ImageWatermarkNode

NODE_CLASS_MAPPINGS = {
    "ImageWatermarkNode": ImageWatermarkNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWatermarkNode": "Add Image Watermark",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
