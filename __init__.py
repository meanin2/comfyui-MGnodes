from .nodes.image_watermark_node import ImageWatermarkNode
from .nodes.text_extractor_node import TextExtractorNode

NODE_CLASS_MAPPINGS = {
    # Unique keys matching the class names in your .py files
    "ImageWatermarkNode": ImageWatermarkNode,
    "TextExtractorNode": TextExtractorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Public-facing names in the ComfyUI “Add Node” menu
    "ImageWatermarkNode": "Add Image Watermark",
    "TextExtractorNode": "Text Extractor Node",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
