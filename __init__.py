from .nodes.watermark_overlay import WatermarkOverlayNode

NODE_CLASS_MAPPINGS = {
    "WatermarkOverlay": WatermarkOverlayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkOverlay": "Image Watermark Overlay",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]

