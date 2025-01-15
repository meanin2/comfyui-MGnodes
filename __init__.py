from .nodes.egtjtxsy import EGCPSYTJNode

NODE_CLASS_MAPPINGS = {
    "EG_CPSYTJ": EGCPSYTJNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EG_CPSYTJ": "2🐕 Add finished watermark image",
}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS'
]
