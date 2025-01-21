from .nodes.image_watermark_node import ImageWatermarkNode
from .nodes.chain_of_thought_processor_node import ChainOfThoughtProcessorNode

NODE_CLASS_MAPPINGS = {
    "ImageWatermarkNode": ImageWatermarkNode,
    "ChainOfThoughtProcessorNode": ChainOfThoughtProcessorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWatermarkNode": "Add Image Watermark",
    "ChainOfThoughtProcessorNode": "Chain of Thought Processor"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS"
]
