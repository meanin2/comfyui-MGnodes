# __init__.py

from .nodes.image_watermark_node import ImageWatermarkNode
from .nodes.chain_of_thought_processor_node import ChainOfThoughtProcessorNode

NODE_CONFIG = {
    # Image Processing Nodes
    "ImageWatermarkNode": {
        "class": ImageWatermarkNode,
        "name": "Add Image Watermark",
        "category": "MGnodes/Image",
        "description": "Overlay a watermark image onto a base image with customizable transparency, position, and color settings.",
        "keywords": ["watermark", "image overlay", "transparency", "positioning", "color manipulation", "resize", "invert colors"]
    },
    
    # Text Processing Nodes
    "TextExtractorNode": {
        "class": ChainOfThoughtProcessorNode,
        "name": "Text Extractor Node",
        "category": "MGnodes/Text",
        "description": "Extract and separate specific sections of text marked within defined tags, enabling advanced text analysis and manipulation.",
        "keywords": ["text processing", "split", "extract", "regex", "content separation", "ollama", "llm", "text analysis"]
    },
    
    # Add more nodes here as you develop them
}

def generate_node_mappings(node_config):
    """
    Generates mappings for node classes, display names, categories, and keywords.
    """
    node_class_mappings = {}
    node_display_name_mappings = {}
    node_category_mappings = {}
    node_keyword_mappings = {}

    for node_name, node_info in node_config.items():
        node_class_mappings[node_name] = node_info["class"]
        node_display_name_mappings[node_name] = node_info.get("name", node_info["class"].__name__)
        node_category_mappings[node_name] = node_info.get("category", "Uncategorized")
        node_keyword_mappings[node_name] = node_info.get("keywords", [])

    return node_class_mappings, node_display_name_mappings, node_category_mappings, node_keyword_mappings

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, NODE_CATEGORY_MAPPINGS, NODE_KEYWORD_MAPPINGS = generate_node_mappings(NODE_CONFIG)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "NODE_CATEGORY_MAPPINGS", "NODE_KEYWORD_MAPPINGS"]
