import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

class ImageWatermarkNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                "position": ("COMBO", {
                    "choices": [
                        "Center",
                        "Top Left",
                        "Top Right",
                        "Bottom Left",
                        "Bottom Right",
                        "Tiled Pattern"
                    ],
                    "default": "Center",
                    "index": True
                }),
                "opacity_percentage": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "label": "Opacity %"
                }),
                "scale_percentage": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 5,
                    "display": "slider",
                    "label": "Size %"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_watermark"
    CATEGORY = "image/watermark"

class TextWatermarkNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {
                    "default": "Watermark",
                    "multiline": True,
                    "placeholder": "Enter watermark text here..."
                }),
                "position": ("COMBO", {
                    "choices": [
                        "Center",
                        "Top Left",
                        "Top Right",
                        "Bottom Left",
                        "Bottom Right",
                        "Tiled Pattern"
                    ],
                    "default": "Center",
                    "index": True
                }),
                "opacity_percentage": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "label": "Opacity %"
                }),
                "scale_percentage": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 5,
                    "display": "slider",
                    "label": "Size %"
                }),
                "text_color": ("STRING", {
                    "default": "white",
                    "placeholder": "white, black, red, #FF0000, etc"
                }),
                "text_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "label": "Text Size (px)"
                }),
                "font_style": ("COMBO", {
                    "choices": ["arial", "times", "courier"],
                    "default": "arial"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_watermark"
    CATEGORY = "image/watermark"
