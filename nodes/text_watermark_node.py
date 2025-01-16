import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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

    def add_watermark(self, image, text, position=0, opacity_percentage=50, 
                     scale_percentage=100, text_color="white", text_size=32, font_style="arial"):
        # Convert the tensor to numpy array properly
        img = image.squeeze(0).cpu().numpy()  # Remove batch dimension
        img = (img * 255).astype(np.uint8)
        
        # Handle image channels properly
        if img.shape[2] == 3:  # RGB
            img = Image.fromarray(img, 'RGB')
        elif img.shape[2] == 4:  # RGBA
            img = Image.fromarray(img, 'RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")

        # Convert percentages to float
        opacity = opacity_percentage / 100.0
        scale = scale_percentage / 100.0

        # Position mapping
        positions = [
            "center",    # 0
            "top-left",  # 1
            "top-right", # 2
            "bottom-left", # 3
            "bottom-right", # 4
            "tiled"      # 5
        ]
        internal_position = positions[position]

        # Create text watermark
        try:
            font = ImageFont.truetype(f"{font_style}.ttf", text_size)
        except:
            font = ImageFont.load_default()
            
        # Create temporary image to calculate text size
        temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        try:
            # Try newer Pillow method first
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fall back to older method
            text_width, text_height = temp_draw.textsize(text, font=font)
        
        # Create watermark image with text
        padding = 2
        watermark = Image.new('RGBA', (text_width + padding*2, text_height + padding*2), (0,0,0,0))
        draw = ImageDraw.Draw(watermark)
        draw.text((padding, padding), text, fill=text_color, font=font)

        # Scale watermark
        if scale != 1.0:
            new_size = (int(watermark.width * scale), int(watermark.height * scale))
            watermark = watermark.resize(new_size, Image.LANCZOS)

        # Apply opacity
        if watermark.mode == 'RGBA':
            r, g, b, a = watermark.split()
            new_alpha = Image.fromarray((np.array(a) * opacity).astype(np.uint8))
            watermark.putalpha(new_alpha)
        else:
            opacity_mask = Image.new('L', watermark.size, int(opacity * 255))
            watermark.putalpha(opacity_mask)

        # Handle tiling
        if internal_position == "tiled":
            tiled = Image.new('RGBA', img.size, (0,0,0,0))
            for y in range(0, img.size[1], watermark.size[1]):
                for x in range(0, img.size[0], watermark.size[0]):
                    tiled.paste(watermark, (x, y), watermark)
            watermark = tiled
        else:
            # Calculate position
            if internal_position == "center":
                x = (img.size[0] - watermark.size[0]) // 2
                y = (img.size[1] - watermark.size[1]) // 2
            elif internal_position == "top-left":
                x, y = 0, 0
            elif internal_position == "top-right":
                x = img.size[0] - watermark.size[0]
                y = 0
            elif internal_position == "bottom-left":
                x = 0
                y = img.size[1] - watermark.size[1]
            elif internal_position == "bottom-right":
                x = img.size[0] - watermark.size[0]
                y = img.size[1] - watermark.size[1]

            # Create a full-size transparent image
            full_watermark = Image.new('RGBA', img.size, (0,0,0,0))
            full_watermark.paste(watermark, (x, y), watermark)
            watermark = full_watermark

        # Convert base image to RGBA if it isn't already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

        # Composite the watermark onto the original image
        img = Image.alpha_composite(img, watermark)

        # Convert back to RGB and then to numpy array
        img = img.convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, 0)  # Add batch dimension back
        
        return (torch.from_numpy(img_array),) 