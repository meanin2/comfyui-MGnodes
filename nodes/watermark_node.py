from PIL import Image, ImageDraw, ImageFont
import numpy as np

class WatermarkNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE", {}),
            },
            "optional": {
                "watermark_image": ("IMAGE", {}),
                "watermark_text": ("STRING", {"default": ""}),
                "opacity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "position": ("COMBO", {"options": ["center", "top-left", "top-right", "bottom-left", "bottom-right", "tiled"], "default": "center"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
                "repeat": ("BOOL", {"default": False}),
                "text_color": ("STRING", {"default": "white"}),
                "text_size": ("INT", {"default": 30}),
                "font_style": ("COMBO", {"options": ["Arial", "Courier", "Times"], "default": "Arial"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_watermark"
    CATEGORY = "Custom Nodes"

    def apply_watermark(self, base_image, watermark_image=None, watermark_text="", opacity=0.5, position="center", scale=1.0, repeat=False, text_color="white", text_size=30, font_style="Arial"):
        # Validate inputs
        if base_image is None:
            self.raise_error("Base image is missing or invalid.")
        if not watermark_text and watermark_image is None:
            self.raise_error("Either a watermark image or watermark text must be provided.")
        if not (0.0 <= opacity <= 1.0):
            self.raise_error("Opacity must be between 0.0 and 1.0.")
        if scale <= 0:
            self.raise_error("Scale must be greater than 0.")

        # Load base image
        base = Image.fromarray(base_image)

        # Prepare watermark
        overlay = self.prepare_watermark(watermark_image, watermark_text, text_color, text_size, font_style, scale, opacity)

        # Apply watermark
        if position == "tiled":
            tiled_overlay = self.tile_overlay(base, overlay)
            base.paste(tiled_overlay, (0, 0), tiled_overlay)
        else:
            x, y = self.get_position(base, overlay, position)
            base.paste(overlay, (x, y), overlay)

        return (np.array(base),)

    def prepare_watermark(self, watermark_image, watermark_text, text_color, text_size, font_style, scale, opacity):
        if watermark_text:
            try:
                font = ImageFont.truetype(f"{font_style}.ttf", text_size)
            except IOError:
                self.raise_error(f"Font '{font_style}.ttf' not found.")
            text_size = font.getsize(watermark_text)
            overlay = Image.new("RGBA", text_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.text((0, 0), watermark_text, fill=text_color, font=font)
        elif watermark_image:
            overlay = Image.fromarray(watermark_image).convert("RGBA")
        else:
            self.raise_error("No watermark provided (text or image).")
        
        # Scale and adjust transparency
        overlay = overlay.resize((int(overlay.width * scale), int(overlay.height * scale)))
        overlay.putalpha(int(opacity * 255))
        return overlay

    def tile_overlay(self, base, overlay):
        tiled = Image.new("RGBA", base.size)
        for y in range(0, base.height, overlay.height):
            for x in range(0, base.width, overlay.width):
                tiled.paste(overlay, (x, y), overlay)
        return tiled

    def get_position(self, base, overlay, position):
        x, y = 0, 0
        if position == "center":
            x = (base.width - overlay.width) // 2
            y = (base.height - overlay.height) // 2
        elif position == "top-left":
            x, y = 0, 0
        elif position == "top-right":
            x = base.width - overlay.width
            y = 0
        elif position == "bottom-left":
            x = 0
            y = base.height - overlay.height
        elif position == "bottom-right":
            x = base.width - overlay.width
            y = base.height - overlay.height
        return x, y

    def raise_error(self, message):
        raise ValueError(f"WatermarkNode Error: {message}")
