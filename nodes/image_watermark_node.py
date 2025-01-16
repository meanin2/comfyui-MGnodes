import numpy as np
import torch
from PIL import Image

class ImageWatermarkNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_image": ("IMAGE",),
                # Revert to COMBO but remove "index": True. Provide a label for clarity.
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
                    "label": "Position"
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

    def add_watermark(self,
                      image,
                      watermark_image,
                      position="Center",
                      opacity_percentage=50,
                      scale_percentage=100):
        """
        Add a watermark with transparency onto an image.
        """

        # Convert main image to numpy (0..255)
        img_np = image.squeeze(0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Convert to PIL and ensure RGBA
        if img_np.shape[2] == 3:
            base_img = Image.fromarray(img_np, mode='RGB').convert('RGBA')
        elif img_np.shape[2] == 4:
            base_img = Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels for main image: {img_np.shape[2]}")

        # Convert watermark to numpy (0..255)
        wmk_np = watermark_image.squeeze(0).cpu().numpy()
        wmk_np = (wmk_np * 255).astype(np.uint8)

        # Convert to PIL and ensure RGBA
        if wmk_np.shape[2] == 3:
            wmk = Image.fromarray(wmk_np, mode='RGB').convert('RGBA')
        elif wmk_np.shape[2] == 4:
            wmk = Image.fromarray(wmk_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels for watermark: {wmk_np.shape[2]}")

        # Scale
        scale = scale_percentage / 100.0
        if scale != 1.0:
            new_size = (int(wmk.width * scale), int(wmk.height * scale))
            wmk = wmk.resize(new_size, Image.LANCZOS)

        # Apply user opacity to watermark’s existing alpha
        opacity = opacity_percentage / 100.0
        wmk_data = np.array(wmk, dtype=np.uint8)
        # Multiply existing alpha by desired opacity
        wmk_data[..., 3] = (wmk_data[..., 3].astype(float) * opacity).astype(np.uint8)
        wmk = Image.fromarray(wmk_data, mode='RGBA')

        # Create an RGBA overlay
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))

        # Mapping for position combos
        pos_map = [
            "Center",
            "Top Left",
            "Top Right",
            "Bottom Left",
            "Bottom Right",
            "Tiled Pattern"
        ]
        if position not in pos_map:
            position = "Center"  # fallback

        # Handle tiled pattern
        if position == "Tiled Pattern":
            for y in range(0, overlay.size[1], wmk.height):
                for x in range(0, overlay.size[0], wmk.width):
                    # Use watermark itself as the mask to maintain transparency
                    overlay.paste(wmk, (x, y), wmk)
        else:
            # Calculate single position
            if position == "Center":
                x = (overlay.width - wmk.width) // 2
                y = (overlay.height - wmk.height) // 2
            elif position == "Top Left":
                x, y = 0, 0
            elif position == "Top Right":
                x = overlay.width - wmk.width
                y = 0
            elif position == "Bottom Left":
                x, y = 0, overlay.height - wmk.height
            elif position == "Bottom Right":
                x = overlay.width - wmk.width
                y = overlay.height - wmk.height
            else:
                x = (overlay.width - wmk.width) // 2
                y = (overlay.height - wmk.height) // 2

            overlay.paste(wmk, (x, y), wmk)

        # Combine overlay with base image
        final_img = Image.alpha_composite(base_img, overlay)

        # Convert back to float32 [0..1]
        final_np = np.array(final_img, dtype=np.float32) / 255.0

        # If original had only RGB channels, drop alpha
        if image.shape[1] == 3:
            final_np = final_np[..., :3]

        # Add batch dimension back for ComfyUI
        final_np = np.expand_dims(final_np, 0)

        return (torch.from_numpy(final_np),)
