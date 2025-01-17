import numpy as np
import torch
from PIL import Image

class ImageWatermarkNode:
    """
    This node adds a watermark image onto an input image, with adjustable
    position (Center, Top Left, Top Right, Bottom Left, Bottom Right, Tiled),
    opacity, and scale. It preserves any existing transparency in the base image.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "watermark_image": ("IMAGE", {}),
                "watermark_mask": ("MASK", {}),
                # NOTE: For ComfyUI combos, we must provide a list of strings 
                # rather than ("COMBO", {...}).
                "position": (
                    [
                        "Center",
                        "Top Left",
                        "Top Right",
                        "Bottom Left",
                        "Bottom Right",
                        "Tiled Pattern",
                    ],
                    {
                        "default": "Center",
                    }
                ),
                "opacity_percentage": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
                "scale_percentage": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 5,
                    "display": "slider"
                }),
                "make_black": ("BOOLEAN", {
                    "default": False,
                    "label": "Make Watermark Black"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_watermark"
    CATEGORY = "image/watermark"
    # Optionally:
    # RETURN_NAMES = ("watermarked_image",)

    def add_watermark(self,
                      image,
                      watermark_image,
                      watermark_mask,
                      position="Center",
                      opacity_percentage=50,
                      scale_percentage=100,
                      make_black=False,
                      **kwargs):
        """
        Add a watermark with transparency onto an image. Preserves existing transparency in the base image.
        """
        # Debug any unexpected arguments from the UI
        if kwargs:
            print(f"Unexpected arguments: {kwargs}")

        # Convert main image from [1, H, W, C] float32 [0..1] to a Pillow Image
        img_np = image.squeeze(0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Convert base image to RGBA
        if img_np.shape[2] == 3:
            base_img = Image.fromarray(img_np, mode='RGB').convert('RGBA')
        elif img_np.shape[2] == 4:
            base_img = Image.fromarray(img_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported number of channels for main image: {img_np.shape[2]}")

        # -------------------------------------
        # FIX: Composite base image onto a blank RGBA to preserve transparency
        # -------------------------------------
        blank_bg = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        base_img = Image.alpha_composite(blank_bg, base_img)

        # Convert watermark image from [1, H, W, C] float32 [0..1]
        wmk_np = watermark_image.squeeze(0).cpu().numpy()
        wmk_np = (wmk_np * 255).astype(np.uint8)

        # Convert watermark to RGB (not RGBA since we'll add alpha from mask)
        wmk = Image.fromarray(wmk_np, mode='RGB')
        
        # Apply black filter if requested
        if make_black:
            # Convert to grayscale then back to RGB to preserve image structure
            wmk = wmk.convert('L').convert('RGB')
            # Create solid black image
            black = Image.new('RGB', wmk.size, (0, 0, 0))
            # Use the grayscale image as opacity to blend with black
            wmk = Image.blend(black, wmk, 0)  # 0 means full black

        # Convert mask to numpy and prepare alpha channel
        mask_np = watermark_mask.squeeze().cpu().numpy()
        # Invert the mask since ComfyUI's masks are typically white for areas to keep
        mask_np = 255 - (mask_np * 255).astype(np.uint8)
        mask = Image.fromarray(mask_np, mode='L')

        # Add the mask as alpha channel
        wmk.putalpha(mask)

        # Scale watermark
        scale = scale_percentage / 100.0
        if scale != 1.0:
            new_size = (int(wmk.width * scale), int(wmk.height * scale))
            wmk = wmk.resize(new_size, Image.LANCZOS)

        # Apply user opacity to watermark’s existing alpha
        opacity = opacity_percentage / 100.0
        wmk_data = np.array(wmk, dtype=np.uint8)
        wmk_data[..., 3] = (wmk_data[..., 3].astype(float) * opacity).astype(np.uint8)
        wmk = Image.fromarray(wmk_data, mode='RGBA')

        # Create an empty RGBA overlay
        overlay = Image.new('RGBA', base_img.size, (0, 0, 0, 0))

        # Validate/fallback position
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
                    overlay.paste(wmk, (x, y), wmk)
        else:
            # Single placement
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

        # Alpha composite the overlay onto the base image
        final_img = Image.alpha_composite(base_img, overlay)

        # Convert back to float32 [0..1]
        final_np = np.array(final_img, dtype=np.float32) / 255.0

        # If original had only RGB channels, drop alpha in the final
        # Otherwise (4 channels), we keep alpha
        if image.shape[1] == 3:
            final_np = final_np[..., :3]

        # Add batch dimension again for ComfyUI
        final_np = np.expand_dims(final_np, 0)

        # Return as a PyTorch tensor
        return (torch.from_numpy(final_np),)
