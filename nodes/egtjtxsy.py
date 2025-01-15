import os
import numpy as np
import torch
import sys
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# If ComfyUI's node-based MAX_RESOLUTION is not available, define our own:
MAX_RESOLUTION = 8192

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a PyTorch float tensor image to a PIL Image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch float tensor image."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def bislerp(samples, Scale_width, Scale_height):
    """A simple bilinear or bicubic fallback if needed.
       Here we’ll just do 'bilinear' for demonstration."""
    return torch.nn.functional.interpolate(
        samples, 
        size=(Scale_height, Scale_width), 
        mode="bilinear", 
        align_corners=True
    )

def lanczos(samples, Scale_width, Scale_height):
    """Optional custom function if you want Lanczos downsampling. 
       For demonstration, we’ll just do 'bilinear' again."""
    return torch.nn.functional.interpolate(
        samples, 
        size=(Scale_height, Scale_width), 
        mode="bilinear", 
        align_corners=True
    )

def common_upscale(samples, Scale_width, Scale_height, upscale_method, crop):
    """
    A helper function to handle different scaling methods and optional center-cropping.
    """
    if crop == "center":
        old_Scale_width = samples.shape[3]
        old_Scale_height = samples.shape[2]
        old_aspect = old_Scale_width / old_Scale_height
        new_aspect = Scale_width / Scale_height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_Scale_width - old_Scale_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_Scale_height - old_Scale_height * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y:old_Scale_height-y, x:old_Scale_width-x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, Scale_width, Scale_height)
    elif upscale_method == "lanczos":
        return lanczos(s, Scale_width, Scale_height)
    else:
        return torch.nn.functional.interpolate(s, size=(Scale_height, Scale_width), mode=upscale_method)

class EGCPSYTJNode:
    """
    Watermark Image Node.
    Overlays a 'Watermark_image' on top of an 'original_image'.
    Allows scaling, rotation, transparency, and optional masking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "Watermark_image": ("IMAGE",),
                "Zoom_mode": (["None", "Fit", "zoom", "Scale_according_to_input_width_and_height"],),
                "Scaling_method": (["nearest-exact", "bilinear", "area", "bislerp", "lanczos"],),
                "Scaling_factor": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.1}),
                "Scale_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "Scale_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "initial_position": (
                    [
                        "Centered", 
                        "Up", 
                        "Down", 
                        "Left", 
                        "Right", 
                        "Up Left", 
                        "Up Right", 
                        "Down Left", 
                        "Down Right"
                    ],
                ),
                "X_direction": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "Y_direction": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "rotate": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "transparency": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5, "display": "slider"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_Watermark_image"
    CATEGORY = "2🐕/🔖Watermark addition"

    def apply_Watermark_image(
        self, 
        original_image, 
        Watermark_image, 
        Zoom_mode, 
        Scaling_method, 
        Scaling_factor,
        Scale_width, 
        Scale_height, 
        X_direction, 
        Y_direction, 
        rotate, 
        transparency, 
        initial_position,
        mask=None
    ):
        """
        Overlays the Watermark_image onto the original_image using the specified parameters.
        """

        # Decide final size (if needed by Scale_according_to_input_width_and_height)
        size = (Scale_width, Scale_height)
        location = (X_direction, Y_direction)

        # Handle zoom/scale modes
        if Zoom_mode != "None":
            # Watermark_image.size() => shape: (batch, height, width, channels). We want (width, height).
            wm_w, wm_h = Watermark_image.size()[2], Watermark_image.size()[1]
            if Zoom_mode == "Fit":
                # Fit watermark to whichever dimension is smaller to keep aspect ratio
                orig_h = original_image.size()[1]
                orig_w = original_image.size()[2]
                h_ratio = orig_h / wm_h
                w_ratio = orig_w / wm_w
                ratio = min(h_ratio, w_ratio)
                wm_w = round(wm_w * ratio)
                wm_h = round(wm_h * ratio)
            elif Zoom_mode == "zoom":
                wm_w = int(wm_w * Scaling_factor)
                wm_h = int(wm_h * Scaling_factor)
            elif Zoom_mode == "Scale_according_to_input_width_and_height":
                wm_w, wm_h = size[0], size[1]

            samples = Watermark_image.movedim(-1, 1)
            Watermark_image = common_upscale(samples, wm_w, wm_h, Scaling_method, False)
            Watermark_image = Watermark_image.movedim(1, -1)

        # Convert watermark to RGBA
        Watermark_image = tensor2pil(Watermark_image)
        Watermark_image = Watermark_image.convert('RGBA')
        Watermark_image.putalpha(Image.new("L", Watermark_image.size, 255))

        # If a mask was provided, invert it and apply as alpha
        if mask is not None:
            mask_pil = tensor2pil(mask)
            mask_pil = mask_pil.resize(Watermark_image.size)
            Watermark_image.putalpha(ImageOps.invert(mask_pil))

        # Rotate
        Watermark_image = Watermark_image.rotate(rotate, expand=True)

        # Adjust transparency by altering the alpha channel
        r, g, b, a = Watermark_image.split()
        a = a.point(lambda x: max(0, int(x * (1 - transparency / 100))))
        Watermark_image.putalpha(a)

        # Determine initial alignment if desired
        orig_w = original_image.size()[2]
        orig_h = original_image.size()[1]
        wm_w, wm_h = Watermark_image.size

        X_direction_int = None
        Y_direction_int = None

        # Compute final (x, y) based on initial_position
        if initial_position == "Centered":
            X_direction_int = int(X_direction + (orig_w - wm_w) / 2)
            Y_direction_int = int(Y_direction + (orig_h - wm_h) / 2)
        elif initial_position == "Up":
            X_direction_int = int(X_direction + (orig_w - wm_w) / 2)
            Y_direction_int = Y_direction
        elif initial_position == "Down":
            X_direction_int = int(X_direction + (orig_w - wm_w) / 2)
            Y_direction_int = int(Y_direction + orig_h - wm_h)
        elif initial_position == "Left":
            Y_direction_int = int(Y_direction + (orig_h - wm_h) / 2)
            X_direction_int = X_direction
        elif initial_position == "Right":
            X_direction_int = int(X_direction + orig_w - wm_w)
            Y_direction_int = int(Y_direction + (orig_h - wm_h) / 2)
        elif initial_position == "Up Left":
            # Up left: no further center calc
            pass
        elif initial_position == "Up Right":
            X_direction_int = int(orig_w - wm_w + X_direction)
            Y_direction_int = Y_direction
        elif initial_position == "Down Left":
            X_direction_int = X_direction
            Y_direction_int = int(orig_h - wm_h + Y_direction)
        elif initial_position == "Down Right":
            X_direction_int = int(X_direction + orig_w - wm_w)
            Y_direction_int = int(Y_direction + orig_h - wm_h)

        # If computed alignment is valid, update location
        if X_direction_int is not None and Y_direction_int is not None:
            location = (X_direction_int, Y_direction_int)

        # Process each image in the batch
        original_list = torch.unbind(original_image, dim=0)
        processed_list = []

        for tensor in original_list:
            base_pil = tensor2pil(tensor).convert("RGBA")

            # If no mask, alpha-blend entire region; else, use watermark alpha
            if mask is None:
                base_pil.paste(Watermark_image, location)
            else:
                base_pil.paste(Watermark_image, location, Watermark_image)

            processed_tensor = pil2tensor(base_pil.convert("RGB"))
            processed_list.append(processed_tensor)

        # Re-stack the batch
        output = torch.stack([img.squeeze() for img in processed_list])
        return (output,)
