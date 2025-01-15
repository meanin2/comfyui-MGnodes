import os
import numpy as np
import torch
from PIL import Image, ImageOps

# If ComfyUI doesn't provide a max resolution, define your own:
MAX_RESOLUTION = 8192

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Converts a PyTorch float tensor [0..1] with shape (C, H, W)
    or (B, C, H, W) to a PIL Image (first image in batch, if needed).
    """
    # Squeeze any singleton batch dimension and clamp values to [0..255]
    array = (image_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    # If 3 or 4 channels, convert accordingly
    if array.ndim == 3:
        # e.g., shape (3, H, W) -> (H, W, 3)
        array = np.moveaxis(array, 0, -1)
    return Image.fromarray(array)

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch float tensor [0..1] with shape (1, C, H, W).
    """
    array = np.array(pil_image).astype(np.float32) / 255.0
    # (H, W, C) -> (C, H, W)
    if array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
    else:
        # Grayscale fallback
        array = np.expand_dims(array, 0)
    return torch.from_numpy(array).unsqueeze(0)

def upscale_image(
    tensor_image: torch.Tensor, 
    width: int, 
    height: int, 
    method: str = "bilinear"
) -> torch.Tensor:
    """
    Upscales or downscales a tensor image to a target (width, height) 
    using the specified interpolation method.
    
    method options (ComfyUI-compatible): 
      - "nearest-exact"
      - "bilinear"
      - "area"
      - "bicubic" (if you prefer)
      - "bislerp"/"lanczos" (if you implement custom variants)
    """
    # Move channels from last dim to (N, C, H, W) -> (N, H, W, C), if needed
    # However, ComfyUI typically uses (N, H, W, C). Adjust accordingly if needed.
    
    # For demonstration, we assume (N, C, H, W).
    # If your input is (N, H, W, C), just adapt accordingly.
    return torch.nn.functional.interpolate(
        tensor_image, 
        size=(height, width), 
        mode=method if method in ("nearest-exact", "bilinear", "area", "bicubic") else "bilinear",
        align_corners=(method == "bilinear")  # Typically True for bilinear
    )

class WatermarkOverlayNode:
    """
    Overlays an 'overlay_image' (i.e., watermark) onto a 'base_image'.

    Features:
    - Zoom/resize the watermark.
    - Rotate it.
    - Adjust transparency.
    - Mask the watermark region if desired.
    - Position the watermark in various presets (center, corners, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "zoom_mode": (["None", "Fit", "zoom", "Scale_by_input_dims"],),
                "scaling_method": (["nearest-exact", "bilinear", "area", "bicubic"],),
                "scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.1}),
                "target_width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "target_height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
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
                "offset_x": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "offset_y": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "rotation_degrees": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "transparency_pct": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5, "display": "slider"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Image Processing/Watermark"

    def apply_overlay(
        self,
        base_image,
        overlay_image,
        zoom_mode,
        scaling_method,
        scaling_factor,
        target_width,
        target_height,
        offset_x,
        offset_y,
        rotation_degrees,
        transparency_pct,
        initial_position,
        mask=None
    ):
        """
        Apply the overlay_image onto the base_image using the specified parameters.

        Parameters
        ----------
        base_image : torch.Tensor
            The main image we want to watermark, shape: (N, C, H, W).
        overlay_image : torch.Tensor
            The watermark image, shape: (N, C, H, W).
        zoom_mode : str
            One of ["None", "Fit", "zoom", "Scale_by_input_dims"].
        scaling_method : str
            Interpolation method, e.g. "bilinear", "nearest-exact", "area", "bicubic".
        scaling_factor : float
            Factor by which to scale the watermark if zoom_mode == "zoom".
        target_width, target_height : int
            Explicit dimensions if zoom_mode == "Scale_by_input_dims".
        offset_x, offset_y : int
            Additional translation from the computed position.
        rotation_degrees : int
            How many degrees to rotate the watermark (e.g., 90 for 90° CCW).
        transparency_pct : float
            0 means no transparency, 100 means fully transparent.
        initial_position : str
            One of "Centered", "Up", "Down", "Left", "Right", 
            "Up Left", "Up Right", "Down Left", "Down Right".
        mask : torch.Tensor, optional
            If provided, will be used to mask out parts of the watermark.

        Returns
        -------
        tuple[torch.Tensor]
            The watermarked image as a single-element tuple.

        Notes
        -----
        - If the watermark is too large, consider resizing or using 
          a smaller overlay to avoid memory issues.
        - Transparency is applied by adjusting the alpha channel 
          before compositing onto the base_image.
        """

        # --- STEP 1: Convert overlay image to PIL
        #    ComfyUI uses Tensors in the form (N, C, H, W)
        #    We'll process them batch-wise.
        overlay_pil_batch = []

        # For zoom calculations, we only need the shape of the first item in the batch
        # (assuming the watermark is the same across the batch).
        _, _, overlay_h, overlay_w = overlay_image.shape
        _, _, base_h, base_w = base_image.shape

        new_w, new_h = overlay_w, overlay_h
        if zoom_mode != "None":
            if zoom_mode == "Fit":
                # Fit to the smaller dimension of base_image
                h_ratio = base_h / overlay_h
                w_ratio = base_w / overlay_w
                ratio = min(h_ratio, w_ratio)
                new_w = round(overlay_w * ratio)
                new_h = round(overlay_h * ratio)

            elif zoom_mode == "zoom":
                new_w = int(overlay_w * scaling_factor)
                new_h = int(overlay_h * scaling_factor)

            elif zoom_mode == "Scale_by_input_dims":
                new_w = target_width
                new_h = target_height

            # Perform the actual scaling
            # Move to (N, C, H, W) if not already
            scaled = upscale_image(overlay_image, new_w, new_h, scaling_method)
            overlay_image = scaled

        # Convert from torch to PIL
        # We'll handle each item in the batch individually
        batch_overlays = torch.unbind(overlay_image, dim=0)
        for overlay_tensor in batch_overlays:
            overlay_pil = tensor_to_pil(overlay_tensor)
            overlay_pil = overlay_pil.convert("RGBA")  # Ensure RGBA for alpha manipulation
            # Potentially re-apply full alpha if needed
            alpha_mask = Image.new("L", overlay_pil.size, 255)
            overlay_pil.putalpha(alpha_mask)
            overlay_pil_batch.append(overlay_pil)

        # --- STEP 2: If a mask is provided, invert and apply it as alpha
        if mask is not None:
            batch_masks = torch.unbind(mask, dim=0)
            for i, mask_tensor in enumerate(batch_masks):
                mask_pil = tensor_to_pil(mask_tensor).convert("L")
                # Resize mask to match overlay
                mask_pil = mask_pil.resize(overlay_pil_batch[i].size, resample=Image.BILINEAR)
                # Invert the mask, then apply as alpha
                inverted_mask = ImageOps.invert(mask_pil)
                overlay_pil_batch[i].putalpha(inverted_mask)

        # --- STEP 3: Rotation & Transparency
        for i, overlay_pil in enumerate(overlay_pil_batch):
            # Rotate with expand=True so the bounding box can grow
            overlay_pil = overlay_pil.rotate(rotation_degrees, expand=True)

            # Adjust alpha channel for transparency
            r, g, b, a = overlay_pil.split()
            # transparency_pct% => alpha reduction
            new_alpha = a.point(lambda x: int(x * (1 - transparency_pct / 100)))
            overlay_pil.putalpha(new_alpha)

            overlay_pil_batch[i] = overlay_pil

        # --- STEP 4: Positioning Calculation
        def compute_offset(pos, base_dim, overlay_dim, user_offset):
            """
            pos : str, initial_position
            base_dim : (width, height) of the base image
            overlay_dim : (width, height) of the overlay
            user_offset : (offset_x, offset_y)
            """
            base_w_, base_h_ = base_dim
            over_w_, over_h_ = overlay_dim
            off_x, off_y = user_offset

            # Default to user offset
            final_x, final_y = off_x, off_y

            if pos == "Centered":
                final_x = int(off_x + (base_w_ - over_w_) / 2)
                final_y = int(off_y + (base_h_ - over_h_) / 2)
            elif pos == "Up":
                final_x = int(off_x + (base_w_ - over_w_) / 2)
                final_y = off_y
            elif pos == "Down":
                final_x = int(off_x + (base_w_ - over_w_) / 2)
                final_y = int(off_y + base_h_ - over_h_)
            elif pos == "Left":
                final_x = off_x
                final_y = int(off_y + (base_h_ - over_h_) / 2)
            elif pos == "Right":
                final_x = int(off_x + base_w_ - over_w_)
                final_y = int(off_y + (base_h_ - over_h_) / 2)
            elif pos == "Up Left":
                # No further offset beyond user_offset
                pass
            elif pos == "Up Right":
                final_x = int(base_w_ - over_w_ + off_x)
                final_y = off_y
            elif pos == "Down Left":
                final_x = off_x
                final_y = int(base_h_ - over_h_ + off_y)
            elif pos == "Down Right":
                final_x = int(off_x + base_w_ - over_w_)
                final_y = int(off_y + base_h_ - over_h_)
            return (final_x, final_y)

        # --- STEP 5: Composite the watermark onto each base image in the batch
        batch_bases = torch.unbind(base_image, dim=0)
        output_batch = []

        for idx, base_tensor in enumerate(batch_bases):
            # Convert base to PIL
            base_pil = tensor_to_pil(base_tensor).convert("RGBA")

            # Current overlay
            overlay_pil = overlay_pil_batch[min(idx, len(overlay_pil_batch) - 1)]
            pos_x, pos_y = compute_offset(
                initial_position,
                (base_pil.width, base_pil.height),
                (overlay_pil.width, overlay_pil.height),
                (offset_x, offset_y),
            )

            # Composite: If we have an alpha mask in the overlay, 
            #            pass it as the mask argument in paste().
            base_pil.paste(overlay_pil, (pos_x, pos_y), overlay_pil)

            # Convert back to RGB to be consistent with ComfyUI
            base_pil_rgb = base_pil.convert("RGB")
            output_batch.append(pil_to_tensor(base_pil_rgb))

        # Re-stack into shape (N, C, H, W)
        output_tensor = torch.cat(output_batch, dim=0)
        return (output_tensor,)
