# WIP the node is not ready and does not really work yet.

# Watermark Node for ComfyUI

This custom node adds watermarking functionality to ComfyUI. It allows users to overlay text or images on a base image, with options for transparency, scaling, positioning, and tiling.

## Features
- Add text or image watermarks.
- Customize watermark opacity (transparency).
- Scale and position watermarks with precision.
- Tile watermarks across the entire image.
- Supports multiple font styles for text watermarks.

## Installation

1. Navigate to your ComfyUI directory.
2. Place the `watermark_node` folder into the `custom_nodes` directory.
3. Restart ComfyUI.

## Usage

### Inputs
- **Base Image** (`IMAGE`): The main image to which the watermark will be applied.
- **Watermark Image** (`IMAGE`, optional): An image to use as a watermark.
- **Watermark Text** (`STRING`, optional): Text to use as a watermark.

### Settings
- **Opacity** (`FLOAT`): Adjust the transparency of the watermark. Range: `0.0` (fully transparent) to `1.0` (fully opaque).
- **Position** (`COMBO`): Select where to place the watermark. Options: `center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`, `tiled`.
- **Scale** (`FLOAT`): Resize the watermark. Default: `1.0`.
- **Repeat** (`BOOL`): Tile the watermark across the image.
- **Text Color** (`STRING`): Hex or named color for text (e.g., `#FFFFFF` or `white`).
- **Text Size** (`INT`): Font size for text watermarks.
- **Font Style** (`COMBO`): Select the font style. Options: `Arial`, `Courier`, `Times`.

### Outputs
- **Watermarked Image** (`IMAGE`): The final image with the watermark applied.

## Example Workflow

1. Add the node to your workflow in ComfyUI.
2. Connect an image to the **Base Image** input.
3. Configure watermark settings directly in the node:
   - Specify text or upload an image as the watermark.
   - Adjust transparency, scale, and position.
4. Connect the output to further processing or saving.

## Troubleshooting
If the node fails, it will turn red in ComfyUI, providing a clear error message. Common issues include:
- Missing font files: Ensure the specified font is installed on your system.
- Invalid input types: Verify the base image and watermark are valid.
