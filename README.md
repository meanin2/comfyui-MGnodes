# ComfyUI Image Watermarking Node

A custom node for ComfyUI that allows you to add image watermarks with advanced controls for transparency, positioning, and color manipulation.

![Showcase](assets/watermark_preview_wf.PNG)

## Features
- Add image watermarks with adjustable transparency
- Multiple positioning options (Center, Top Left, Top Right, Bottom Left, Bottom Right, Tiled)
- Scale watermark size
- Remove white background from watermark
- Invert watermark colors
- Preserve transparency in both base image and watermark

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes/
```

2. Clone this repository into your custom nodes directory:
```bash
git clone https://github.com/meanin2/comfyui-watermarking.git
```

3. Restart ComfyUI

## Usage

1. Load your base image using a "Load Image" node
2. Load your watermark image using another "Load Image" node
3. Connect both to the "Add Image Watermark" node
4. Configure the watermark settings:
   - **Position**: Choose where to place the watermark
   - **Opacity**: Adjust transparency (0-100%)
   - **Scale**: Resize the watermark (10-1000%)
   - **Make Black**: Invert the watermark colors
   - **Remove White**: Make white pixels transparent

### Example Settings

As shown in the showcase image:
- Center position with 65% scale and 100% opacity
- Remove white background enabled
- Original brush watermark vs inverted (black) version

### Example Workflow

Check out the `examples` folder for:
- A complete test workflow you can import into ComfyUI
- Sample images used in the showcase
- Test watermark images

## Tips
- For best results with "Remove White", ensure your watermark has a clean white background
- When using "Make Black", the inversion happens before white removal
- The watermark mask input is optional and can be used for additional transparency control
- Scale values above 100% will enlarge the watermark, below 100% will shrink it
