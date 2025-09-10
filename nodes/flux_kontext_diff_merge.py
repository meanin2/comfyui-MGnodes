import torch
import numpy as np
import warnings

# Handle optional dependencies gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("opencv-python not installed. Some features will be limited. Install with: pip install opencv-python")

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not installed. SSIM detection will be unavailable. Install with: pip install scikit-image")

from scipy.ndimage import binary_dilation, binary_erosion
import comfy.model_management as model_management


class FluxKontextDiffMerge:
    """Flux Kontext Diff Merge - Optimized Version
    
    Preserves image quality by selectively merging only changed regions
    from AI-generated edits back into the original image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        detection_methods = ["adaptive", "color_diff"]
        if SKIMAGE_AVAILABLE:
            detection_methods.append("ssim")
        detection_methods.append("combined")
        
        blend_methods = ["alpha"]  # Always available
        if CV2_AVAILABLE:
            blend_methods = ["poisson", "alpha", "multiband", "gaussian"]
        
        return {
            "required": {
                "original_image": ("IMAGE",),
                "edited_image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.02,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Sensitivity of change detection; lower values detect more subtle changes"
                }),
                "detection_method": (detection_methods, {
                    "default": "adaptive",
                    "tooltip": "Algorithm used to build the initial mask of differences"
                }),
                "blend_method": (blend_methods, {
                    "default": "poisson" if CV2_AVAILABLE else "alpha",
                    "tooltip": "How to merge the edited pixels back onto the original"
                }),
                "mask_blur": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Gaussian blur radius (in pixels) applied to the mask to soften edges"
                }),
                "mask_expand": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Dilate the detected mask by this many pixels before blurring"
                }),
                "edge_feather": ("INT", {
                    "default": 15,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Additional feathering (fine Gaussian) after the main blur"
                }),
                "min_change_area": ("INT", {
                    "default": 250,
                    "min": 0,
                    "max": 5000,
                    "step": 10,
                    "tooltip": "Ignore change blobs smaller than this area (pixel²)"
                }),
                "global_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Controls how aggressively the adaptive detector compensates for global shifts"
                }),
            },
            "optional": {
                "manual_mask": ("MASK", {"tooltip": "User-supplied 1-channel mask to override automatic detection"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("merged_image", "difference_mask", "preview_diff")
    FUNCTION = "merge_diff"
    CATEGORY = "image/postprocessing"
    
    def tensor_to_numpy_batch(self, tensor):
        """Convert ComfyUI tensor to numpy arrays - optimized for batch processing"""
        # ComfyUI tensors are (B, H, W, C) in float32 [0, 1]
        if tensor.dtype != torch.uint8:
            tensor = (tensor * 255).to(torch.uint8)
        return tensor.cpu().numpy()
    
    def numpy_to_tensor_batch(self, numpy_array):
        """Convert numpy array back to ComfyUI tensor - optimized"""
        if numpy_array.dtype == np.uint8:
            numpy_array = numpy_array.astype(np.float32) / 255.0
        return torch.from_numpy(numpy_array)
    
    def adaptive_detection(self, original, edited, threshold=0.02, global_threshold=0.15):
        """Adaptive detection that's robust to global changes - CV2 optional version"""
        if CV2_AVAILABLE:
            # Use LAB color space for better perceptual differences
            orig_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
            edit_lab = cv2.cvtColor(edited, cv2.COLOR_RGB2LAB)
            
            diff_l = np.abs(orig_lab[:,:,0].astype(np.float32) - edit_lab[:,:,0].astype(np.float32))
            diff_a = np.abs(orig_lab[:,:,1].astype(np.float32) - edit_lab[:,:,1].astype(np.float32))
            diff_b = np.abs(orig_lab[:,:,2].astype(np.float32) - edit_lab[:,:,2].astype(np.float32))
            
            combined_diff = (diff_l * 0.5 + diff_a * 0.25 + diff_b * 0.25)
        else:
            # Fallback to RGB difference if CV2 not available
            diff = np.abs(original.astype(np.float32) - edited.astype(np.float32))
            combined_diff = np.mean(diff, axis=2)
        
        # Calculate global average difference
        global_avg = np.mean(combined_diff)
        
        # Adaptive thresholding
        if global_avg > global_threshold * 255:
            thres = global_avg + (threshold * 255)
        else:
            thres = threshold * 255
        
        mask = (combined_diff > thres).astype(np.uint8) * 255
        return mask
    
    def detect_color_changes(self, original, edited, threshold=0.02):
        """Simple RGB color change detection"""
        diff = np.abs(original.astype(np.float32) - edited.astype(np.float32))
        max_diff = np.max(diff, axis=2)
        thres = threshold * 255
        mask = (max_diff > thres).astype(np.uint8) * 255
        return mask
    
    def detect_ssim_changes(self, original, edited, threshold=0.02):
        """SSIM-based change detection"""
        if not SKIMAGE_AVAILABLE:
            # Fallback to color detection
            return self.detect_color_changes(original, edited, threshold)
        
        # Convert to grayscale using numpy
        orig_gray = np.dot(original[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        edit_gray = np.dot(edited[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        try:
            score, diff = ssim(orig_gray, edit_gray, full=True, data_range=255)
        except Exception:
            diff = np.abs(orig_gray.astype(np.float32) - edit_gray.astype(np.float32)) / 255.0
        
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        mask = (diff_normalized > threshold).astype(np.uint8) * 255
        return mask
    
    def edge_aware_detection(self, original, edited, threshold=0.02):
        """Edge-aware detection - requires CV2"""
        if not CV2_AVAILABLE:
            return self.detect_color_changes(original, edited, threshold)
        
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        edit_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)
        
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        edit_edges = cv2.Canny(edit_gray, 50, 150)
        
        edge_diff = cv2.absdiff(orig_edges, edit_edges)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_diff = cv2.dilate(edge_diff, kernel, iterations=2)
        
        intensity_diff = cv2.absdiff(orig_gray, edit_gray)
        thres = threshold * 255
        intensity_mask = (intensity_diff > thres).astype(np.uint8) * 255
        
        combined_mask = cv2.bitwise_or(edge_diff, intensity_mask)
        return combined_mask
    
    def detect_changes(self, original, edited, threshold=0.02, method="adaptive", global_threshold=0.15):
        """Main change detection with fallbacks for missing dependencies"""
        if method == "adaptive":
            mask = self.adaptive_detection(original, edited, threshold, global_threshold)
        elif method == "color_diff":
            mask = self.detect_color_changes(original, edited, threshold)
        elif method == "ssim":
            mask = self.detect_ssim_changes(original, edited, threshold)
        elif method == "combined":
            mask1 = self.adaptive_detection(original, edited, threshold, global_threshold)
            mask2 = self.edge_aware_detection(original, edited, threshold)
            if CV2_AVAILABLE:
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = np.maximum(mask1, mask2)
        else:
            mask = self.adaptive_detection(original, edited, threshold, global_threshold)
        
        return mask
    
    def filter_small_changes(self, mask, min_area=250):
        """Remove small areas - with CV2 fallback"""
        if CV2_AVAILABLE:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(mask)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.fillPoly(filtered_mask, [contour], 255)
        else:
            # Simple scipy-based filtering
            from scipy import ndimage
            labeled, num_features = ndimage.label(mask > 127)
            filtered_mask = np.zeros_like(mask)
            
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > min_area:
                    filtered_mask[component] = 255
        
        return filtered_mask
    
    def refine_mask(self, mask, expand_pixels=8, blur_amount=15, feather_amount=15):
        """Refine mask with fallbacks for missing CV2"""
        if CV2_AVAILABLE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Morphological operations
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Expand
            if expand_pixels > 0:
                expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (expand_pixels*2+1, expand_pixels*2+1))
                mask = cv2.dilate(mask, expand_kernel, iterations=1)
            
            # Blur
            if blur_amount > 0:
                mask = cv2.GaussianBlur(mask, (blur_amount*2+1, blur_amount*2+1), 0)
            
            # Feather
            if feather_amount > 0:
                mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 
                                       feather_amount/3)
        else:
            # Scipy fallback
            from scipy.ndimage import gaussian_filter, binary_dilation
            
            # Simple dilation for expansion
            if expand_pixels > 0:
                struct = np.ones((expand_pixels*2+1, expand_pixels*2+1))
                mask = binary_dilation(mask > 127, structure=struct).astype(np.uint8) * 255
            
            # Gaussian blur
            if blur_amount > 0:
                mask = gaussian_filter(mask.astype(np.float32), blur_amount)
                mask = np.clip(mask, 0, 255).astype(np.uint8)
            
            # Additional feathering
            if feather_amount > 0:
                mask = gaussian_filter(mask.astype(np.float32), feather_amount/3)
                mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        return mask
    
    def poisson_blend(self, source, target, mask):
        """Poisson blending - CV2 required"""
        if not CV2_AVAILABLE:
            return self.alpha_blend(source, target, mask)
        
        try:
            if source.shape != target.shape:
                return self.alpha_blend(source, target, mask)

            binary_mask = (mask > 127).astype(np.uint8) * 255
            if np.sum(binary_mask) == 0:
                return self.alpha_blend(source, target, mask)

            h, w = binary_mask.shape

            # Find stable center using distance transform
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            inner_mask = cv2.erode(binary_mask, kernel, iterations=1)
            if np.sum(inner_mask) == 0:
                inner_mask = binary_mask

            dist = cv2.distanceTransform((inner_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
            if dist[cy, cx] <= 0:
                x, y, ww, hh = cv2.boundingRect(binary_mask)
                cx = x + ww // 2
                cy = y + hh // 2

            cx = int(max(1, min(cx, w - 2)))
            cy = int(max(1, min(cy, h - 2)))

            # Handle edge cases with padding
            touches_edge = (
                np.any(binary_mask[0, :]) or np.any(binary_mask[-1, :]) or
                np.any(binary_mask[:, 0]) or np.any(binary_mask[:, -1])
            )

            if touches_edge:
                pad = max(16, (max(h, w) // 100) * 2 + 8)
                src_p = cv2.copyMakeBorder(source, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
                dst_p = cv2.copyMakeBorder(target, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
                msk_p = cv2.copyMakeBorder(binary_mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

                center = (cx + pad, cy + pad)
                result_p = cv2.seamlessClone(src_p, dst_p, msk_p, center, cv2.NORMAL_CLONE)
                result = result_p[pad:pad + h, pad:pad + w]
                return result
            else:
                center = (cx, cy)
                result = cv2.seamlessClone(source, target, binary_mask, center, cv2.NORMAL_CLONE)
                return result

        except Exception as e:
            print(f"Poisson blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def alpha_blend(self, source, target, mask):
        """Simple alpha blending - always available"""
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        result = target.astype(np.float32) * (1 - mask_3ch) + source.astype(np.float32) * mask_3ch
        return result.astype(np.uint8)
    
    def multiband_blend(self, source, target, mask):
        """Multi-band blending - requires CV2"""
        if not CV2_AVAILABLE:
            return self.alpha_blend(source, target, mask)
        
        try:
            levels = 6
            mask_normalized = mask.astype(np.float32) / 255.0
            
            if source.shape != target.shape:
                return self.alpha_blend(source, target, mask)
            
            source_pyr = [source.astype(np.float32)]
            target_pyr = [target.astype(np.float32)]
            mask_pyr = [mask_normalized]
            
            for i in range(levels):
                if source_pyr[i].shape[0] < 4 or source_pyr[i].shape[1] < 4:
                    levels = i
                    break
                source_pyr.append(cv2.pyrDown(source_pyr[i]))
                target_pyr.append(cv2.pyrDown(target_pyr[i]))
                mask_pyr.append(cv2.pyrDown(mask_pyr[i]))
            
            result_pyr = []
            for i in range(levels + 1):
                mask_3ch = np.stack([mask_pyr[i]] * 3, axis=-1)
                blended = target_pyr[i] * (1 - mask_3ch) + source_pyr[i] * mask_3ch
                result_pyr.append(blended)
            
            result = result_pyr[levels]
            for i in range(levels - 1, -1, -1):
                result = cv2.pyrUp(result, dstsize=(result_pyr[i].shape[1], result_pyr[i].shape[0]))
                result = result + result_pyr[i]
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Multiband blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def gaussian_blend(self, source, target, mask):
        """Gaussian-weighted blending - requires CV2"""
        if not CV2_AVAILABLE:
            return self.alpha_blend(source, target, mask)
        
        try:
            if source.shape != target.shape:
                return self.alpha_blend(source, target, mask)
            
            mask_normalized = mask.astype(np.float32) / 255.0
            
            if np.sum(mask_normalized) == 0:
                return target
            
            dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            dist_transform = dist_transform / (dist_transform.max() + 1e-8)
            
            gaussian_mask = cv2.GaussianBlur(dist_transform, (21, 21), 0)
            mask_3ch = np.stack([gaussian_mask] * 3, axis=-1)
            
            result = target.astype(np.float32) * (1 - mask_3ch) + source.astype(np.float32) * mask_3ch
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Gaussian blending failed: {e}, falling back to alpha blending")
            return self.alpha_blend(source, target, mask)
    
    def create_preview_diff(self, original, edited, mask):
        """Create a preview showing the differences"""
        preview = original.copy()
        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
        
        tint_color = np.array([255, 100, 100])  # Red tint
        tinted = preview.astype(np.float32) * (1 - mask_3ch * 0.3) + tint_color * mask_3ch * 0.3
        
        return tinted.astype(np.uint8)
    
    def merge_diff(self, original_image, edited_image, threshold, detection_method,
                   blend_method, mask_blur, mask_expand, edge_feather, 
                   min_change_area, global_threshold, manual_mask=None):
        """Main entry point with batch processing support"""
        
        # Convert to numpy batch
        original_batch = self.tensor_to_numpy_batch(original_image)
        edited_batch = self.tensor_to_numpy_batch(edited_image)
        
        batch_size = original_batch.shape[0]
        edited_size = edited_batch.shape[0]
        
        # Handle broadcasting
        if batch_size != edited_size:
            if batch_size == 1 and edited_size > 1:
                print(f"Broadcasting single original image to match edited batch of size {edited_size}")
                original_batch = np.repeat(original_batch, edited_size, axis=0)
                batch_size = edited_size
            else:
                raise ValueError(f"Batch size mismatch: {batch_size} vs {edited_size}")
        
        # Process manual masks if provided
        manual_mask_batch = None
        if manual_mask is not None:
            manual_mask_np = manual_mask.detach().cpu().numpy()
            if len(manual_mask_np.shape) == 3:  # Batched masks
                manual_mask_batch = (manual_mask_np * 255).astype(np.uint8)
            else:  # Single mask
                manual_mask_batch = np.repeat(
                    np.expand_dims((manual_mask_np * 255).astype(np.uint8), 0),
                    batch_size, axis=0
                )
        
        # Process each item
        results = []
        masks = []
        previews = []
        
        for idx in range(batch_size):
            original_np = original_batch[idx]
            edited_np = edited_batch[idx]
            
            # Resize if needed
            if CV2_AVAILABLE and original_np.shape != edited_np.shape:
                edited_np = cv2.resize(edited_np, (original_np.shape[1], original_np.shape[0]))
            elif original_np.shape != edited_np.shape:
                # Fallback resize using numpy/scipy
                from scipy.ndimage import zoom
                factors = (original_np.shape[0] / edited_np.shape[0],
                          original_np.shape[1] / edited_np.shape[1], 1)
                edited_np = zoom(edited_np, factors, order=1).astype(np.uint8)
            
            # Get or detect mask
            if manual_mask_batch is not None:
                mask = manual_mask_batch[idx]
                if CV2_AVAILABLE and mask.shape != original_np.shape[:2]:
                    mask = cv2.resize(mask, (original_np.shape[1], original_np.shape[0]))
                elif mask.shape != original_np.shape[:2]:
                    from scipy.ndimage import zoom
                    factors = (original_np.shape[0] / mask.shape[0],
                              original_np.shape[1] / mask.shape[1])
                    mask = zoom(mask, factors, order=1).astype(np.uint8)
            else:
                mask = self.detect_changes(original_np, edited_np, threshold, 
                                         detection_method, global_threshold)
                if min_change_area > 0:
                    mask = self.filter_small_changes(mask, min_change_area)
            
            # Refine mask
            refined_mask = self.refine_mask(mask, mask_expand, mask_blur, edge_feather)
            
            # Apply blending
            if blend_method == "poisson":
                result_np = self.poisson_blend(edited_np, original_np, refined_mask)
            elif blend_method == "alpha":
                result_np = self.alpha_blend(edited_np, original_np, refined_mask)
            elif blend_method == "multiband":
                result_np = self.multiband_blend(edited_np, original_np, refined_mask)
            elif blend_method == "gaussian":
                result_np = self.gaussian_blend(edited_np, original_np, refined_mask)
            else:
                result_np = self.alpha_blend(edited_np, original_np, refined_mask)
            
            # Create preview
            preview_np = self.create_preview_diff(original_np, edited_np, refined_mask)
            
            results.append(result_np)
            masks.append(refined_mask)
            previews.append(preview_np)
        
        # Convert back to tensors
        result_tensor = self.numpy_to_tensor_batch(np.array(results))
        
        # Handle mask tensor conversion
        mask_array = np.array(masks).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array)
        
        preview_tensor = self.numpy_to_tensor_batch(np.array(previews))
        
        return (result_tensor, mask_tensor, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "FluxKontextDiffMerge": FluxKontextDiffMerge
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextDiffMerge": "Flux Kontext Diff Merge (Optimized)"
}
