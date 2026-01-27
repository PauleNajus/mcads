from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")  # Headless-safe backend (server/Docker)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from django.conf import settings
from .models import XRayImage, VisualizationResult

logger = logging.getLogger(__name__)

# Re-export functions from interpretability.py if they are generic enough, 
# OR keep them there if they are tightly coupled. 
# The previous `utils.py` implementation had duplication of saving logic.
# Wait, `interpretability.py` HAS the saving functions now?
# Let me check interpretability.py again. I thought I just saw them in utils.py
# Ah, I might have duplicated them or `utils.py` imported them.
# Let's check `interpretability.py` content from previous turn.

# Checking previous turn's `interpretability.py` write...
# It contains `apply_gradcam`, `apply_pixel_interpretability` etc.
# AND it contains `save_interpretability_visualization`, `save_overlay_visualization`, `save_saliency_map`, etc.
#
# So `utils.py` WAS duplicating them? Or importing them?
# In `utils.py` (from `read_file` earlier):
# def save_interpretability_visualization(...): ... implementation ...
#
# It seems `utils.py` had the implementations.
# Did I move them to `interpretability.py`?
#
# Let's check `interpretability.py` again.
# The `Write` to `interpretability.py` in previous turn ended with `apply_combined_pixel_interpretability`.
# It did NOT contain `save_*` functions.
#
# Wait, I might be misremembering where I put them or if I put them.
# I see `from .interpretability import ... save_interpretability_visualization ...` in `services.py` import?
# No, `services.py` imports `save_and_record_visualization` from `.utils`.
#
# Let's look at `utils.py` again. It HAS the `save_*` functions.
#
# So `visualization_utils.py` is the right place for them.

def save_interpretability_visualization(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save interpretability visualization to a file
    
    Args:
        interpretation_results: Results from process_image_with_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    # Create figure with subplots
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(interpretation_results['original'], cmap='gray')
    plt.title('Original X-ray')
    plt.axis('off')
    
    # Plot method-specific visualizations
    if interpretation_results.get('method') == 'gradcam':
        # Plot heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['heatmap'], cmap='jet')
        plt.title(f'Heatmap\n{interpretation_results["target_class"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(interpretation_results['overlay'], cv2.COLOR_BGR2RGB))
        plt.title('Grad-CAM Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'combined_gradcam':
        # Plot combined heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['heatmap'], cmap='jet')
        plt.title(f'Combined Heatmap\n{len(interpretation_results["selected_pathologies"])} pathologies > {interpretation_results["threshold"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(interpretation_results['overlay'], cv2.COLOR_BGR2RGB))
        plt.title('Combined Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'combined_pli':
        # Plot combined saliency map
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['saliency_map'], cmap='jet')
        plt.title(f'Combined PLI Saliency\n{len(interpretation_results["selected_pathologies"])} pathologies > {interpretation_results["threshold"]}')
        plt.axis('off')
        
        # Plot overlay
        plt.subplot(1, 3, 3)
        plt.imshow(interpretation_results['overlay'])
        plt.title('Combined PLI Overlay')
        plt.axis('off')
    
    elif interpretation_results.get('method') == 'pli':
        # Plot saliency map
        plt.subplot(1, 3, 2)
        plt.imshow(interpretation_results['saliency_map'], cmap='jet')
        plt.title(f'Saliency Map\n{interpretation_results["target_class"]}')
        plt.axis('off')
        
        # Plot overlay rather than colored saliency
        plt.subplot(1, 3, 3)
        plt.imshow(interpretation_results['overlay'])
        plt.title('Pixel-Level Overlay')
        plt.axis('off')
    
    # Add image metadata as text if available
    if 'metadata' in interpretation_results:
        metadata = interpretation_results['metadata']
        metadata_text = f"Image: {metadata.get('name', 'Unknown')} | Format: {metadata.get('format', 'Unknown')} | Size: {metadata.get('size', 'Unknown')} | Resolution: {metadata.get('resolution', 'Unknown')}"
        plt.figtext(0.5, 0.01, metadata_text, ha='center', fontsize=8, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save the figure with metadata preserved
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=300, bbox_inches='tight', metadata={'Interpretation': interpretation_results.get('method', 'Unknown'), 
                                                                                    'Target': interpretation_results.get('target_class', 'Unknown')})
    plt.close()
    
    return output_path


def save_overlay_visualization(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the overlay visualization to a file for pixel-level interpretability without white spaces
    
    Args:
        interpretation_results: Results from apply_pixel_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'overlay' in interpretation_results:
        # Get the overlay data (should already be in proper RGB format)
        overlay = interpretation_results['overlay']
        
        # Save directly as image without matplotlib padding
        Image.fromarray(overlay).save(output_path, format=format.upper())
    
    return output_path


def save_saliency_map(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the saliency map to a file for pixel-level interpretability without white spaces
    
    Args:
        interpretation_results: Results from apply_pixel_interpretability
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'saliency_map' in interpretation_results:
        # Get the saliency map data
        saliency_map = interpretation_results['saliency_map']
        
        # Apply colormap to saliency map (convert to 0-255 range)
        saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)  # type: ignore
        
        # Convert BGR to RGB for proper color display
        saliency_rgb = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        Image.fromarray(saliency_rgb).save(output_path, format=format.upper())
    
    return output_path


def save_heatmap(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the heatmap to a file without white spaces
    
    Args:
        interpretation_results: Results from interpretability analysis
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'heatmap' in interpretation_results and 'original' in interpretation_results:
        # Get the heatmap data and original image dimensions
        heatmap = interpretation_results['heatmap']
        original_shape = interpretation_results['original'].shape
        
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
        
        # Apply colormap to heatmap (convert to 0-255 range)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)  # type: ignore
        
        # Convert BGR to RGB for proper color display
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        Image.fromarray(heatmap_rgb).save(output_path, format=format.upper())
    
    return output_path


def save_overlay(
    interpretation_results: dict[str, Any],
    output_path: Path,
    format: str = 'png',
) -> Path:
    """
    Save only the overlay to a file without white spaces
    
    Args:
        interpretation_results: Results from interpretability analysis
        output_path: Path to save the visualization
        format: Image format to save (png, jpg, etc.)
        
    Returns:
        Path to saved file
    """
    if 'overlay' in interpretation_results:
        # Get the overlay data (already in RGB format from overlay_heatmap method)
        overlay = interpretation_results['overlay']
        
        # Convert BGR to RGB if needed (overlay_heatmap returns BGR format)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        
        # Save directly as image without matplotlib padding
        Image.fromarray(overlay_rgb).save(output_path, format=format.upper())
    
    return output_path


def save_segmentation_visualization(
    segmentation_results: dict[str, Any],
    output_path: Path,
    structures_to_show: list[int] | None = None,
) -> Path:
    """
    Save segmentation visualization with colored overlays for each anatomical structure
    
    Args:
        segmentation_results: Results from apply_segmentation
        output_path: Path to save the visualization
        structures_to_show: List of structure indices to show (None = show all)
        
    Returns:
        Path to saved file
    """
    original = segmentation_results['original']
    binary_masks = segmentation_results['binary_masks']
    anatomical_structures = segmentation_results['anatomical_structures']
    
    # `original` is stored in TorchXRayVision's normalized space (~[-1024, 1024]).
    # Convert it to a displayable 8-bit image before compositing overlays.
    #
    # This fixed mapping is intentional (fast + consistent across images):
    #   -1024 -> 0, 0 -> 127/128, 1024 -> 255
    if len(original.shape) == 2:
        orig01 = (original.astype("float32") / 1024.0 + 1.0) * 0.5
        orig01 = np.clip(orig01, 0.0, 1.0)
        original_u8 = (orig01 * 255.0).astype(np.uint8)
        # Keep OpenCV images in BGR because our overlay colors are defined in BGR.
        original_bgr = cv2.cvtColor(original_u8, cv2.COLOR_GRAY2BGR)
    else:
        # Fallback for unexpected 3-channel inputs (assume already in [0, 1]).
        original_bgr = (np.clip(original, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Resize original to match mask dimensions (512x512)
    original_resized = cv2.resize(original_bgr, (512, 512))
    
    # Create overlay image
    overlay = original_resized.copy()
    
    # Define colors for each anatomical structure (BGR format)
    colors = [
        (255, 0, 0),     # Blue - Left Clavicle
        (0, 255, 0),     # Green - Right Clavicle
        (0, 0, 255),     # Red - Left Scapula
        (255, 255, 0),   # Cyan - Right Scapula
        (255, 0, 255),   # Magenta - Left Lung
        (0, 255, 255),   # Yellow - Right Lung
        (128, 0, 255),   # Purple - Left Hilus Pulmonis
        (255, 128, 0),   # Orange - Right Hilus Pulmonis
        (0, 128, 255),   # Light Blue - Heart
        (255, 0, 128),   # Pink - Aorta
        (128, 255, 0),   # Lime - Facies Diaphragmatica
        (0, 255, 128),   # Spring Green - Mediastinum
        (128, 128, 255), # Light Purple - Weasand
        (255, 128, 128)  # Light Red - Spine
    ]
    
    # Apply masks
    if structures_to_show is None:
        structures_to_show = list(range(len(anatomical_structures)))
    
    for idx in structures_to_show:
        if idx < len(binary_masks):
            mask = binary_masks[idx]
            color = colors[idx % len(colors)]
            
            # Create colored mask
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :] = color
            
            # Apply mask with transparency
            # Multiply colored mask by the binary mask and alpha value
            masked_color = (colored_mask * mask[:, :, np.newaxis] * 0.3).astype(overlay.dtype)
            overlay = cv2.addWeighted(overlay, 1.0, masked_color, 1.0, 0)
    
    # Convert BGR to RGB for saving
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Save the visualization
    Image.fromarray(overlay_rgb).save(output_path, format='PNG')
    
    return output_path


def save_individual_segmentation_masks(
    segmentation_results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """
    Save individual binary masks for each anatomical structure
    
    Args:
        segmentation_results: Results from apply_segmentation
        output_dir: Directory to save individual masks
        
    Returns:
        Dictionary mapping structure names to file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    binary_masks = segmentation_results['binary_masks']
    anatomical_structures = segmentation_results['anatomical_structures']
    
    saved_paths = {}
    
    for idx, structure_name in enumerate(anatomical_structures):
        if idx < len(binary_masks):
            mask = binary_masks[idx]
            
            # Convert to uint8 (0 or 255)
            mask_img = (mask * 255).astype(np.uint8)
            
            # Clean filename
            filename = f"{structure_name.lower().replace(' ', '_')}_mask.png"
            filepath = output_dir / filename
            
            # Save mask
            Image.fromarray(mask_img, mode='L').save(filepath)
            saved_paths[structure_name] = str(filepath)
    
    return saved_paths
