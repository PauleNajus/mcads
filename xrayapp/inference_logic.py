from __future__ import annotations

import os
import logging
from typing import Any, cast
from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
import torchvision

from .model_loader import load_model, load_segmentation_model, load_autoencoder
from .transforms import get_xrv_transform
from .xrv_io import load_xrv_image as _load_xrv_image
from .image_processing import extract_image_metadata
from .models import XRayImage, VisualizationResult
from .visualization_utils import (
    save_interpretability_visualization,
    save_heatmap,
    save_overlay,
    save_saliency_map,
    save_overlay_visualization,
    save_segmentation_visualization,
    save_individual_segmentation_masks,
)

logger = logging.getLogger(__name__)

_model_cache: dict[str, Any] = {}
_calibration_cache_key = "calibration"


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def load_calibration() -> dict[str, dict[str, float]]:
    """Load per-pathology temperature calibration and thresholds."""
    if _calibration_cache_key in _model_cache:
        return _model_cache[_calibration_cache_key]

    # Defaults
    temps: dict[str, float] = {}
    thrs: dict[str, float] = {}

    # Try env JSON
    import json
    env_json = os.environ.get('XRV_CALIBRATION_JSON')
    if env_json:
        try:
            blob = json.loads(env_json)
            temps = {k: float(v) for k, v in blob.get('temperature', {}).items()}
            thrs = {k: float(v) for k, v in blob.get('thresholds', {}).items()}
        except Exception as e:
            logger.error(f"Failed to parse XRV_CALIBRATION_JSON: {e}")

    # Try file
    if not temps and not thrs:
        file_path = os.environ.get('XRV_CALIBRATION_FILE')
        if not file_path:
            # Default location under project base
            base_dir = Path(__file__).resolve().parent.parent
            file_path = str(base_dir / 'config' / 'calibration.json')
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    blob = json.load(f)
                    temps = {k: float(v) for k, v in blob.get('temperature', {}).items()}
                    thrs = {k: float(v) for k, v in blob.get('thresholds', {}).items()}
        except Exception as e:
            logger.error(f"Failed to load calibration file {file_path}: {e}")

    # Fill with identity defaults if missing based on known default pathologies
    default_pathologies = list(xrv.datasets.default_pathologies)
    for name in default_pathologies:
        temps.setdefault(name, 1.0)
        thrs.setdefault(name, 0.5)

    cfg = { 'temperature': temps, 'thresholds': thrs }
    _model_cache[_calibration_cache_key] = cfg
    return cfg


def apply_calibration_and_thresholds(results: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply temperature scaling (per label) and generate binary labels."""
    cfg = load_calibration()
    temps = cfg['temperature']
    thrs = cfg['thresholds']

    calibrated: dict[str, float] = {}
    binary: dict[str, bool] = {}
    for k, p in list(results.items()):
        # Skip non-pathology keys (added later) by checking value type
        if not isinstance(p, (float, int, np.floating)):
            continue
        T = max(1e-3, float(temps.get(k, 1.0)))
        # Temperature scaling in logit space
        logit = _logit(float(p))
        p_cal = _sigmoid(logit / T)
        calibrated[k] = p_cal
        thr = float(thrs.get(k, 0.5))
        binary[k] = bool(p_cal >= thr)

    # Merge back
    updated = dict(results)
    updated.update({k: float(v) for k, v in calibrated.items()})
    meta = {
        'calibration_temperatures': temps,
        'decision_thresholds': thrs,
        'predicted_labels': binary
    }
    return updated, meta


def compute_ood_score(img_np: np.ndarray) -> dict[str, float | bool]:
    """Compute an OOD score via reconstruction error from the autoencoder."""
    try:
        ae, ae_resize = load_autoencoder()
    except Exception:
        # If AE cannot be loaded, gracefully skip OOD gate
        return { 'ood_score': float('nan'), 'is_ood': False, 'threshold': float('inf') }

    # Ensure shape is (1, H, W)
    if img_np.ndim == 2:
        img_np = img_np[None, :, :]

    # Resize to AE input (center crop then resize like classifier preprocessing)
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(ae_resize)
    ])
    img_small = transform(img_np)

    # To tensor on CPU
    img_tensor = torch.from_numpy(img_small).unsqueeze(0)  # (1, 1, ae_resize, ae_resize)

    with torch.no_grad():
        # Encode and decode (AE API not typed; cast to Any for tooling)
        ae_mod: Any = ae
        z = ae_mod.encode(img_tensor)
        recon = ae_mod.decode(z)

    # Compute normalized MSE per pixel
    recon_err = (recon - img_tensor).pow(2).mean().item()

    # Threshold from env or default conservative value
    default_thr = 0.015  # empirically safe; adjustable via env
    raw_thr = os.environ.get('XRV_AE_OOD_THRESHOLD')
    if raw_thr is None:
        thr = default_thr
    else:
        try:
            thr = float(raw_thr)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid XRV_AE_OOD_THRESHOLD=%r; using default %.6f",
                raw_thr,
                default_thr,
            )
            thr = default_thr
    is_ood = recon_err > thr

    return { 'ood_score': float(recon_err), 'is_ood': bool(is_ood), 'threshold': float(thr) }


def process_image(
    image_path: str | Path,
    xray_instance: XRayImage | None = None,
    model_type: str = 'densenet',
) -> dict[str, Any]:
    """
    Process an X-ray image and return predictions
    If xray_instance is provided, will update progress
    """
    # Update status to processing
    if xray_instance:
        xray_instance.processing_status = 'processing'
        xray_instance.progress = 5
        xray_instance.save(update_fields=['processing_status', 'progress'])
    
    # Extract and save image metadata
    if xray_instance:
        metadata = extract_image_metadata(image_path)
        # If the upload was DICOM (converted to PNG for processing), keep the
        # user-facing format as DICOM instead of overwriting with "PNG".
        if not (xray_instance.image_format and str(xray_instance.image_format).upper() == "DICOM"):
            xray_instance.image_format = metadata.format
        xray_instance.image_size = metadata.size
        xray_instance.image_resolution = metadata.resolution
        xray_instance.image_date_created = metadata.date_created
        xray_instance.save(update_fields=['image_format', 'image_size', 'image_resolution', 'image_date_created'])
    
    # Load and preprocess the image
    # Update progress to 10%
    if xray_instance:
        xray_instance.progress = 10
        xray_instance.save(update_fields=['progress'])
    
    # Ensure image_path is a Path object.
    image_path = Path(image_path)
    
    # Normalize the image
    # Update progress to 20%
    if xray_instance:
        xray_instance.progress = 20
        xray_instance.save(update_fields=['progress'])
    
    # TorchXRayVision helper: supports DICOM + normal images and returns the
    # expected model input (normalized, shape (1, H, W)).
    img = _load_xrv_image(image_path)

    # OOD gate (reconstruction error)
    ood = compute_ood_score(img[0])  # pass (H, W)
    if xray_instance:
        # Flag for review if likely OOD
        xray_instance.requires_expert_review = bool(ood.get('is_ood', False))
        xray_instance.save(update_fields=['requires_expert_review'])
    
    # Load model and get resize dimension
    # Update progress to 40%
    if xray_instance:
        xray_instance.progress = 40
        xray_instance.save(update_fields=['progress'])
    
    model, resize_dim = load_model(model_type)
    model = cast(torch.nn.Module, model)
    
    # Apply transforms for model: center-crop then resize.
    transform = get_xrv_transform(resize_dim)
    img = transform(img)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img)
    
    # Add batch dimension
    if len(img_tensor.shape) < 3:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model = model.to(device)
    
    # Update progress to 60%
    if xray_instance:
        xray_instance.progress = 60
        xray_instance.save(update_fields=['progress'])
    
    # Get predictions
    # Update progress to 75%
    if xray_instance:
        xray_instance.progress = 75
        xray_instance.save(update_fields=['progress'])
    
    with torch.inference_mode():
        # Use the model's forward method for both model types
        preds = model(img_tensor).cpu()
    
    # Create a dictionary of pathology predictions
    # For ResNet, ALWAYS use default_pathologies for correct mapping
    if 'resnet' in model_type:
        # ResNet model outputs 18 values in the order of default_pathologies
        results = dict(zip(list(xrv.datasets.default_pathologies), preds[0].detach().numpy()))
    else:
        # For DenseNet, we can use the model's pathologies directly (access dynamically)
        model_pathologies: list[str] = list(getattr(model, 'pathologies', list(xrv.datasets.default_pathologies)))
        results = dict(zip(model_pathologies, preds[0].detach().numpy()))
    
    # Filter out specific classes for ResNet if needed
    # Note: These classes will always output 0.5 for ResNet as they're not trained
    if 'resnet' in model_type:
        excluded_classes = ["Enlarged Cardiomediastinum", "Lung Lesion"]
        results = {k: v for k, v in results.items() if k not in excluded_classes}
    
    # Apply specific multiplier for resnet50-res512-all
    if model_type == 'resnet50-res512-all':
        results = {k: min(float(v) * 2.0, 1.0) for k, v in results.items()}

    # Apply calibration and thresholds
    results, calib_meta = apply_calibration_and_thresholds(results)

    # If we have an XRay instance, report "almost done".
    if xray_instance:
        xray_instance.progress = 90
        xray_instance.save(update_fields=['progress'])
    
    # Attach OOD + calibration metadata (not persisted to DB fields)
    results["OOD_Score"] = ood.get('ood_score')
    results["OOD_Threshold"] = ood.get('threshold')
    results["Is_OOD"] = ood.get('is_ood')
    results["Calibration"] = {
        'temperatures': calib_meta.get('calibration_temperatures'),
        'thresholds': calib_meta.get('decision_thresholds'),
        'labels': calib_meta.get('predicted_labels'),
    }

    return results


def apply_segmentation(image_path: str) -> dict[str, Any]:
    """
    Apply anatomical segmentation to a chest X-ray image using PSPNet
    """
    logger.info(f"Applying segmentation to image: {image_path}")
    
    # Load the segmentation model
    seg_model, resize_dim = load_segmentation_model()
    
    # Load + normalize image (supports DICOM via pydicom).
    img = _load_xrv_image(image_path)

    # OOD gate (reconstruction error)
    ood = compute_ood_score(img[0])  # pass (H, W)
    
    # Preserve original image for visualization (2D)
    original_img = img[0].copy()
    
    # Resize to model input size (typically 512x512 for PSPNet)
    transform = get_xrv_transform(resize_dim)
    
    # Apply transform to numpy array first
    img_transformed = transform(img)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_transformed).float().unsqueeze(0)
    
    # Perform segmentation
    with torch.no_grad():
        output = seg_model(img_tensor)
    
    # Output shape is [1, 14, 512, 512]
    # 14 channels for different anatomical structures
    segmentation_masks = output.squeeze(0).cpu().numpy()
    
    # Define the anatomical structures
    anatomical_structures = [
        'Left Clavicle', 'Right Clavicle', 'Left Scapula', 'Right Scapula',
        'Left Lung', 'Right Lung', 'Left Hilus Pulmonis', 'Right Hilus Pulmonis',
        'Heart', 'Aorta', 'Facies Diaphragmatica', 'Mediastinum',
        'Weasand', 'Spine'
    ]
    
    # Apply sigmoid to get probabilities
    segmentation_probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
    
    # Create binary masks with threshold
    threshold = 0.5
    binary_masks = (segmentation_probs > threshold).astype(np.uint8)
    
    # Store results
    results = {
        'segmentation_masks': segmentation_masks,
        'segmentation_probs': segmentation_probs,
        'binary_masks': binary_masks,
        'anatomical_structures': anatomical_structures,
        'original': original_img,
        'num_structures': len(anatomical_structures),
        'OOD_Score': ood.get('ood_score'),
        'OOD_Threshold': ood.get('threshold'),
        'Is_OOD': ood.get('is_ood'),
    }
    
    logger.info("Segmentation completed successfully")
    return results


def save_segmentation_results(xray_instance: XRayImage, results: dict[str, Any], model_type: str = 'pspnet') -> None:
    """Save segmentation results and create VisualizationResult entries."""
    from django.conf import settings
    
    output_dir = Path(settings.MEDIA_ROOT) / 'segmentation' / str(xray_instance.pk)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Save combined visualization
    combined_filename = f"segmentation_{xray_instance.pk}_all_structures.png"
    combined_path = output_dir / combined_filename
    save_segmentation_visualization(results, combined_path)
    
    # Save individual masks
    mask_paths = save_individual_segmentation_masks(results, masks_dir)
    
    # Check for OOD and prepare metadata
    if results.get("Is_OOD"):
        xray_instance.requires_expert_review = True
        xray_instance.save(update_fields=['requires_expert_review'])

    ood_metadata = {
        'ood_score': results.get("OOD_Score"),
        'ood_threshold': results.get("OOD_Threshold"),
        'is_ood': results.get("Is_OOD"),
    }

    # Store visualization results for each structure
    for idx, structure_name in enumerate(results['anatomical_structures']):
        # Get confidence score (max probability in the mask)
        confidence = float(results['segmentation_probs'][idx].max())
        
        # Create visualization result entry
        VisualizationResult.objects.update_or_create(
            xray=xray_instance,
            visualization_type='segmentation',
            target_pathology=structure_name,
            defaults={
                'model_used': model_type,
                'visualization_path': f"segmentation/{xray_instance.pk}/{combined_filename}",
                'confidence_score': confidence,
                'metadata': {
                    'structure_index': idx,
                    'mask_path': f"segmentation/{xray_instance.pk}/masks/{Path(mask_paths.get(structure_name, '')).name}",
                    'threshold': 0.5,
                    **ood_metadata,
                }
            },
        )
    
    # Store combined result
    VisualizationResult.objects.update_or_create(
        xray=xray_instance,
        visualization_type='segmentation_combined',
        target_pathology='All Structures',
        defaults={
            'model_used': model_type,
            'visualization_path': f"segmentation/{xray_instance.pk}/{combined_filename}",
            'metadata': {
                'num_structures': len(results['anatomical_structures']),
                'structures': results['anatomical_structures'],
                **ood_metadata,
            }
        },
    )


def save_and_record_visualization(
    xray_instance: XRayImage,
    model_type: str,
    results: dict[str, Any],
    method: str,
) -> None:
    """Helper to save files and create DB records for visualizations."""
    from django.conf import settings
    
    base_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / method
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if method == 'gradcam':
        target = results['target_class']
        filenames = {
            'combined_filename': (f"gradcam_{xray_instance.id}_{target}.png", save_interpretability_visualization),
            'heatmap_filename': (f"heatmap_{xray_instance.id}_{target}.png", save_heatmap),
            'overlay_filename': (f"gradcam_overlay_{xray_instance.id}_{target}.png", save_overlay),
        }
        legacy_map = {
            'combined_filename': 'gradcam_visualization',
            'heatmap_filename': 'gradcam_heatmap',
            'overlay_filename': 'gradcam_overlay',
        }
        target_value = target
        legacy_has_field = 'has_gradcam'
        legacy_target_field = 'gradcam_target_class'
        visualization_type = 'gradcam'
        
    elif method == 'pli':
        target = results['target_class']
        filenames = {
            'saliency_filename': (f"pli_{xray_instance.id}_{target}.png", save_interpretability_visualization),
            'separate_saliency_filename': (f"pli_saliency_{xray_instance.id}_{target}.png", save_saliency_map),
            'overlay_filename': (f"pli_overlay_{xray_instance.id}_{target}.png", save_overlay_visualization),
        }
        legacy_map = {
            'saliency_filename': 'pli_visualization',
            'separate_saliency_filename': 'pli_saliency_map',
            'overlay_filename': 'pli_overlay_visualization',
        }
        target_value = target
        legacy_has_field = 'has_pli'
        legacy_target_field = 'pli_target_class'
        visualization_type = 'pli'
        
    elif method == 'combined_gradcam':
        threshold = results['threshold']
        filenames = {
            'combined_filename': (f"combined_gradcam_{xray_instance.id}_threshold_{threshold}.png", save_interpretability_visualization),
            'heatmap_filename': (f"combined_heatmap_{xray_instance.id}_threshold_{threshold}.png", save_heatmap),
            'overlay_filename': (f"combined_gradcam_overlay_{xray_instance.id}_threshold_{threshold}.png", save_overlay),
        }
        legacy_map = {
            'combined_filename': 'gradcam_visualization',
            'heatmap_filename': 'gradcam_heatmap',
            'overlay_filename': 'gradcam_overlay',
        }
        target_value = results['pathology_summary']
        legacy_has_field = 'has_gradcam'
        legacy_target_field = 'gradcam_target_class'
        visualization_type = 'combined_gradcam'
        
    elif method == 'combined_pli':
        threshold = results['threshold']
        filenames = {
            'combined_filename': (f"combined_pli_{xray_instance.id}_threshold_{threshold}.png", save_interpretability_visualization),
            'saliency_filename': (f"combined_pli_saliency_{xray_instance.id}_threshold_{threshold}.png", save_saliency_map),
            'overlay_filename': (f"combined_pli_overlay_{xray_instance.id}_threshold_{threshold}.png", save_overlay_visualization),
        }
        legacy_map = {
            'combined_filename': 'pli_visualization',
            'saliency_filename': 'pli_saliency_map',
            'overlay_filename': 'pli_overlay_visualization',
        }
        target_value = results['pathology_summary']
        legacy_has_field = 'has_pli'
        legacy_target_field = 'pli_target_class'
        visualization_type = 'combined_pli'
    else:
        raise ValueError(f"Unknown interpretation method: {method}")

    # Process files
    viz_data = {}
    if 'threshold' in results:
        viz_data['threshold'] = results['threshold']
        
    rel_prefix = f"interpretability/{method}/"
    
    defaults = {'model_used': model_type}
    
    for key, (fname, saver) in filenames.items():
        full_path = base_dir / fname
        logger.info(f"Saving {method} result {key} to {full_path}")
        saver(results, full_path)
        
        rel_path = f"{rel_prefix}{fname}"
        viz_data[key] = rel_path
        results[key] = rel_path  # Update results for caller if needed
        
        # Update legacy field
        if key in legacy_map:
            setattr(xray_instance, legacy_map[key], rel_path)
            
        # Map to VisualizationResult fields
        if visualization_type in ['gradcam', 'combined_gradcam']:
             if key == 'combined_filename': defaults['visualization_path'] = rel_path
             if key == 'heatmap_filename': defaults['heatmap_path'] = rel_path
             if key == 'overlay_filename': defaults['overlay_path'] = rel_path
        elif visualization_type in ['pli', 'combined_pli']:
             if key == 'saliency_filename' or key == 'combined_filename': defaults['visualization_path'] = rel_path
             if key == 'separate_saliency_filename' or key == 'saliency_filename': defaults['saliency_path'] = rel_path
             if key == 'overlay_filename': defaults['overlay_path'] = rel_path

    if 'threshold' in results:
        defaults['threshold'] = results['threshold']

    # Create/Update VisualizationResult
    VisualizationResult.objects.update_or_create(
        xray=xray_instance,
        visualization_type=visualization_type,
        target_pathology=target_value,
        defaults=defaults
    )
    
    # Finalize legacy flags
    setattr(xray_instance, legacy_has_field, True)
    setattr(xray_instance, legacy_target_field, target_value)
