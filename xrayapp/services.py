from __future__ import annotations

import logging
from pathlib import Path
from django.conf import settings
from .models import PATHOLOGY_FIELDS, XRayImage, PredictionHistory
from .inference_logic import (
    process_image,
    apply_segmentation,
    save_segmentation_results,
    save_and_record_visualization,
)
from .interpretability import (
    apply_gradcam,
    apply_pixel_interpretability,
    apply_combined_gradcam,
    apply_combined_pixel_interpretability,
)

logger = logging.getLogger(__name__)


def perform_inference(xray_instance: XRayImage, model_type: str = 'densenet') -> None:
    """
    Run full inference pipeline: process image, update xray record, create history.
    
    This function encapsulates the core business logic for inference, allowing it
    to be called from both Celery tasks and synchronous threads (fallback).
    """
    try:
        image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
        
        # Process image (this updates progress from 5% to ~90%)
        # Note: process_image handles OOD checking and metadata extraction internally
        results = process_image(image_path, xray_instance, model_type)
        
        # Persist predictions (single normalization point)
        xray_instance.apply_predictions_from_results(results)
        xray_instance.severity_level = xray_instance.calculate_severity_level
        xray_instance.processing_status = 'completed'
        xray_instance.progress = 100
        xray_instance.save(update_fields=[
            *PATHOLOGY_FIELDS,
            'severity_level',
            'processing_status',
            'progress',
        ])
        
        # Create prediction history snapshot (if user exists).
        PredictionHistory.create_from_xray(xray_instance, model_type)
        
        logger.info(f"Inference completed successfully for X-ray {xray_instance.pk}")
        
    except Exception:
        # Log the full traceback for debugging/operations.
        logger.exception(f"Inference pipeline failed for X-ray {xray_instance.pk}")
        xray_instance.processing_status = 'error'
        xray_instance.save(update_fields=['processing_status'])
        # Re-raise so the caller (Task/Thread) knows it failed
        raise


def perform_interpretability(
    xray_instance: XRayImage, 
    model_type: str, 
    method: str, 
    target_class: str | None = None
) -> None:
    """
    Run interpretability pipeline: generate visualizations, save files, record results.
    """
    try:
        image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
        
        # Set initial progress
        xray_instance.progress = 10
        xray_instance.processing_status = 'processing'
        xray_instance.save(update_fields=['progress', 'processing_status'])
        
        logger.info(f"Starting {method} visualization for X-ray {xray_instance.pk}")
        
        # Update progress to show model loading
        xray_instance.progress = 20
        xray_instance.save(update_fields=['progress'])

        results = {}
        if method == 'gradcam':
            logger.info(f"Applying Grad-CAM for image {xray_instance.pk}")
            results = apply_gradcam(str(image_path), model_type, target_class)
            results['method'] = 'gradcam'
            
        elif method == 'pli':
            results = apply_pixel_interpretability(str(image_path), model_type, target_class)
            results['method'] = 'pli'
            
        elif method == 'combined_gradcam':
            results = apply_combined_gradcam(str(image_path), model_type)
            results['method'] = 'combined_gradcam'

        elif method == 'combined_pli':
            results = apply_combined_pixel_interpretability(str(image_path), model_type)
            results['method'] = 'combined_pli'
            
        else:
            raise ValueError(f"Unknown interpretation method: {method}")
            
        # Update progress after computation
        xray_instance.progress = 60
        xray_instance.save(update_fields=['progress'])
        
        # Save results using shared helper
        save_and_record_visualization(xray_instance, model_type, results, method)
        
        # Finalize
        xray_instance.progress = 100
        xray_instance.processing_status = 'completed'
        xray_instance.save(update_fields=['progress', 'processing_status'])
        
        # Keep the most recent history entry in sync with any updated model_used.
        PredictionHistory.update_latest_for_xray(xray_instance, model_used=model_type)
        
        logger.info(f"Interpretability {method} completed for X-ray {xray_instance.pk}")

    except Exception:
        logger.exception(f"Interpretability pipeline failed for X-ray {xray_instance.pk}")
        xray_instance.processing_status = 'error'
        xray_instance.save(update_fields=['processing_status'])
        raise


def perform_segmentation(xray_instance: XRayImage) -> None:
    """
    Run segmentation pipeline: generate masks, save visualizations.
    """
    try:
        image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
        
        xray_instance.progress = 10
        xray_instance.processing_status = 'processing'
        xray_instance.save(update_fields=['progress', 'processing_status'])
        
        logger.info(f"Starting segmentation for X-ray {xray_instance.pk}")
        
        # Update progress to show model loading
        xray_instance.progress = 20
        xray_instance.save(update_fields=['progress'])
        
        # Apply segmentation
        results = apply_segmentation(str(image_path))
        
        # Update progress after segmentation
        xray_instance.progress = 60
        xray_instance.save(update_fields=['progress'])
        
        # Save results using shared helper
        save_segmentation_results(xray_instance, results)
        
        # Finalize
        xray_instance.progress = 100
        xray_instance.processing_status = 'completed'
        xray_instance.save(update_fields=['progress', 'processing_status'])
        
        logger.info(f"Segmentation completed for X-ray {xray_instance.pk}")
        
    except Exception:
        logger.exception(f"Segmentation pipeline failed for X-ray {xray_instance.pk}")
        xray_instance.processing_status = 'error'
        xray_instance.save(update_fields=['processing_status'])
        raise
