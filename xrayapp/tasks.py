from __future__ import annotations

from pathlib import Path
from typing import Optional

from celery import shared_task
from django.conf import settings

from .models import PATHOLOGY_FIELDS, XRayImage, PredictionHistory, VisualizationResult
from .utils import (
    process_image,
    save_interpretability_visualization,
    save_overlay_visualization,
    save_saliency_map,
    save_heatmap,
    save_overlay,
    apply_segmentation,
    save_segmentation_visualization,
    save_individual_segmentation_masks,
)
from .interpretability import (
    apply_gradcam,
    apply_pixel_interpretability,
    apply_combined_gradcam,
    apply_combined_pixel_interpretability,
)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_inference_task(self, xray_id: int, model_type: str = 'densenet') -> Optional[int]:
    """Celery task to process a single XRayImage by id.

    Returns the XRayImage id on success, or None if not found.
    """
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        return None

    image_path = Path(settings.MEDIA_ROOT) / xray.image.name
    results = process_image(image_path, xray, model_type)

    # Persist predictions (single normalization point).
    xray.apply_predictions_from_results(results)
    xray.severity_level = xray.calculate_severity_level
    xray.processing_status = 'completed'
    xray.progress = 100
    xray.save(update_fields=[
        *PATHOLOGY_FIELDS,
        'severity_level',
        'processing_status',
        'progress',
    ])

    # Create prediction history snapshot (if user exists).
    PredictionHistory.create_from_xray(xray, model_type)
    return xray.pk


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_interpretability_task(self, xray_id: int, model_type: str = 'densenet', interpretation_method: str = 'gradcam', target_class: str | None = None) -> Optional[int]:
    """Celery task to generate interpretability visualizations and persist files + records."""
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        return None

    image_path = Path(settings.MEDIA_ROOT) / xray.image.name

    # Set processing flags
    xray.progress = 10
    xray.processing_status = 'processing'
    xray.save(update_fields=['progress', 'processing_status'])
    
    # Update progress to show model loading
    xray.progress = 20
    xray.save(update_fields=['progress'])
    
    # Log the start of interpretability processing
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting interpretability task for image {xray_id}, method: {interpretation_method}")

    # Compute interpretability
    try:
        if interpretation_method == 'gradcam':
            logger.info(f"Applying Grad-CAM for image {xray_id}")
            results = apply_gradcam(str(image_path), model_type, target_class)
            results['method'] = 'gradcam'
            logger.info(f"Grad-CAM completed for image {xray_id}")
            
            # Update progress after computation
            xray.progress = 60
            xray.save(update_fields=['progress'])
            
            output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'gradcam'
            output_dir.mkdir(parents=True, exist_ok=True)

            combined_filename = f"gradcam_{xray.pk}_{results['target_class']}.png"
            heatmap_filename = f"heatmap_{xray.pk}_{results['target_class']}.png"
            overlay_filename = f"gradcam_overlay_{xray.pk}_{results['target_class']}.png"

            save_interpretability_visualization(results, output_dir / combined_filename)
            save_heatmap(results, output_dir / heatmap_filename)
            save_overlay(results, output_dir / overlay_filename)

            VisualizationResult.objects.update_or_create(
                xray=xray,
                visualization_type='gradcam',
                target_pathology=results['target_class'],
                defaults={
                    'model_used': model_type,
                    'visualization_path': f"interpretability/gradcam/{combined_filename}",
                    'heatmap_path': f"interpretability/gradcam/{heatmap_filename}",
                    'overlay_path': f"interpretability/gradcam/{overlay_filename}",
                },
            )

        elif interpretation_method == 'pli':
            results = apply_pixel_interpretability(str(image_path), model_type, target_class)
            results['method'] = 'pli'
            
            # Update progress after computation
            xray.progress = 60
            xray.save(update_fields=['progress'])
        
            output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'pli'
            output_dir.mkdir(parents=True, exist_ok=True)

            saliency_filename = f"pli_{xray.pk}_{results['target_class']}.png"
            overlay_filename = f"pli_overlay_{xray.pk}_{results['target_class']}.png"
            separate_saliency_filename = f"pli_saliency_{xray.pk}_{results['target_class']}.png"

            save_interpretability_visualization(results, output_dir / saliency_filename)
            save_overlay_visualization(results, output_dir / overlay_filename)
            save_saliency_map(results, output_dir / separate_saliency_filename)

            VisualizationResult.objects.update_or_create(
                xray=xray,
                visualization_type='pli',
                target_pathology=results['target_class'],
                defaults={
                    'model_used': model_type,
                    'visualization_path': f"interpretability/pli/{saliency_filename}",
                    'overlay_path': f"interpretability/pli/{overlay_filename}",
                    'saliency_path': f"interpretability/pli/{separate_saliency_filename}",
                },
            )

        elif interpretation_method == 'combined_gradcam':
            results = apply_combined_gradcam(str(image_path), model_type)
            output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'combined_gradcam'
            output_dir.mkdir(parents=True, exist_ok=True)

            combined_filename = f"combined_gradcam_{xray.pk}_threshold_{results['threshold']}.png"
            heatmap_filename = f"combined_heatmap_{xray.pk}_threshold_{results['threshold']}.png"
            overlay_filename = f"combined_gradcam_overlay_{xray.pk}_threshold_{results['threshold']}.png"

            save_interpretability_visualization(results, output_dir / combined_filename)
            save_heatmap(results, output_dir / heatmap_filename)
            save_overlay(results, output_dir / overlay_filename)

            VisualizationResult.objects.update_or_create(
                xray=xray,
                visualization_type='combined_gradcam',
                target_pathology=results['pathology_summary'],
                defaults={
                    'model_used': model_type,
                    'visualization_path': f"interpretability/combined_gradcam/{combined_filename}",
                    'heatmap_path': f"interpretability/combined_gradcam/{heatmap_filename}",
                    'overlay_path': f"interpretability/combined_gradcam/{overlay_filename}",
                    'threshold': results.get('threshold'),
                },
            )

        elif interpretation_method == 'combined_pli':
            results = apply_combined_pixel_interpretability(str(image_path), model_type)
            output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'combined_pli'
            output_dir.mkdir(parents=True, exist_ok=True)

            combined_filename = f"combined_pli_{xray.pk}_threshold_{results['threshold']}.png"
            saliency_filename = f"combined_pli_saliency_{xray.pk}_threshold_{results['threshold']}.png"
            overlay_filename = f"combined_pli_overlay_{xray.pk}_threshold_{results['threshold']}.png"

            save_interpretability_visualization(results, output_dir / combined_filename)
            save_saliency_map(results, output_dir / saliency_filename)
            save_overlay_visualization(results, output_dir / overlay_filename)

            VisualizationResult.objects.update_or_create(
                xray=xray,
                visualization_type='combined_pli',
                target_pathology=results['pathology_summary'],
                defaults={
                    'model_used': model_type,
                    'visualization_path': f"interpretability/combined_pli/{combined_filename}",
                    'saliency_path': f"interpretability/combined_pli/{saliency_filename}",
                    'overlay_path': f"interpretability/combined_pli/{overlay_filename}",
                    'threshold': results.get('threshold'),
                },
            )

    except Exception as e:
        logger.error(f"Interpretability task failed for image {xray_id}: {e}", exc_info=True)
        xray.processing_status = 'error'
        xray.save(update_fields=['processing_status'])
        raise
    
    # Finalize
    xray.progress = 100
    xray.processing_status = 'completed'
    xray.save(update_fields=['progress', 'processing_status'])

    # Keep the most recent history entry in sync with any updated model_used.
    PredictionHistory.update_latest_for_xray(xray, model_used=model_type)

    return xray.pk


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_segmentation_task(self, xray_id: int) -> Optional[int]:
    """Celery task to perform anatomical segmentation on an X-ray image.
    
    Args:
        xray_id: ID of the XRayImage instance
        
    Returns:
        xray_id if successful, None otherwise
    """
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        return None
    
    image_path = Path(settings.MEDIA_ROOT) / xray.image.name
    
    # Set processing flags
    xray.progress = 10
    xray.processing_status = 'processing'
    xray.save(update_fields=['progress', 'processing_status'])
    
    # Log the start of segmentation processing
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Starting segmentation task for image {xray_id}")
    
    # Update progress to show model loading
    xray.progress = 20
    xray.save(update_fields=['progress'])
    
    try:
        # Apply segmentation
        logger.info(f"Applying segmentation for image {xray_id}")
        results = apply_segmentation(str(image_path))
        logger.info(f"Segmentation completed for image {xray_id}")
        
        # Update progress after segmentation
        xray.progress = 60
        xray.save(update_fields=['progress'])
        
        # Create output directories
        output_dir = Path(settings.MEDIA_ROOT) / 'segmentation' / str(xray.pk)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        masks_dir = output_dir / 'masks'
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Save combined visualization
        combined_filename = f"segmentation_{xray.pk}_all_structures.png"
        combined_path = output_dir / combined_filename
        save_segmentation_visualization(results, combined_path)
        
        # Save individual masks
        mask_paths = save_individual_segmentation_masks(results, masks_dir)
        
        # Update progress
        xray.progress = 80
        xray.save(update_fields=['progress'])
        
        # Store visualization results for each structure
        for idx, structure_name in enumerate(results['anatomical_structures']):
            # Get confidence score (max probability in the mask)
            confidence = float(results['segmentation_probs'][idx].max())
            
            # Create visualization result entry
            VisualizationResult.objects.update_or_create(
                xray=xray,
                visualization_type='segmentation',
                target_pathology=structure_name,
                defaults={
                    'model_used': 'pspnet',
                    'visualization_path': f"segmentation/{xray.pk}/{combined_filename}",
                    'confidence_score': confidence,
                    'metadata': {
                        'structure_index': idx,
                        'mask_path': mask_paths.get(structure_name, ''),
                        'threshold': 0.5,
                    }
                },
            )
        
        # Store combined result
        VisualizationResult.objects.update_or_create(
            xray=xray,
            visualization_type='segmentation_combined',
            target_pathology='All Structures',
            defaults={
                'model_used': 'pspnet',
                'visualization_path': f"segmentation/{xray.pk}/{combined_filename}",
                'metadata': {
                    'num_structures': len(results['anatomical_structures']),
                    'structures': results['anatomical_structures'],
                }
            },
        )
        
    except Exception as e:
        logger.error(f"Segmentation task failed for image {xray_id}: {e}", exc_info=True)
        xray.processing_status = 'error'
        xray.save(update_fields=['processing_status'])
        raise
    
    # Finalize
    xray.progress = 100
    xray.processing_status = 'completed'
    xray.save(update_fields=['progress', 'processing_status'])
    
    logger.info(f"Segmentation task completed successfully for image {xray_id}")
    return xray.pk


