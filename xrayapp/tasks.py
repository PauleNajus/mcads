from __future__ import annotations

from pathlib import Path
from typing import Optional

from celery import shared_task
from django.conf import settings
from django.utils import timezone
from django.conf import settings

from .models import XRayImage, PredictionHistory, VisualizationResult
from .utils import (
    process_image,
    save_interpretability_visualization,
    save_overlay_visualization,
    save_saliency_map,
    save_heatmap,
    save_overlay,
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

    # Persist predictions
    xray.atelectasis = results.get('Atelectasis')
    xray.cardiomegaly = results.get('Cardiomegaly')
    xray.consolidation = results.get('Consolidation')
    xray.edema = results.get('Edema')
    xray.effusion = results.get('Effusion')
    xray.emphysema = results.get('Emphysema')
    xray.fibrosis = results.get('Fibrosis')
    xray.hernia = results.get('Hernia')
    xray.infiltration = results.get('Infiltration')
    xray.mass = results.get('Mass')
    xray.nodule = results.get('Nodule')
    xray.pleural_thickening = results.get('Pleural_Thickening')
    xray.pneumonia = results.get('Pneumonia')
    xray.pneumothorax = results.get('Pneumothorax')
    xray.fracture = results.get('Fracture')
    xray.lung_opacity = results.get('Lung Opacity')

    if 'Enlarged Cardiomediastinum' in results:
        xray.enlarged_cardiomediastinum = results.get('Enlarged Cardiomediastinum')
    if 'Lung Lesion' in results:
        xray.lung_lesion = results.get('Lung Lesion')

    xray.processing_status = 'completed'
    xray.progress = 100
    xray.save(update_fields=[
        'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion',
        'emphysema', 'fibrosis', 'hernia', 'infiltration', 'mass', 'nodule',
        'pleural_thickening', 'pneumonia', 'pneumothorax', 'fracture', 'lung_opacity',
        'enlarged_cardiomediastinum', 'lung_lesion', 'processing_status', 'progress'
    ])

    # Create/update prediction history
    history = PredictionHistory(
        user=xray.user,
        xray=xray,
        model_used=model_type,
        atelectasis=xray.atelectasis,
        cardiomegaly=xray.cardiomegaly,
        consolidation=xray.consolidation,
        edema=xray.edema,
        effusion=xray.effusion,
        emphysema=xray.emphysema,
        fibrosis=xray.fibrosis,
        hernia=xray.hernia,
        infiltration=xray.infiltration,
        mass=xray.mass,
        nodule=xray.nodule,
        pleural_thickening=xray.pleural_thickening,
        pneumonia=xray.pneumonia,
        pneumothorax=xray.pneumothorax,
        fracture=xray.fracture,
        lung_opacity=xray.lung_opacity,
        enlarged_cardiomediastinum=xray.enlarged_cardiomediastinum,
        lung_lesion=xray.lung_lesion,
        severity_level=xray.severity_level,
    )
    if xray.user:
        history.save()
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

    # Update most recent prediction history to keep in sync
    hist = PredictionHistory.objects.filter(xray=xray).order_by('-created_at').first()
    if hist:
        hist.atelectasis = xray.atelectasis
        hist.cardiomegaly = xray.cardiomegaly
        hist.consolidation = xray.consolidation
        hist.edema = xray.edema
        hist.effusion = xray.effusion
        hist.emphysema = xray.emphysema
        hist.fibrosis = xray.fibrosis
        hist.hernia = xray.hernia
        hist.infiltration = xray.infiltration
        hist.mass = xray.mass
        hist.nodule = xray.nodule
        hist.pleural_thickening = xray.pleural_thickening
        hist.pneumonia = xray.pneumonia
        hist.pneumothorax = xray.pneumothorax
        hist.fracture = xray.fracture
        hist.lung_opacity = xray.lung_opacity
        hist.enlarged_cardiomediastinum = xray.enlarged_cardiomediastinum
        hist.lung_lesion = xray.lung_lesion
        hist.severity_level = xray.severity_level
        hist.save()

    return xray.pk


