from __future__ import annotations

import logging
from typing import Optional

from celery import shared_task
from .models import XRayImage
from .services import perform_inference, perform_interpretability, perform_segmentation

logger = logging.getLogger(__name__)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_inference_task(self, xray_id: int, model_type: str = 'densenet') -> Optional[int]:
    """Celery task to process a single XRayImage by id.

    Returns the XRayImage id on success, or None if not found.
    """
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        logger.warning(f"XRayImage {xray_id} not found for inference task")
        return None

    try:
        perform_inference(xray, model_type)
        return xray.pk
    except Exception:
        # Mark as error only when retries are exhausted (avoid flickering UI).
        retries = int(getattr(self.request, "retries", 0) or 0)
        max_retries = getattr(self, "max_retries", None)
        
        # If we can't determine retries, err on the side of not marking failure yet
        # unless it's obviously the last attempt.
        if max_retries is not None and retries >= int(max_retries):
            logger.error(f"Inference task failed permanently for X-ray {xray_id} after {retries} retries.")
            xray.processing_status = 'error'
            xray.save(update_fields=['processing_status'])
        # Re-raise to trigger Celery retry
        raise


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_interpretability_task(self, xray_id: int, model_type: str = 'densenet', interpretation_method: str = 'gradcam', target_class: str | None = None) -> Optional[int]:
    """Celery task to generate interpretability visualizations and persist files + records."""
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        logger.warning(f"XRayImage {xray_id} not found for interpretability task")
        return None

    try:
        perform_interpretability(xray, model_type, interpretation_method, target_class)
        return xray.pk
    except Exception:
        retries = int(getattr(self.request, "retries", 0) or 0)
        max_retries = getattr(self, "max_retries", None)
        
        if max_retries is not None and retries >= int(max_retries):
            logger.error(f"Interpretability task failed permanently for X-ray {xray_id} after {retries} retries.")
            # We don't explicitly set status to error here because perform_interpretability
            # handles its own status updates, but we log the final failure.
        raise


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def run_segmentation_task(self, xray_id: int) -> Optional[int]:
    """Celery task to perform anatomical segmentation on an X-ray image."""
    try:
        xray = XRayImage.objects.get(pk=xray_id)
    except XRayImage.DoesNotExist:
        logger.warning(f"XRayImage {xray_id} not found for segmentation task")
        return None
    
    try:
        perform_segmentation(xray)
        return xray.pk
    except Exception:
        retries = int(getattr(self.request, "retries", 0) or 0)
        max_retries = getattr(self, "max_retries", None)
        
        if max_retries is not None and retries >= int(max_retries):
            logger.error(f"Segmentation task failed permanently for X-ray {xray_id} after {retries} retries.")
        raise
