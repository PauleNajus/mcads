from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from xrayapp.models import (
    XRayImage,
    VisualizationResult,
)
from xrayapp.tasks import run_interpretability_task, run_segmentation_task
from .utils import (
    _get_user_hospital,
    process_with_interpretability_async,
    process_segmentation_async,
)

logger = logging.getLogger(__name__)


@login_required
@require_POST
def delete_visualization(request: HttpRequest, pk: int) -> JsonResponse:
    """Delete a visualization result"""
    try:
        # Get the visualization
        visualization = VisualizationResult.objects.with_xray_user_profile().get(pk=pk)
        
        # Check if user has permission to delete (must be from same hospital)
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return JsonResponse({'success': False, 'error': _('Permission denied')}, status=403)
        owner_user = visualization.xray.user
        if owner_user is None:
            owner_profile = None
        else:
            try:
                owner_profile = owner_user.profile  # type: ignore[attr-defined]
            except ObjectDoesNotExist:
                owner_profile = None
        owner_hospital = getattr(owner_profile, 'hospital', None)
        if owner_hospital != user_hospital:
            return JsonResponse({'success': False, 'error': _('Permission denied')}, status=403)
        
        # Delete associated files
        file_paths = [
            visualization.visualization_path,
            visualization.heatmap_path,
            visualization.overlay_path,
            visualization.saliency_path,
        ]
        media_root = Path(settings.MEDIA_ROOT).resolve()
        for file_path in file_paths:
            if file_path:
                try:
                    # Defensive: ensure we only delete within MEDIA_ROOT even if the
                    # DB contains unexpected paths.
                    candidate = Path(str(file_path))
                    full_path = (candidate if candidate.is_absolute() else (media_root / candidate)).resolve()
                    if media_root not in full_path.parents:
                        logger.warning("Refusing to delete path outside MEDIA_ROOT: %s", full_path)
                        continue
                    if full_path.exists():
                        full_path.unlink()
                except Exception:
                    logger.warning("Error deleting file %s", file_path, exc_info=True)
        
        # Delete the visualization record
        visualization.delete()
        
        return JsonResponse({'success': True})
        
    except VisualizationResult.DoesNotExist:
        return JsonResponse({'success': False, 'error': _('Visualization not found')}, status=404)
    except Exception as e:
        logger.exception("Error deleting visualization pk=%s", pk)
        payload: dict[str, Any] = {
            'success': False,
            'error': _('An error occurred while deleting the visualization.'),
        }
        if settings.DEBUG:
            payload['details'] = str(e)
        return JsonResponse(payload, status=500)


@login_required
def generate_interpretability(request: HttpRequest, pk: int) -> HttpResponse:
    """Generate interpretability visualization for an X-ray image"""
    # Get user's hospital from profile
    user_hospital = _get_user_hospital(request.user)
    if user_hospital is None:
        # This endpoint is often called via fetch(); return JSON for AJAX callers.
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json':
            return JsonResponse(
                {
                    'status': 'error',
                    'error': _('Permission denied'),
                    'message': _('Your user profile is missing a hospital assignment.'),
                },
                status=403,
            )
        return redirect('home')
    
    # Allow access to any X-ray from the same hospital
    xray_instance = XRayImage.objects.for_hospital(user_hospital).with_user_profile().get(pk=pk)
    
    # Get parameters from request
    interpretation_method = request.GET.get('method', 'gradcam')  # Default to Grad-CAM
    # Default to the model used for the prediction that produced this results page.
    # This prevents "wrong" interpretability results when users run inference with
    # ResNet but the UI doesn't pass `model_type` (historically defaulted to DenseNet).
    model_type = request.GET.get('model_type') or getattr(xray_instance, 'model_used', None) or 'densenet'
    target_class = request.GET.get('target_class', None)  # Default to None (use highest probability class)
    
    # Reset progress to 0 and set status to processing
    xray_instance.progress = 0
    xray_instance.processing_status = 'processing'
    xray_instance.save()
    
    # Get the image path
    image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
    
    # Start background processing (prefer Celery only if explicitly enabled)
    use_celery = settings.USE_CELERY
    started = False
    if use_celery:
        try:
            run_interpretability_task.delay(xray_instance.pk, model_type, interpretation_method, target_class)
            started = True
        except Exception as e:
            logger.warning(f"Celery unavailable; falling back to thread: {e}")
    if not started:
        thread = threading.Thread(
            target=process_with_interpretability_async,
            args=(image_path, xray_instance, model_type, interpretation_method, target_class),
            daemon=True,
        )
        thread.start()
    
    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json':
        # Return JSON response for AJAX requests
        return JsonResponse({
            'status': 'started',
            'message': _('Interpretability generation started'),
            'xray_id': xray_instance.pk
        })
    
    # Redirect to the results page for non-AJAX requests
    return redirect('xray_results', pk=pk)


@login_required
def generate_segmentation(request: HttpRequest, pk: int) -> HttpResponse:
    """Generate anatomical segmentation for an X-ray image"""
    # Get user's hospital from profile
    user_hospital = _get_user_hospital(request.user)
    if user_hospital is None:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json':
            return JsonResponse(
                {
                    'status': 'error',
                    'error': _('Permission denied'),
                    'message': _('Your user profile is missing a hospital assignment.'),
                },
                status=403,
            )
        return redirect('home')
    
    # Allow access to any X-ray from the same hospital
    xray_instance = XRayImage.objects.for_hospital(user_hospital).with_user_profile().get(pk=pk)
    
    # Reset progress to 0 and set status to processing
    xray_instance.progress = 0
    xray_instance.processing_status = 'processing'
    xray_instance.save()
    
    # Start background processing (prefer Celery only if explicitly enabled)
    use_celery = settings.USE_CELERY
    started = False
    if use_celery:
        try:
            run_segmentation_task.delay(xray_instance.pk)
            started = True
        except Exception as e:
            logger.warning(f"Celery unavailable; falling back to thread: {e}")
    if not started:
        # Fallback to threaded processing
        thread = threading.Thread(
            target=process_segmentation_async,
            args=(xray_instance,),
            daemon=True,
        )
        thread.start()
    
    # Check if this is an AJAX request
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json':
        # Return JSON response for AJAX requests
        return JsonResponse({
            'status': 'started',
            'message': _('Segmentation processing started'),
            'xray_id': xray_instance.pk
        })
    
    # Redirect to the results page for non-AJAX requests
    return redirect('xray_results', pk=pk)
