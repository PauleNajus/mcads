from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import gettext_lazy as _

from xrayapp.forms import XRayUploadForm
from xrayapp.models import (
    PATHOLOGY_FIELDS,
    RESULT_KEY_TO_FIELD,
    UserProfile,
    VisualizationResult,
    XRayImage,
)
from xrayapp.pathology import PATHOLOGY_EXPLANATIONS
from xrayapp.tasks import run_inference_task
from .utils import (
    _get_user_hospital,
    _serialize_visualization,
    process_image_async,
)

logger = logging.getLogger(__name__)


def is_ajax(request: HttpRequest) -> bool:
    """Check if request is AJAX (supports standard header or Accept header)"""
    return (
        request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
        'application/json' in request.headers.get('Accept', '')
    )

@login_required
def home(request: HttpRequest) -> HttpResponse:
    """
    Home page with image upload form.
    
    Only users with Administrator, Radiographer, or Technologist roles
    are allowed to upload X-ray images. Radiologists can view but not upload.
    """
    if request.method == 'POST':
        # Check if user has permission to upload X-rays
        # Allowed roles: Administrator, Radiographer, Technologist
        try:
            if not request.user.profile.can_upload_xrays():
                if is_ajax(request):
                    return JsonResponse({
                        'error': _('You do not have permission to upload X-ray images. Please contact your administrator.')
                    }, status=403)
                else:
                    messages.error(request, _('You do not have permission to upload X-ray images.'))
                    return redirect('home')
        except (AttributeError, ObjectDoesNotExist):
            # User doesn't have a profile - create one with default role
            logger.warning("User %s doesn't have a profile. Creating one.", request.user.username)
            UserProfile.objects.create(user=request.user, role='Radiographer')
            # Continue with upload since default role allows it
        
        form = XRayUploadForm(request.POST, request.FILES, user=request.user)
        
        # Debug logging for troubleshooting
        # Avoid noisy INFO logs (and potential PII leakage) in production.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Upload attempt by user=%s (id=%s)",
                request.user.username,
                request.user.pk,
            )
            logger.debug("Form fields received: %s", list(request.POST.keys()))
            logger.debug("Files received: %s", list(request.FILES.keys()))
        
        if form.is_valid():
            logger.info(f"Form validation PASSED for user {request.user.username}")
            logger.info(f"Request headers: {request.headers}")
            logger.info(f"Is AJAX: {is_ajax(request)}")
            
            # Create a new XRayImage instance with all form data but don't save yet
            xray_instance = form.save(commit=False)
            # Assign the current user
            xray_instance.user = request.user
            # Set the requires_expert_review field (default to False)
            xray_instance.requires_expert_review = False
            # NOTE: CharField/TextField defaults are empty strings; no need to
            # manually backfill blanks here.
            
            # Set model_used field
            model_type = request.POST.get('model_type', 'densenet')
            xray_instance.model_used = model_type

            # Preserve source format for DICOM uploads (stored and analyzed as DICOM).
            source_format = getattr(form, "_mcads_source_format", None)
            if source_format:
                xray_instance.image_format = str(source_format)
            # Now save the instance
            try:
                xray_instance.save()
            except Exception as e:
                # Log server-side details; don't leak internal exceptions to clients.
                logger.exception("Error saving XRayImage for user=%s", request.user.pk)
                
                # Return error response for AJAX requests
                if is_ajax(request):
                    payload: dict[str, Any] = {
                        'error': _('Database error occurred while saving image'),
                    }
                    if settings.DEBUG:
                        payload['details'] = str(e)
                    return JsonResponse(payload, status=500)
                # For non-AJAX requests, re-raise the exception
                raise

            # Set immediate queued state to avoid UI showing 0% for long
            xray_instance.processing_status = 'queued'
            xray_instance.progress = 1
            xray_instance.save(update_fields=['processing_status', 'progress'])

            # Save image to disk
            image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
            
            # Start background processing (prefer Celery only if explicitly enabled)
            use_celery = settings.USE_CELERY
            started = False
            if use_celery:
                try:
                    run_inference_task.delay(xray_instance.pk, model_type)
                    started = True
                except Exception as e:
                    logger.warning("Celery unavailable; falling back to thread: %s", e)
            if not started:
                # Fallback to lightweight thread on the web worker
                thread = threading.Thread(
                    target=process_image_async,
                    args=(image_path, xray_instance, model_type),
                    daemon=True,
                )
                thread.start()
            
            # Check if it's an AJAX request
            if is_ajax(request):
                logger.info(f"Returning success response for upload ID: {xray_instance.pk}")
                # Return JSON response for AJAX requests
                return JsonResponse({
                    'upload_id': xray_instance.pk,
                    'xray_id': xray_instance.pk
                })
            else:
                # Redirect for normal form submissions
                return redirect('xray_results', pk=xray_instance.pk)
        else:
            # Handle form validation errors for AJAX requests
            logger.error(f"Form validation FAILED for user {request.user.username}")
            logger.error(f"Form errors: {form.errors}")
            logger.error(f"Form errors as JSON: {form.errors.as_json()}")
            
            if is_ajax(request):
                return JsonResponse({
                    'error': 'Form validation failed',
                    'errors': form.errors
                }, status=400)
    else:
        form = XRayUploadForm(user=request.user)
    
    # Check if user has upload permission to conditionally display form
    try:
        can_upload = bool(request.user.profile.can_upload_xrays())
    except (AttributeError, ObjectDoesNotExist):
        can_upload = False
        
    return render(request, 'xrayapp/home.html', {
        'form': form,
        'today_date': timezone.now(),
        'user_first_name': request.user.first_name if request.user.is_authenticated else '',
        'user_last_name': request.user.last_name if request.user.is_authenticated else '',
        'can_upload': can_upload,
    })


@login_required
def xray_results(request: HttpRequest, pk: int) -> HttpResponse:
    """View the results of the X-ray analysis"""
    # Get user's hospital from profile
    user_hospital = _get_user_hospital(request.user)
    if user_hospital is None:
        return redirect('home')
    
    # Allow access to any X-ray from the same hospital
    xray_instance = XRayImage.objects.for_hospital(user_hospital).with_user_profile().get(pk=pk)
    
    # Build predictions dictionary based on model fields
    predictions = {}
    
    # Create reverse mapping for display labels
    FIELD_TO_LABEL = {v: k for k, v in RESULT_KEY_TO_FIELD.items()}
    
    for field in PATHOLOGY_FIELDS:
        value = getattr(xray_instance, field)
        if value is not None:
            # Use the original key as label if available, otherwise capitalize field name
            label = FIELD_TO_LABEL.get(field, field.replace('_', ' ').title())
            predictions[label] = value
            
    # Sort predictions by value (highest to lowest)
    predictions = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))
    
    # Pathology explanations for educational purposes (centralized for reuse).
    pathology_explanations = PATHOLOGY_EXPLANATIONS
    
    # Calculate patient age if date_of_birth is provided
    patient_age = None
    if xray_instance.date_of_birth:
        today = timezone.now().date()
        patient_age = relativedelta(today, xray_instance.date_of_birth).years
    
    # Format patient information for display
    patient_info = {
        _('Patient ID'): xray_instance.patient_id,
        _('Name'): f"{xray_instance.first_name} {xray_instance.last_name}".strip() if xray_instance.first_name or xray_instance.last_name else None,
        _('Gender'): xray_instance.gender.capitalize() if xray_instance.gender else None,
        _('Age'): f"{patient_age} {_('years')}" if patient_age is not None else None,
        _('Date of Birth'): xray_instance.date_of_birth.strftime("%Y-%m-%d") if xray_instance.date_of_birth else None,
        _('X-ray Date'): xray_instance.date_of_xray.strftime("%Y-%m-%d") if xray_instance.date_of_xray else None,
        _('Additional Information'): xray_instance.additional_info if xray_instance.additional_info else None,
        _('Technologist'): f"{xray_instance.technologist_first_name} {xray_instance.technologist_last_name}".strip() if xray_instance.technologist_first_name or xray_instance.technologist_last_name else None,
    }
    
    # Create image metadata dictionary
    image_metadata = {
        _('Image name'): Path(xray_instance.image.name).name,
        _('Format'): xray_instance.image_format,
        _('Size'): xray_instance.image_size,
        _('Resolution'): xray_instance.image_resolution,
        _('Date Created'): xray_instance.image_date_created.strftime("%Y-%m-%d %H:%M") if xray_instance.image_date_created else _("Unknown"),
    }
    
    # Filter out None values
    patient_info = {k: v for k, v in patient_info.items() if v is not None}
    
    # Get image URL
    image_url = xray_instance.image.url
    image_is_dicom = (
        Path(xray_instance.image.name).suffix.lower() in {".dcm", ".dicom"}
        or str(xray_instance.image_format or "").upper() == "DICOM"
    )
    
    # Ensure severity level is calculated and stored
    if xray_instance.severity_level is None:
        xray_instance.severity_level = xray_instance.calculate_severity_level
        xray_instance.save()
    
    # Get all visualizations for this X-ray
    visualizations = VisualizationResult.objects.filter(xray=xray_instance).order_by('-created_at')
    
    # Group visualizations by type for display
    gradcam_visualizations = []
    pli_visualizations = []
    segmentation_visualizations = []
    
    for viz in visualizations:
        viz_data = _serialize_visualization(viz)
        
        if viz.visualization_type in ['gradcam', 'combined_gradcam']:
            gradcam_visualizations.append(viz_data)
        elif viz.visualization_type in ['pli', 'combined_pli']:
            pli_visualizations.append(viz_data)
        elif viz.visualization_type in ['segmentation', 'segmentation_combined']:
            segmentation_visualizations.append(viz_data)
    
    # Prepare legacy GRAD-CAM URLs for backward compatibility
    media_url = settings.MEDIA_URL
    gradcam_url = f"{media_url}{xray_instance.gradcam_visualization}" if xray_instance.has_gradcam and xray_instance.gradcam_visualization else None
    heatmap_url = f"{media_url}{xray_instance.gradcam_heatmap}" if xray_instance.has_gradcam and xray_instance.gradcam_heatmap else None
    gradcam_overlay_url = f"{media_url}{xray_instance.gradcam_overlay}" if xray_instance.has_gradcam and xray_instance.gradcam_overlay else None

    # Prepare legacy PLI URLs for backward compatibility
    pli_url = f"{media_url}{xray_instance.pli_visualization}" if xray_instance.has_pli and xray_instance.pli_visualization else None
    pli_saliency_url = f"{media_url}{xray_instance.pli_saliency_map}" if xray_instance.has_pli and xray_instance.pli_saliency_map else None
    pli_overlay_url = f"{media_url}{xray_instance.pli_overlay_visualization}" if xray_instance.has_pli and xray_instance.pli_overlay_visualization else None

    context = {
        'xray': xray_instance,
        'image_url': image_url,
        'image_is_dicom': image_is_dicom,
        'predictions': predictions,
        'patient_info': patient_info,
        'image_metadata': image_metadata,
        'severity_level': xray_instance.severity_level,
        'severity_label': xray_instance.severity_label,
        # Multiple visualization support
        'gradcam_visualizations': gradcam_visualizations,
        'pli_visualizations': pli_visualizations,
        'segmentation_visualizations': segmentation_visualizations,
        'has_multiple_visualizations': len(visualizations) > 0,
        # Legacy support for single visualizations
        'has_gradcam': xray_instance.has_gradcam,
        'gradcam_url': gradcam_url,
        'heatmap_url': heatmap_url,
        'gradcam_overlay_url': gradcam_overlay_url,
        'gradcam_target': xray_instance.gradcam_target_class,
        'has_pli': xray_instance.has_pli,
        'pli_url': pli_url,
        'pli_saliency_url': pli_saliency_url,
        'pli_overlay_url': pli_overlay_url,
        'pli_target': xray_instance.pli_target_class,
        'pathology_explanations': pathology_explanations,
    }
    
    return render(request, 'xrayapp/results.html', context)


def check_progress(request: HttpRequest, pk: int) -> JsonResponse:
    """AJAX endpoint to check processing progress - lightweight version for memory-constrained systems"""
    
    # Check authentication manually to provide better JSON error responses
    if not request.user.is_authenticated:
        return JsonResponse({
            'error': 'Authentication required',
            'message': 'Please log in to check progress'
        }, status=401)
    
    try:
        # Allow access to any X-ray from the same hospital (matches `xray_results`
        # and enables radiologists to monitor progress for shared records).
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return JsonResponse(
                {
                    'error': 'Permission denied',
                    'message': 'User profile is missing a hospital assignment.',
                },
                status=403,
            )

        xray_instance = XRayImage.objects.for_hospital(user_hospital).get(pk=pk)
        
        response_data = {
            'status': xray_instance.processing_status,
            'progress': xray_instance.progress,
            'xray_id': xray_instance.pk,
            'requires_expert_review': xray_instance.requires_expert_review
        }
        
        # If processing is complete, include visualization data
        if xray_instance.progress >= 100 and xray_instance.processing_status == 'completed':
            media_url = settings.MEDIA_URL
            
            # Get all visualizations for this X-ray
            visualizations = VisualizationResult.objects.filter(xray=xray_instance).order_by('-created_at')
            
            # Group visualizations by type
            gradcam_visualizations = []
            pli_visualizations = []
            segmentation_visualizations = []
            
            for viz in visualizations:
                viz_data = _serialize_visualization(viz)
                # Convert datetime to string for JSON serialization
                if hasattr(viz_data['created_at'], 'isoformat'):
                    viz_data['created_at'] = viz_data['created_at'].isoformat()
                
                if viz.visualization_type in ['gradcam', 'combined_gradcam']:
                    gradcam_visualizations.append(viz_data)
                elif viz.visualization_type in ['pli', 'combined_pli']:
                    pli_visualizations.append(viz_data)
                elif viz.visualization_type in ['segmentation', 'segmentation_combined']:
                    segmentation_visualizations.append(viz_data)
            
            # Include visualization data in response
            if gradcam_visualizations:
                response_data['gradcam'] = {
                    'has_gradcam': True,
                    'visualizations': gradcam_visualizations
                }
            
            if pli_visualizations:
                response_data['pli'] = {
                    'has_pli': True,
                    'visualizations': pli_visualizations
                }
            
            if segmentation_visualizations:
                response_data['segmentation'] = {
                    'has_segmentation': True,
                    'visualizations': segmentation_visualizations
                }
            
            # Include backward compatibility data for latest visualizations
            if xray_instance.has_gradcam:
                response_data['gradcam_legacy'] = {
                    'gradcam_url': f"{media_url}{xray_instance.gradcam_visualization}" if xray_instance.gradcam_visualization else None,
                    'heatmap_url': f"{media_url}{xray_instance.gradcam_heatmap}" if xray_instance.gradcam_heatmap else None,
                    'gradcam_overlay_url': f"{media_url}{xray_instance.gradcam_overlay}" if xray_instance.gradcam_overlay else None,
                    'gradcam_target': xray_instance.gradcam_target_class
                }
            
            if xray_instance.has_pli:
                response_data['pli_legacy'] = {
                    'pli_url': f"{media_url}{xray_instance.pli_visualization}" if xray_instance.pli_visualization else None,
                    'pli_saliency_url': f"{media_url}{xray_instance.pli_saliency_map}" if xray_instance.pli_saliency_map else None,
                    'pli_overlay_url': f"{media_url}{xray_instance.pli_overlay_visualization}" if xray_instance.pli_overlay_visualization else None,
                    'pli_target': xray_instance.pli_target_class
                }
            
            # Include image URL for display
            response_data['image_url'] = xray_instance.image.url
        
        # Force garbage collection to free memory
        gc.collect()
        return JsonResponse(response_data)
    except XRayImage.DoesNotExist:
        # Log the 404 for debugging
        logger.warning("XRayImage %s not found/denied for user=%s", pk, request.user.username)
        
        return JsonResponse({
            'error': 'Image not found or access denied',
            'message': f'X-ray image {pk} was not found or you do not have permission to access it.'
        }, status=404)
    except Exception as e:
        # Log server-side details; don't leak internal exceptions to clients.
        logger.exception("Error in check_progress for pk=%s", pk)
        
        # Return JSON error response instead of HTML error page
        payload: dict[str, Any] = {
            'error': _('An error occurred while checking progress'),
        }
        if settings.DEBUG:
            payload['details'] = str(e)
        return JsonResponse(payload, status=500)
