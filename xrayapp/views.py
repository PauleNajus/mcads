"""Views for MCADS.

This module contains various view functions. Some dynamic attributes on Django
models (like `.id`) are accessed via `.pk` to satisfy static type checkers.
User profile access is guarded because `request.user` may not have a related
`profile` yet at the time of access.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Case, F, FloatField, IntegerField, Value, When
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone, translation
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from .forms import (
    ChangePasswordForm,
    PredictionHistoryFilterForm,
    UserInfoForm,
    UserProfileForm,
    XRayUploadForm,
)
from .interpretability import (
    apply_combined_gradcam,
    apply_combined_pixel_interpretability,
    apply_gradcam,
    apply_pixel_interpretability,
)
from .models import PredictionHistory, SavedRecord, UserProfile, VisualizationResult, XRayImage
from .pathology import PATHOLOGY_EXPLANATIONS
from .tasks import run_inference_task, run_interpretability_task, run_segmentation_task
from .utils import (
    process_image,
    save_heatmap,
    save_interpretability_visualization,
    save_overlay,
    save_overlay_visualization,
    save_saliency_map,
)

logger = logging.getLogger(__name__)


def _get_user_hospital(user: Any) -> str | None:
    """Return the user's hospital (from `UserProfile`) if available."""

    profile = getattr(user, "profile", None)
    return getattr(profile, "hospital", None)


def _apply_history_filters(
    query: Any,
    cleaned_data: dict[str, Any],
    *,
    xray_prefix: str,
    pathology_prefix: str,
) -> Any:
    """Apply shared filter form logic to a queryset.

    This keeps `prediction_history` and `saved_records` filtering consistent and
    avoids duplicated, slightly divergent logic.
    """

    # Gender filter (xray fields)
    if (gender := cleaned_data.get("gender")):
        query = query.filter(**{f"{xray_prefix}gender": gender})

    # Age range filters (xray date_of_birth)
    now_date = timezone.now().date()
    if (age_min := cleaned_data.get("age_min")) is not None:
        min_age_date = now_date - relativedelta(years=age_min)
        query = query.filter(**{f"{xray_prefix}date_of_birth__lte": min_age_date})

    if (age_max := cleaned_data.get("age_max")) is not None:
        max_age_date = now_date - relativedelta(years=age_max + 1)
        query = query.filter(**{f"{xray_prefix}date_of_birth__gte": max_age_date})

    # Date range filters (xray date_of_xray)
    if (date_min := cleaned_data.get("date_min")):
        query = query.filter(**{f"{xray_prefix}date_of_xray__gte": date_min})

    if (date_max := cleaned_data.get("date_max")):
        query = query.filter(**{f"{xray_prefix}date_of_xray__lte": date_max})

    # Pathology threshold filter (history fields)
    if (pathology := cleaned_data.get("pathology")) and (threshold := cleaned_data.get("pathology_threshold")) is not None:
        query = query.filter(**{f"{pathology_prefix}{pathology}__gte": threshold})

    return query

def health(request: HttpRequest) -> HttpResponse:
    """Health check endpoint (used by Docker / reverse proxies).

    Keep this intentionally lightweight: return 200 if the Django process is up.
    """
    return HttpResponse("ok", content_type="text/plain")


def process_image_async(image_path: Path, xray_instance: XRayImage, model_type: str) -> None:
    """Process the image in a background thread and persist predictions.

    Note: For production workloads prefer Celery. This thread fallback exists
    for deployments where Celery is disabled/unavailable.
    """
    try:
        logger.info(f"Starting image processing for {image_path} with model {model_type}")
        
        # Simple processing without signal timeout (signals don't work well with threads)
        results = process_image(image_path, xray_instance, model_type)
        logger.info(f"Image processing completed successfully")
        
        # Persist predictions and derived fields in one place.
        xray_instance.apply_predictions_from_results(results)
        xray_instance.severity_level = xray_instance.calculate_severity_level
        xray_instance.save()

        # Create prediction history record (snapshot for audit/history page).
        PredictionHistory.create_from_xray(xray_instance, model_type)

        # Mark the record as fully completed *after* persisting predictions.
        xray_instance.progress = 100
        xray_instance.processing_status = 'completed'
        xray_instance.save(update_fields=['progress', 'processing_status'])
        
        logger.info(f"Successfully processed and saved results for {image_path}")
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        xray_instance.processing_status = 'error'
        xray_instance.save(update_fields=['processing_status'])


def process_with_interpretability_async(
    image_path: Path,
    xray_instance: XRayImage,
    model_type: str,
    interpretation_method: str,
    target_class: str | None = None,
) -> None:
    """Process the image with interpretability visualization in a background thread"""
    try:
        # Set initial progress
        xray_instance.progress = 10
        xray_instance.save()
        
        logger.info(f"Starting {interpretation_method} visualization for image {image_path} with model {model_type}")
        
        # Process the image with the selected interpretability method
        if interpretation_method == 'gradcam':
            try:
                results = apply_gradcam(image_path, model_type, target_class)
                results['method'] = 'gradcam'
                logger.info(f"GradCAM generation completed successfully for {target_class}")
            except Exception as e:
                logger.error(f"Error in GradCAM generation: {str(e)}")
                raise
        elif interpretation_method == 'pli':
            try:
                results = apply_pixel_interpretability(image_path, model_type, target_class)
                results['method'] = 'pli'
                logger.info(f"PLI generation completed successfully for {target_class}")
            except Exception as e:
                logger.error(f"Error in PLI generation: {str(e)}")
                raise
        elif interpretation_method == 'combined_gradcam':
            try:
                results = apply_combined_gradcam(image_path, model_type)
                logger.info(f"Combined GradCAM generation completed successfully for {len(results['selected_pathologies'])} pathologies")
            except Exception as e:
                logger.error(f"Error in Combined GradCAM generation: {str(e)}")
                raise
        elif interpretation_method == 'combined_pli':
            try:
                results = apply_combined_pixel_interpretability(image_path, model_type)
                logger.info(f"Combined PLI generation completed successfully for {len(results['selected_pathologies'])} pathologies")
            except Exception as e:
                logger.error(f"Error in Combined PLI generation: {str(e)}")
                raise
        else:
            # Invalid method, return error
            logger.error(f"Invalid interpretation method: {interpretation_method}")
            xray_instance.processing_status = 'error'
            xray_instance.save()
            return
        
        # Update progress
        xray_instance.progress = 70
        xray_instance.save()
        
        logger.info(f"Saving visualization results for {interpretation_method}")
        
        # Save interpretability visualizations
        if interpretation_method and 'method' in results:
            if results['method'] == 'gradcam':
                # Create output directory if it doesn't exist
                output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'gradcam'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filenames
                combined_filename = f"gradcam_{xray_instance.id}_{results['target_class']}.png"
                heatmap_filename = f"heatmap_{xray_instance.id}_{results['target_class']}.png"
                overlay_filename = f"gradcam_overlay_{xray_instance.id}_{results['target_class']}.png"
                
                # Generate paths
                combined_path = output_dir / combined_filename
                heatmap_path = output_dir / heatmap_filename
                overlay_path = output_dir / overlay_filename
                
                logger.info(f"Saving Grad-CAM combined visualization to {combined_path}")
                
                # Save combined visualization
                save_interpretability_visualization(results, combined_path)
                
                logger.info(f"Saving heatmap to {heatmap_path}")
                
                # Save heatmap separately
                save_heatmap(results, heatmap_path)
                
                logger.info(f"Saving overlay to {overlay_path}")
                
                # Save overlay separately
                save_overlay(results, overlay_path)
                
                # Create VisualizationResult record
                visualization_data = {
                    'combined_filename': f"interpretability/gradcam/{combined_filename}",
                    'heatmap_filename': f"interpretability/gradcam/{heatmap_filename}",
                    'overlay_filename': f"interpretability/gradcam/{overlay_filename}"
                }
                create_visualization_result(xray_instance, 'gradcam', results['target_class'], visualization_data, model_type)
                
                # Update XRayImage flags for backward compatibility
                xray_instance.has_gradcam = True
                xray_instance.gradcam_visualization = f"interpretability/gradcam/{combined_filename}"
                xray_instance.gradcam_heatmap = f"interpretability/gradcam/{heatmap_filename}"
                xray_instance.gradcam_overlay = f"interpretability/gradcam/{overlay_filename}"
                xray_instance.gradcam_target_class = results['target_class']
                
            elif results['method'] == 'pli':
                try:
                    # Create output directory if it doesn't exist
                    output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'pli'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate filename for saliency map
                    saliency_filename = f"pli_{xray_instance.id}_{results['target_class']}.png"
                    saliency_path = output_dir / saliency_filename
                    
                    # Generate filename for overlay
                    overlay_filename = f"pli_overlay_{xray_instance.id}_{results['target_class']}.png"
                    overlay_path = output_dir / overlay_filename
                    
                    # Generate filename for separate saliency map
                    separate_saliency_filename = f"pli_saliency_{xray_instance.id}_{results['target_class']}.png"
                    separate_saliency_path = output_dir / separate_saliency_filename
                    
                    logger.info(f"Saving PLI visualization to {saliency_path}")
                    
                    # Save saliency visualization (combined visualization)
                    save_interpretability_visualization(results, saliency_path)
                    
                    logger.info(f"Saving PLI overlay to {overlay_path}")
                    
                    # Save overlay separately
                    save_overlay_visualization(results, overlay_path)
                    
                    logger.info(f"Saving PLI saliency map to {separate_saliency_path}")
                    
                    # Save saliency map separately
                    save_saliency_map(results, separate_saliency_path)
                    
                    # Create VisualizationResult record
                    visualization_data = {
                        'saliency_filename': f"interpretability/pli/{saliency_filename}",
                        'overlay_filename': f"interpretability/pli/{overlay_filename}",
                        'separate_saliency_filename': f"interpretability/pli/{separate_saliency_filename}"
                    }
                    create_visualization_result(xray_instance, 'pli', results['target_class'], visualization_data, model_type)
                    
                    # Update XRayImage flags for backward compatibility
                    xray_instance.has_pli = True
                    xray_instance.pli_visualization = f"interpretability/pli/{saliency_filename}"
                    xray_instance.pli_overlay_visualization = f"interpretability/pli/{overlay_filename}"
                    xray_instance.pli_saliency_map = f"interpretability/pli/{separate_saliency_filename}"
                    xray_instance.pli_target_class = results['target_class']
                except Exception as e:
                    logger.error(f"Error saving PLI results: {str(e)}")
                    raise
                
            elif results['method'] == 'combined_gradcam':
                # Create output directory if it doesn't exist
                output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'combined_gradcam'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filenames
                combined_filename = f"combined_gradcam_{xray_instance.id}_threshold_{results['threshold']}.png"
                heatmap_filename = f"combined_heatmap_{xray_instance.id}_threshold_{results['threshold']}.png"
                overlay_filename = f"combined_gradcam_overlay_{xray_instance.id}_threshold_{results['threshold']}.png"
                
                # Generate paths
                combined_path = output_dir / combined_filename
                heatmap_path = output_dir / heatmap_filename
                overlay_path = output_dir / overlay_filename
                
                logger.info(f"Saving Combined Grad-CAM visualization to {combined_path}")
                
                # Save combined visualization
                save_interpretability_visualization(results, combined_path)
                
                logger.info(f"Saving combined heatmap to {heatmap_path}")
                
                # Save heatmap separately
                save_heatmap(results, heatmap_path)
                
                logger.info(f"Saving combined overlay to {overlay_path}")
                
                # Save overlay separately
                save_overlay(results, overlay_path)
                
                # Create VisualizationResult record
                visualization_data = {
                    'combined_filename': f"interpretability/combined_gradcam/{combined_filename}",
                    'heatmap_filename': f"interpretability/combined_gradcam/{heatmap_filename}",
                    'overlay_filename': f"interpretability/combined_gradcam/{overlay_filename}",
                    'threshold': results.get('threshold')
                }
                create_visualization_result(xray_instance, 'combined_gradcam', results['pathology_summary'], visualization_data, model_type)
                
                # Update XRayImage flags for backward compatibility
                xray_instance.has_gradcam = True  # Use the same field as regular gradcam
                xray_instance.gradcam_visualization = f"interpretability/combined_gradcam/{combined_filename}"
                xray_instance.gradcam_heatmap = f"interpretability/combined_gradcam/{heatmap_filename}"
                xray_instance.gradcam_overlay = f"interpretability/combined_gradcam/{overlay_filename}"
                xray_instance.gradcam_target_class = results['pathology_summary']  # Store summary of selected pathologies
            
            elif results['method'] == 'combined_pli':
                # Create output directory if it doesn't exist
                output_dir = Path(settings.MEDIA_ROOT) / 'interpretability' / 'combined_pli'
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate filenames
                combined_filename = f"combined_pli_{xray_instance.id}_threshold_{results['threshold']}.png"
                saliency_filename = f"combined_pli_saliency_{xray_instance.id}_threshold_{results['threshold']}.png"
                overlay_filename = f"combined_pli_overlay_{xray_instance.id}_threshold_{results['threshold']}.png"
                
                # Generate paths
                combined_path = output_dir / combined_filename
                saliency_path = output_dir / saliency_filename
                overlay_path = output_dir / overlay_filename
                
                logger.info(f"Saving Combined PLI visualization to {combined_path}")
                
                # Save combined visualization
                save_interpretability_visualization(results, combined_path)
                
                logger.info(f"Saving Combined PLI saliency map to {saliency_path}")
                
                # Save saliency map separately
                save_saliency_map(results, saliency_path)
                
                logger.info(f"Saving Combined PLI overlay to {overlay_path}")
                
                # Save overlay separately
                save_overlay_visualization(results, overlay_path)
                
                # Create VisualizationResult record
                visualization_data = {
                    'saliency_filename': f"interpretability/combined_pli/{combined_filename}",
                    'separate_saliency_filename': f"interpretability/combined_pli/{saliency_filename}",
                    'overlay_filename': f"interpretability/combined_pli/{overlay_filename}",
                    'threshold': results.get('threshold')
                }
                create_visualization_result(xray_instance, 'combined_pli', results['pathology_summary'], visualization_data, model_type)
                
                # Update XRayImage flags for backward compatibility
                xray_instance.has_pli = True  # Use the same field as regular PLI
                xray_instance.pli_visualization = f"interpretability/combined_pli/{combined_filename}"
                xray_instance.pli_saliency_map = f"interpretability/combined_pli/{saliency_filename}"
                xray_instance.pli_overlay_visualization = f"interpretability/combined_pli/{overlay_filename}"
                xray_instance.pli_target_class = results['pathology_summary']  # Store summary of selected pathologies
        
        xray_instance.progress = 90
        xray_instance.processing_status = 'completed'
        xray_instance.save()
        
        # Update existing prediction history record instead of creating a new one
        # This ensures visualizations are saved to the same record in "History Records"
        update_existing_prediction_history(xray_instance, model_type)
        
        # Set progress to 100%
        xray_instance.progress = 100
        xray_instance.save()
        
        logger.info(f"Interpretability visualization complete for {interpretation_method}")
        
    except Exception as e:
        import traceback
        logger.error(f"Error in interpretability processing: {str(e)}")
        logger.error(traceback.format_exc())
        xray_instance.processing_status = 'error'
        xray_instance.save()


def create_prediction_history(xray_instance: XRayImage, model_type: str) -> None:
    """Create a prediction history record for an XRayImage.

    Deprecated wrapper kept for backwards compatibility inside this module.
    """
    PredictionHistory.create_from_xray(xray_instance, model_type)


def create_visualization_result(
    xray_instance: XRayImage,
    visualization_type: str,
    target_pathology: str,
    results: dict[str, Any],
    model_type: str,
) -> VisualizationResult:
    """Create or update a VisualizationResult record for a specific pathology"""
    try:
        # Try to get existing visualization of same type for same pathology
        # This will either update existing or create new (preventing duplicates due to unique constraint)
        visualization, created = VisualizationResult.objects.get_or_create(
            xray=xray_instance,
            visualization_type=visualization_type,
            target_pathology=target_pathology,
            defaults={
                'model_used': model_type,
            }
        )
        
        # Update the visualization paths based on type
        if visualization_type in ['gradcam', 'combined_gradcam']:
            # For GRAD-CAM visualizations
            if 'combined_filename' in results:
                visualization.visualization_path = results['combined_filename']
            if 'heatmap_filename' in results:
                visualization.heatmap_path = results['heatmap_filename']
            if 'overlay_filename' in results:
                visualization.overlay_path = results['overlay_filename']
                
        elif visualization_type in ['pli', 'combined_pli']:
            # For PLI visualizations
            if 'saliency_filename' in results:
                visualization.visualization_path = results['saliency_filename']
            if 'separate_saliency_filename' in results:
                visualization.saliency_path = results['separate_saliency_filename']
            if 'overlay_filename' in results:
                visualization.overlay_path = results['overlay_filename']
            if 'threshold' in results:
                visualization.threshold = results['threshold']
        
        # Update model used if it's different
        if visualization.model_used != model_type:
            visualization.model_used = f"{visualization.model_used}+{model_type}"
        
        visualization.save()
        
        action = "Created" if created else "Updated"
        logger.info(f"{action} visualization result: {visualization_type} - {target_pathology} for X-ray #{xray_instance.pk}")
        
        return visualization
        
    except Exception as e:
        logger.error(f"Error creating visualization result: {str(e)}")
        raise


def update_existing_prediction_history(xray_instance: XRayImage, model_type: str) -> None:
    """Update existing prediction history record instead of creating a new one.

    Deprecated wrapper kept for backwards compatibility inside this module.
    """
    PredictionHistory.update_latest_for_xray(xray_instance, model_type)


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
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': _('You do not have permission to upload X-ray images. Please contact your administrator.')
                    }, status=403)
                else:
                    messages.error(request, _('You do not have permission to upload X-ray images.'))
                    return redirect('home')
        except AttributeError:
            # User doesn't have a profile - create one with default role
            logger.warning(f"User {request.user.username} doesn't have a profile. Creating one.")
            UserProfile.objects.create(user=request.user, role='Radiographer')
            # Continue with upload since default role allows it
        
        form = XRayUploadForm(request.POST, request.FILES, user=request.user)
        
        # Debug logging for troubleshooting
        logger.info(f"Upload attempt by user: {request.user.username} (ID: {request.user.id})")
        logger.info(f"Form fields received: {list(request.POST.keys())}")
        logger.info(f"Files received: {list(request.FILES.keys())}")
        
        if form.is_valid():
            logger.info(f"Form validation PASSED for user {request.user.username}")
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

            # Preserve source format for DICOM uploads (stored as PNG for processing).
            source_format = getattr(form, "_mcads_source_format", None)
            if source_format:
                xray_instance.image_format = str(source_format)
            # Now save the instance
            try:
                xray_instance.save()
            except Exception as e:
                # Log the error for debugging
                logger.error(f"Error saving XRayImage: {e}")
                
                # Return error response for AJAX requests
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'error': 'Database error occurred while saving image',
                        'details': str(e)
                    }, status=500)
                else:
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
                    logger.warning(f"Celery unavailable; falling back to thread: {e}")
            if not started:
                # Fallback to lightweight thread on the web worker
                thread = threading.Thread(
                    target=process_image_async,
                    args=(image_path, xray_instance, model_type),
                    daemon=True,
                )
                thread.start()
            
            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
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
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'error': 'Form validation failed',
                    'errors': form.errors
                }, status=400)
    else:
        form = XRayUploadForm(user=request.user)
    
    # Check if user has upload permission to conditionally display form
    can_upload = request.user.profile.can_upload_xrays() if hasattr(request.user, 'profile') else False
        
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
    predictions = {
        'Atelectasis': xray_instance.atelectasis,
        'Cardiomegaly': xray_instance.cardiomegaly,
        'Consolidation': xray_instance.consolidation,
        'Edema': xray_instance.edema,
        'Effusion': xray_instance.effusion,
        'Emphysema': xray_instance.emphysema,
        'Fibrosis': xray_instance.fibrosis,
        'Hernia': xray_instance.hernia,
        'Infiltration': xray_instance.infiltration,
        'Mass': xray_instance.mass,
        'Nodule': xray_instance.nodule,
        'Pleural Thickening': xray_instance.pleural_thickening,
        'Pneumonia': xray_instance.pneumonia,
        'Pneumothorax': xray_instance.pneumothorax,
        'Fracture': xray_instance.fracture,
        'Lung Opacity': xray_instance.lung_opacity,
    }
    
    # Add DenseNet exclusive fields if they have values
    if xray_instance.enlarged_cardiomediastinum is not None:
        predictions['Enlarged Cardiomediastinum'] = xray_instance.enlarged_cardiomediastinum
    if xray_instance.lung_lesion is not None:
        predictions['Lung Lesion'] = xray_instance.lung_lesion
    
    # Filter out predictions with None values
    predictions = {k: v for k, v in predictions.items() if v is not None}
    
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
        viz_data = {
            'id': viz.pk,
            'target_pathology': viz.target_pathology,
            'created_at': viz.created_at,
            'model_used': viz.model_used,
            'visualization_url': viz.visualization_url,
            'heatmap_url': viz.heatmap_url,
            'overlay_url': viz.overlay_url,
            'saliency_url': viz.saliency_url,
            'threshold': viz.threshold,
            'visualization_type': viz.visualization_type,
            'confidence_score': viz.confidence_score,
            'metadata': viz.metadata,
            'visualization_path': viz.visualization_path
        }
        
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

    context = {
        'xray': xray_instance,
        'image_url': image_url,
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
        'pathology_explanations': pathology_explanations,
    }
    
    return render(request, 'xrayapp/results.html', context)


@login_required
def prediction_history(request: HttpRequest) -> HttpResponse:
    """View prediction history with advanced filtering and pagination"""
    form = PredictionHistoryFilterForm(request.GET)
    
    # Get user's hospital from profile
    user_hospital = _get_user_hospital(request.user)
    if user_hospital is None:
        return redirect('home')
    
    # Filter by users from the same hospital and pre-join common relations to
    # avoid N+1 queries in templates/admin.
    query = PredictionHistory.objects.for_hospital(user_hospital).with_related()
    
    # Apply filters if the form is valid
    if form.is_valid():
        query = _apply_history_filters(
            query,
            form.cleaned_data,
            xray_prefix="xray__",
            pathology_prefix="",
        )
        
        # Apply sorting
        sort_by = form.cleaned_data.get('sort_by', '')
        sort_order = form.cleaned_data.get('sort_order', 'desc')
        
        if sort_by == 'severity':
            # Sort by severity_level (1=Insignificant, 2=Moderate, 3=Significant)
            # For records without severity_level, calculate it dynamically
            from django.db.models.functions import Coalesce
            
            # List of all pathology fields
            pathology_fields = [
                'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion',
                'emphysema', 'fibrosis', 'hernia', 'infiltration', 'mass', 'nodule',
                'pleural_thickening', 'pneumonia', 'pneumothorax', 'fracture',
                'lung_opacity', 'enlarged_cardiomediastinum', 'lung_lesion'
            ]
            
            # Calculate average of all pathology fields for records without severity_level
            sum_expr = sum([Coalesce(F(field), Value(0.0), output_field=FloatField()) 
                           for field in pathology_fields], Value(0.0, output_field=FloatField()))
            avg_expr = sum_expr / Value(len(pathology_fields), output_field=FloatField())
            
            # First, annotate with average pathology
            query = query.annotate(avg_pathology=avg_expr)
            
            # Then, use existing severity_level if available, otherwise calculate it
            query = query.annotate(
                sort_severity=Case(
                    # If severity_level exists, use it
                    When(severity_level__isnull=False, then=F('severity_level')),
                    # Otherwise calculate from average pathology probability
                    When(avg_pathology__lte=0.19, then=Value(1)),
                    When(avg_pathology__lte=0.30, then=Value(2)),
                    default=Value(3),
                    output_field=IntegerField()
                )
            )
            
            # Sort by severity
            # desc: Significant → Moderate → Insignificant (3, 2, 1)
            # asc: Insignificant → Moderate → Significant (1, 2, 3)
            order_field = 'sort_severity' if sort_order == 'asc' else '-sort_severity'
            query = query.order_by(order_field, '-created_at')
        elif sort_by == 'xray_date':
            # Sort by X-ray date
            order_field = 'xray__date_of_xray' if sort_order == 'asc' else '-xray__date_of_xray'
            query = query.order_by(order_field, '-created_at')
        else:
            # Default sorting by prediction date (created_at)
            order_field = 'created_at' if sort_order == 'asc' else '-created_at'
            query = query.order_by(order_field)
    else:
        # If form is not valid, apply default sorting
        query = query.order_by('-created_at')
    
    # Add pagination for better performance
    records_per_page = 25  # Default value
    if form.is_valid() and form.cleaned_data.get('records_per_page'):
        records_per_page = int(form.cleaned_data['records_per_page'])
    
    paginator = Paginator(query, records_per_page)
    page_number = request.GET.get('page')
    history_items = paginator.get_page(page_number)
    
    # Get total count for the filtered query
    total_count = paginator.count
    
    # Get saved record IDs for current user to show star status
    saved_record_ids = set(SavedRecord.objects.filter(
        user=request.user,
        prediction_history__in=[item.pk for item in history_items]
    ).values_list('prediction_history_id', flat=True))
    
    context = {
        'form': form,
        'history_items': history_items,
        'total_count': total_count,
        'saved_record_ids': saved_record_ids,
    }
    
    return render(request, 'xrayapp/prediction_history.html', context)


@login_required
def delete_prediction_history(request: HttpRequest, pk: int) -> HttpResponse:
    """Delete a prediction history record"""
    try:
        # Get user's hospital from profile
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return redirect('prediction_history')
        
        # Allow deletion of any record from the same hospital
        history_item = PredictionHistory.objects.for_hospital(user_hospital).get(pk=pk)
        history_item.delete()
        messages.success(request, _('Prediction history record has been deleted.'))
    except PredictionHistory.DoesNotExist:
        messages.error(request, _('Prediction history record not found.'))
    
    return redirect('prediction_history')


@login_required
def delete_all_prediction_history(request: HttpRequest) -> HttpResponse:
    """Delete all prediction history records"""
    if request.method == 'POST':
        # Get user's hospital from profile
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return redirect('prediction_history')
        
        # Count records before deletion for current hospital
        count = PredictionHistory.objects.for_hospital(user_hospital).count()
        
        # Delete all records for current hospital
        PredictionHistory.objects.for_hospital(user_hospital).delete()
        
        if count > 0:
            messages.success(request, _('All %(count)d prediction history records have been deleted.') % {'count': count})
        else:
            messages.info(request, _('No prediction history records to delete.'))
    
    return redirect('prediction_history')


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
        owner_profile = getattr(visualization.xray.user, 'profile', None)
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
        for file_path in file_paths:
            if file_path:
                try:
                    full_path = Path(settings.MEDIA_ROOT) / file_path
                    if full_path.exists():
                        full_path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")
        
        # Delete the visualization record
        visualization.delete()
        
        return JsonResponse({'success': True})
        
    except VisualizationResult.DoesNotExist:
        return JsonResponse({'success': False, 'error': _('Visualization not found')}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def edit_prediction_history(request: HttpRequest, pk: int) -> HttpResponse:
    """Edit a prediction history record"""
    try:
        # Get user's hospital from profile
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return redirect('prediction_history')
        
        # Allow editing of any record from the same hospital
        history_item = PredictionHistory.objects.for_hospital(user_hospital).get(pk=pk)
        
        if request.method == 'POST':
            # Handle form submission
            form = XRayUploadForm(request.POST, instance=history_item.xray)
            if form.is_valid():
                form.save()
                messages.success(request, _('Prediction record has been updated.'))
                return redirect('prediction_history')
        else:
            # Display form with current values
            form = XRayUploadForm(instance=history_item.xray)
        
        return render(request, 'xrayapp/edit_prediction.html', {
            'form': form,
            'history_item': history_item
        })
    except PredictionHistory.DoesNotExist:
        messages.error(request, _('Prediction history record not found.'))
        return redirect('prediction_history')


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
        # For now, segmentation requires Celery
        # Could implement a threaded version later if needed
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.content_type == 'application/json':
            return JsonResponse(
                {
                    'status': 'error',
                    'error': _('Segmentation processing is currently unavailable. Please try again later.'),
                },
                status=503,
            )
        messages.error(request, _('Segmentation processing is currently unavailable. Please try again later.'))
        return redirect('xray_results', pk=pk)
    
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


def check_progress(request: HttpRequest, pk: int) -> JsonResponse:
    """AJAX endpoint to check processing progress - lightweight version for memory-constrained systems"""
    
    # Import here to avoid loading heavy dependencies
    import gc
    from django.http import JsonResponse
    
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
            'xray_id': xray_instance.pk
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
                viz_data = {
                    'id': viz.pk,
                    'target_pathology': viz.target_pathology,
                    'created_at': viz.created_at.isoformat(),
                    'model_used': viz.model_used,
                    'visualization_url': viz.visualization_url,
                    'heatmap_url': viz.heatmap_url,
                    'overlay_url': viz.overlay_url,
                    'saliency_url': viz.saliency_url,
                    'threshold': viz.threshold,
                    'confidence_score': viz.confidence_score,
                    'metadata': viz.metadata,
                    'visualization_path': viz.visualization_path
                }
                
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
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("XRayImage %s not found/denied for user=%s", pk, request.user.username)
        
        return JsonResponse({
            'error': 'Image not found or access denied',
            'message': f'X-ray image {pk} was not found or you do not have permission to access it.'
        }, status=404)
    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in check_progress for pk={pk}: {e}")
        
        # Return JSON error response instead of HTML error page
        return JsonResponse({
            'error': 'An error occurred while checking progress',
            'details': str(e)
        }, status=500)


@login_required
def account_settings(request: HttpRequest) -> HttpResponse:
    """View for managing user account settings"""
    # Ensure user has a profile
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Handle form submissions
    active_tab = request.GET.get('tab', 'profile')
    
    # Process profile info form
    if request.method == 'POST' and 'update_profile' in request.POST:
        user_form = UserInfoForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, _('Your profile information has been updated successfully.'))
            active_tab = 'profile'
        else:
            messages.error(request, _('Please correct the errors below.'))
            active_tab = 'profile'
    else:
        user_form = UserInfoForm(instance=request.user)
    
    # Process settings form
    if request.method == 'POST' and 'update_settings' in request.POST:
        settings_form = UserProfileForm(request.POST, instance=profile)
        if settings_form.is_valid():
            settings_form.save()
            messages.success(request, _('Your preferences have been updated successfully. Please refresh the page to see theme changes.'))
            active_tab = 'settings'
        else:
            messages.error(request, _('Please correct the errors in your preferences.'))
            active_tab = 'settings'
    else:
        settings_form = UserProfileForm(instance=profile)
    
    # Process password change form
    if request.method == 'POST' and 'change_password' in request.POST:
        password_form = ChangePasswordForm(request.user, request.POST)
        if password_form.is_valid():
            user = request.user
            new_password = password_form.cleaned_data['new_password']
            user.set_password(new_password)
            user.save()
            # Update the session to prevent the user from being logged out
            update_session_auth_hash(request, user)
            messages.success(request, _('Your password has been changed successfully.'))
            active_tab = 'security'
        else:
            messages.error(request, _('Please correct the errors in your password form.'))
            active_tab = 'security'
    else:
        password_form = ChangePasswordForm(request.user)
    
    context = {
        'user_form': user_form,
        'settings_form': settings_form,
        'password_form': password_form,
        'active_tab': active_tab,
    }
    
    return render(request, 'xrayapp/account_settings.html', context)


def logout_confirmation(request: HttpRequest) -> HttpResponse:
    """Display a confirmation page before logging out the user."""
    return render(request, 'registration/logout.html')


@require_POST
def set_language(request: HttpRequest) -> HttpResponse:
    """Custom language switching view that integrates with user profile"""
    language = request.POST.get('language')
    
    if language and language in [lang[0] for lang in settings.LANGUAGES]:
        # Activate the language for this session
        translation.activate(language)
        
        # Get the redirect URL before creating the response
        redirect_url = request.META.get('HTTP_REFERER') or '/'
        if not url_has_allowed_host_and_scheme(
            url=redirect_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            redirect_url = '/'
        response = redirect(redirect_url)
        
        # Set language in session using Django's standard session key
        # Django's LocaleMiddleware looks for 'django_language' in the session
        request.session['django_language'] = language
        
        # Also set the language cookie for immediate effect
        response.set_cookie(
            settings.LANGUAGE_COOKIE_NAME,
            language,
            max_age=settings.LANGUAGE_COOKIE_AGE,
        )
        
        # Update user profile if user is authenticated
        if request.user.is_authenticated:
            try:
                profile, created = UserProfile.objects.get_or_create(user=request.user)
                profile.preferred_language = language
                profile.save()
            except Exception as e:
                logger.error(f"Error updating user language preference: {str(e)}")
        
        return response
    
    # If invalid language, redirect to home
    return redirect('/')


@login_required
@require_POST
def toggle_save_record(request: HttpRequest, pk: int) -> JsonResponse:
    """Toggle save/unsave a prediction history record via AJAX"""
    try:
        # Get user's hospital from profile
        user_hospital = _get_user_hospital(request.user)
        if user_hospital is None:
            return JsonResponse({'success': False, 'error': _('Permission denied')}, status=403)
        
        # Get the prediction history record (must be from same hospital)
        prediction_record = PredictionHistory.objects.for_hospital(user_hospital).get(pk=pk)
        
        # Check if record is already saved by this user
        saved_record, created = SavedRecord.objects.get_or_create(
            user=request.user,
            prediction_history=prediction_record
        )
        
        if created:
            # Record was saved
            return JsonResponse({
                'success': True,
                'saved': True,
                'message': _('Record saved successfully')
            })
        else:
            # Record was already saved, so unsave it
            saved_record.delete()
            return JsonResponse({
                'success': True,
                'saved': False,
                'message': _('Record removed from saved')
            })
            
    except PredictionHistory.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': _('Prediction record not found')
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@login_required
def saved_records(request: HttpRequest) -> HttpResponse:
    """View saved prediction history records with advanced filtering"""
    form = PredictionHistoryFilterForm(request.GET)
    
    # Get user's saved records with optimized queries (avoid N+1 in templates).
    saved_records_query = SavedRecord.objects.for_user(request.user).with_related().order_by('-saved_at')
    
    # Apply filters if the form is valid
    if form.is_valid():
        saved_records_query = _apply_history_filters(
            saved_records_query,
            form.cleaned_data,
            xray_prefix="prediction_history__xray__",
            pathology_prefix="prediction_history__",
        )
    
    # Add pagination for better performance
    records_per_page = 25  # Default value
    if form.is_valid() and form.cleaned_data.get('records_per_page'):
        records_per_page = int(form.cleaned_data['records_per_page'])
    
    paginator = Paginator(saved_records_query, records_per_page)
    page_number = request.GET.get('page')
    saved_records_page = paginator.get_page(page_number)
    
    # Get total count
    total_count = paginator.count
    
    context = {
        'form': form,
        'saved_records': saved_records_page,
        'total_count': total_count,
    }
    
    return render(request, 'xrayapp/saved_records.html', context)


# Error handler views
def handler400(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """400 Bad Request handler."""
    return render(request, 'errors/400.html', status=400)

def handler401(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """401 Unauthorized handler."""
    return render(request, 'errors/401.html', status=401)

def handler403(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """403 Forbidden handler."""
    return render(request, 'errors/403.html', status=403)

def handler404(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """404 Not Found handler."""
    return render(request, 'errors/404.html', status=404)

def handler408(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """408 Request Timeout handler."""
    return render(request, 'errors/408.html', status=408)

def handler429(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """429 Too Many Requests handler."""
    return render(request, 'errors/429.html', status=429)

def handler500(request: HttpRequest) -> HttpResponse:
    """500 Internal Server Error handler."""
    return render(request, 'errors/500.html', status=500)

def handler502(request: HttpRequest) -> HttpResponse:
    """502 Bad Gateway handler."""
    return render(request, 'errors/502.html', status=502)

def handler503(request: HttpRequest) -> HttpResponse:
    """503 Service Unavailable handler."""
    return render(request, 'errors/503.html', status=503)

def handler504(request: HttpRequest) -> HttpResponse:
    """504 Gateway Timeout handler."""
    return render(request, 'errors/504.html', status=504)

def terms_of_service(request: HttpRequest) -> HttpResponse:
    return render(request, 'xrayapp/terms_of_service.html')
