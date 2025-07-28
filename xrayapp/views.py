from django.shortcuts import render, redirect
import threading
import os
from pathlib import Path
from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone, translation
from django.db.models import Q, Prefetch
from django.core.paginator import Paginator
from django.views.decorators.cache import cache_page
from django.core.cache import cache
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from .forms import XRayUploadForm, PredictionHistoryFilterForm, UserInfoForm, UserProfileForm, ChangePasswordForm
from .models import XRayImage, PredictionHistory, UserProfile, VisualizationResult, SavedRecord
from .utils import (process_image, process_image_with_interpretability,
                   save_interpretability_visualization, save_overlay_visualization, save_saliency_map,
                   save_heatmap, save_overlay)
from .interpretability import apply_gradcam, apply_pixel_interpretability, apply_combined_gradcam, apply_combined_pixel_interpretability
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.utils.translation import activate, gettext_lazy as _
import logging

# Set up logging
logger = logging.getLogger(__name__)


def process_image_async(image_path, xray_instance, model_type):
    """Process the image in a background thread and update the model with progress"""
    try:
        logger.info(f"Starting image processing for {image_path} with model {model_type}")
        
        # Simple processing without signal timeout (signals don't work well with threads)
        results = process_image(image_path, xray_instance, model_type)
        logger.info(f"Image processing completed successfully")
        
        # Save predictions to the database - only save what's available in the results
        xray_instance.atelectasis = results.get('Atelectasis', None)
        xray_instance.cardiomegaly = results.get('Cardiomegaly', None)
        xray_instance.consolidation = results.get('Consolidation', None)
        xray_instance.edema = results.get('Edema', None)
        xray_instance.effusion = results.get('Effusion', None)
        xray_instance.emphysema = results.get('Emphysema', None)
        xray_instance.fibrosis = results.get('Fibrosis', None)
        xray_instance.hernia = results.get('Hernia', None)
        xray_instance.infiltration = results.get('Infiltration', None)
        xray_instance.mass = results.get('Mass', None)
        xray_instance.nodule = results.get('Nodule', None)
        xray_instance.pleural_thickening = results.get('Pleural_Thickening', None)
        xray_instance.pneumonia = results.get('Pneumonia', None)
        xray_instance.pneumothorax = results.get('Pneumothorax', None)
        xray_instance.fracture = results.get('Fracture', None)
        xray_instance.lung_opacity = results.get('Lung Opacity', None)
        
        # These fields will only be present in DenseNet results
        if 'Enlarged Cardiomediastinum' in results:
            xray_instance.enlarged_cardiomediastinum = results.get('Enlarged Cardiomediastinum', None)
        if 'Lung Lesion' in results:
            xray_instance.lung_lesion = results.get('Lung Lesion', None)
        
        xray_instance.save()
        
        # Create prediction history record
        create_prediction_history(xray_instance, model_type)
        
        logger.info(f"Successfully processed and saved results for {image_path}")
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        xray_instance.processing_status = 'error'
        xray_instance.save()


def process_with_interpretability_async(image_path, xray_instance, model_type, interpretation_method, target_class=None):
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
        xray_instance.processing_status = 'complete'
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


def create_prediction_history(xray_instance, model_type):
    """Create a prediction history record for an XRayImage"""
    history = PredictionHistory(
        user=xray_instance.user,
        xray=xray_instance,
        model_used=model_type,
        # Copy all pathology values for historical record
        atelectasis=xray_instance.atelectasis,
        cardiomegaly=xray_instance.cardiomegaly,
        consolidation=xray_instance.consolidation,
        edema=xray_instance.edema,
        effusion=xray_instance.effusion,
        emphysema=xray_instance.emphysema,
        fibrosis=xray_instance.fibrosis,
        hernia=xray_instance.hernia,
        infiltration=xray_instance.infiltration,
        mass=xray_instance.mass,
        nodule=xray_instance.nodule,
        pleural_thickening=xray_instance.pleural_thickening,
        pneumonia=xray_instance.pneumonia,
        pneumothorax=xray_instance.pneumothorax,
        fracture=xray_instance.fracture,
        lung_opacity=xray_instance.lung_opacity,
        enlarged_cardiomediastinum=xray_instance.enlarged_cardiomediastinum,
        lung_lesion=xray_instance.lung_lesion,
        # Copy severity level
        severity_level=xray_instance.severity_level,
    )
    # Only save if we have a user assigned
    if xray_instance.user:
        history.save()
    else:
        logger.warning("Warning: XRayImage has no user assigned, skipping prediction history creation")


def create_visualization_result(xray_instance, visualization_type, target_pathology, results, model_type):
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
        logger.info(f"{action} visualization result: {visualization_type} - {target_pathology} for X-ray #{xray_instance.id}")
        
        return visualization
        
    except Exception as e:
        logger.error(f"Error creating visualization result: {str(e)}")
        raise


def update_existing_prediction_history(xray_instance, model_type):
    """Update existing prediction history record instead of creating a new one"""
    try:
        # Find the most recent prediction history record for this XRayImage
        existing_history = PredictionHistory.objects.filter(xray=xray_instance).order_by('-created_at').first()
        
        if existing_history:
            # Update the existing record with current data
            existing_history.atelectasis = xray_instance.atelectasis
            existing_history.cardiomegaly = xray_instance.cardiomegaly
            existing_history.consolidation = xray_instance.consolidation
            existing_history.edema = xray_instance.edema
            existing_history.effusion = xray_instance.effusion
            existing_history.emphysema = xray_instance.emphysema
            existing_history.fibrosis = xray_instance.fibrosis
            existing_history.hernia = xray_instance.hernia
            existing_history.infiltration = xray_instance.infiltration
            existing_history.mass = xray_instance.mass
            existing_history.nodule = xray_instance.nodule
            existing_history.pleural_thickening = xray_instance.pleural_thickening
            existing_history.pneumonia = xray_instance.pneumonia
            existing_history.pneumothorax = xray_instance.pneumothorax
            existing_history.fracture = xray_instance.fracture
            existing_history.lung_opacity = xray_instance.lung_opacity
            existing_history.enlarged_cardiomediastinum = xray_instance.enlarged_cardiomediastinum
            existing_history.lung_lesion = xray_instance.lung_lesion
            existing_history.severity_level = xray_instance.severity_level
            
            # Update model used if different (in case visualization uses different model)
            if existing_history.model_used != model_type:
                existing_history.model_used = f"{existing_history.model_used}+{model_type}"
            
            existing_history.save()
            logger.info(f"Updated existing prediction history record #{existing_history.id} with visualization data")
        else:
            # If no existing record found, create a new one as fallback
            logger.warning(f"No existing prediction history found for XRayImage #{xray_instance.id}, creating new record")
            create_prediction_history(xray_instance, model_type)
    except Exception as e:
        logger.error(f"Error updating prediction history: {str(e)}")
        # Fallback to creating new record if update fails
        create_prediction_history(xray_instance, model_type)


@login_required
def home(request):
    """Home page with image upload form"""
    if request.method == 'POST':
        form = XRayUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            # Create a new XRayImage instance with all form data but don't save yet
            xray_instance = form.save(commit=False)
            # Assign the current user
            xray_instance.user = request.user
            # Now save the instance
            xray_instance.save()
            
            # Get the model type from the form
            model_type = request.POST.get('model_type', 'densenet')
            
            # Save image to disk
            image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
            
            # Start background processing
            thread = threading.Thread(
                target=process_image_async, 
                args=(image_path, xray_instance, model_type)
            )
            thread.start()
            
            # Check if it's an AJAX request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                # Return JSON response for AJAX requests
                return JsonResponse({
                    'upload_id': xray_instance.pk,
                    'xray_id': xray_instance.pk
                })
            else:
                # Redirect for normal form submissions
                return redirect('xray_results', pk=xray_instance.pk)
    else:
        form = XRayUploadForm(user=request.user)
        
    return render(request, 'xrayapp/home.html', {
        'form': form,
        'today_date': timezone.now(),
        'user_first_name': request.user.first_name if request.user.is_authenticated else '',
        'user_last_name': request.user.last_name if request.user.is_authenticated else '',
    })


@login_required
def xray_results(request, pk):
    """View the results of the X-ray analysis"""
    # Get user's hospital from profile
    user_hospital = request.user.profile.hospital
    
    # Allow access to any X-ray from the same hospital
    xray_instance = XRayImage.objects.get(pk=pk, user__profile__hospital=user_hospital)
    
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
    
    # Pathology explanations for educational purposes
    pathology_explanations = {
        'Atelectasis': _('Atelectasis refers to the collapse or incomplete expansion of lung tissue, resulting in reduced gas exchange. It can be caused by airway obstruction (resorption atelectasis), external compression of the lung (compressive atelectasis), or insufficient surfactant production. Common causes include mucus plugging, foreign bodies, tumors, pneumothorax, pleural effusion, and post-surgical complications. On chest X-rays, atelectasis appears as areas of increased opacity with volume loss, often accompanied by compensatory changes such as mediastinal shift, elevated hemidiaphragm, or rib crowding.'),
        
        'Cardiomegaly': _('Cardiomegaly refers to enlargement of the heart, typically identified on chest X-rays when the cardiothoracic ratio exceeds 0.50 on a posteroanterior (PA) view. It can result from various conditions including hypertension, heart failure, cardiomyopathy, valvular disease, or congenital heart defects. The enlarged cardiac silhouette may appear globular or have specific chamber enlargement patterns depending on the underlying cause. Cardiomegaly often indicates underlying cardiovascular disease and may require further evaluation with echocardiography or other cardiac imaging.'),
        
        'Consolidation': _('Consolidation represents the filling of alveolar spaces with fluid, pus, blood, or other material, replacing normal air-filled lung tissue. It appears as homogeneous opacity on chest X-rays, often with air bronchograms visible within the consolidated area. Common causes include pneumonia (bacterial, viral, or fungal), pulmonary edema, hemorrhage, or aspiration. The opacity typically has well-defined borders and may be lobar, segmental, or patchy in distribution. Clinical correlation is essential for determining the underlying cause and appropriate treatment.'),
        
        'Edema': _('Pulmonary edema is the accumulation of excess fluid in the lung tissue and alveolar spaces, resulting in impaired gas exchange. It can be cardiogenic (due to heart failure, valve disease, or fluid overload) or non-cardiogenic (due to acute lung injury, infection, or capillary leak). On chest X-rays, pulmonary edema appears as bilateral, symmetrical opacities that may have a "bat wing" or perihilar distribution. Additional findings may include cardiomegaly, pleural effusions, and prominent pulmonary vasculature. Early recognition is crucial as it can be life-threatening.'),
        
        'Effusion': _('Pleural effusion is the abnormal accumulation of fluid in the pleural space between the lung and chest wall. It can be transudative (due to heart failure, liver disease, or kidney disease) or exudative (due to infection, malignancy, or inflammatory conditions). On chest X-rays, pleural effusion appears as a homogeneous opacity that obscures the diaphragm and costophrenic angle, with a meniscus sign at the fluid-air interface. Large effusions can cause mediastinal shift away from the affected side and require drainage for both diagnostic and therapeutic purposes.'),
        
        'Emphysema': _('Emphysema is a chronic obstructive pulmonary disease characterized by permanent enlargement and destruction of alveolar spaces distal to terminal bronchioles. It is most commonly caused by smoking but can also result from alpha-1 antitrypsin deficiency or occupational exposures. On chest X-rays, emphysema appears as hyperinflation with flattened hemidiaphragms, increased anteroposterior diameter, and decreased lung markings. Advanced cases may show bullae formation and signs of pulmonary hypertension. High-resolution CT is more sensitive for detecting early emphysematous changes.'),
        
        'Fibrosis': _('Pulmonary fibrosis involves the thickening and scarring of lung tissue, leading to progressive loss of lung function. It can be idiopathic or secondary to various causes including occupational exposures (asbestosis, silicosis), medications, radiation therapy, or connective tissue diseases. On chest X-rays, fibrosis appears as reticular or reticulonodular opacities, often with a lower lobe predominance. Advanced cases may show honeycombing, traction bronchiectasis, and loss of lung volume. Early detection and treatment are important to slow disease progression.'),
        
        'Hernia': _('Hiatal hernia occurs when part of the stomach protrudes through the diaphragmatic opening into the thoracic cavity. It can be sliding (most common) or paraesophageal (less common but more serious). On chest X-rays, hiatal hernia may appear as a retrocardiac mass or air-fluid level behind the heart. Large hernias can compress adjacent structures and may be associated with complications such as gastric volvulus, obstruction, or strangulation. Barium studies or CT imaging may be needed for detailed evaluation.'),
        
        'Infiltration': _('Pulmonary infiltration refers to the abnormal accumulation of substances in lung tissue, including inflammatory cells, fluid, or other materials. It appears as areas of increased opacity on chest X-rays and can be caused by various conditions such as pneumonia, pulmonary edema, hemorrhage, or interstitial lung disease. The pattern and distribution of infiltrates can help narrow the differential diagnosis. Infiltrates may be patchy, diffuse, or have specific patterns like ground-glass opacity or crazy-paving pattern on high-resolution imaging.'),
        
        'Mass': _('A pulmonary mass is a focal opacity greater than 3 cm in diameter that appears on chest imaging. Masses can be benign (such as hamartomas or granulomas) or malignant (primary lung cancer or metastases). On chest X-rays, masses appear as well-defined or spiculated opacities that may be accompanied by additional findings such as pleural effusion, lymphadenopathy, or bone lesions. Further evaluation with CT imaging, PET scanning, and tissue sampling is typically required to determine the nature and extent of the mass.'),
        
        'Nodule': _('A pulmonary nodule is a focal opacity less than or equal to 3 cm in diameter surrounded by normal lung tissue. Nodules can be solitary or multiple and may be benign (granulomas, hamartomas) or malignant (primary or metastatic cancer). On chest X-rays, nodules appear as rounded opacities that may be calcified or non-calcified. The size, morphology, and growth rate of nodules help determine the need for further evaluation. CT imaging is often required for detailed characterization and follow-up.'),
        
        'Pleural Thickening': _('Pleural thickening involves the abnormal thickening of the pleural membranes surrounding the lungs, which can be focal or diffuse. It may result from previous infection (empyema, tuberculosis), asbestos exposure, trauma, or malignancy. On chest X-rays, pleural thickening appears as increased opacity along the pleural surfaces, often with blunting of the costophrenic angles. Extensive pleural thickening can restrict lung expansion and cause respiratory symptoms. High-resolution CT is more sensitive for detecting subtle pleural abnormalities and assessing disease extent.'),
        
        'Pneumonia': _('Pneumonia is an inflammatory condition of the lung tissue, usually caused by bacterial, viral, or fungal infections. It can also result from aspiration or chemical irritants. On chest X-rays, pneumonia appears as areas of consolidation or infiltration that may be lobar, segmental, or patchy in distribution. Air bronchograms are often visible within the consolidated areas. Clinical symptoms include fever, cough, shortness of breath, and chest pain. Prompt diagnosis and appropriate antibiotic therapy are essential for optimal outcomes.'),
        
        'Pneumothorax': _('Pneumothorax is the presence of air in the pleural space, causing partial or complete lung collapse. It can be spontaneous (primary in healthy individuals or secondary in those with lung disease) or traumatic (due to injury or medical procedures). On chest X-rays, pneumothorax appears as a lucent area without lung markings, with a visible pleural line separating the collapsed lung from the chest wall. Large pneumothoraces may cause mediastinal shift and require immediate decompression. Small pneumothoraces may resolve spontaneously with observation.'),
        
        'Fracture': _('Rib fractures are breaks in one or more of the bones that form the ribcage, commonly resulting from trauma, falls, or repetitive stress. On chest X-rays, fractures may appear as lucent lines, discontinuity of the cortex, or displacement of bone fragments. Multiple rib fractures can be associated with serious complications including pneumothorax, hemothorax, or injury to underlying organs. Pathological fractures may occur in patients with bone metastases or metabolic bone disease. Pain management and monitoring for complications are important aspects of treatment.'),
        
        'Lung Opacity': _('Lung opacity refers to any area of increased density on chest imaging that obscures normal lung anatomy. It is a general term that encompasses various pathological processes including consolidation, ground-glass opacity, or mass lesions. The pattern, distribution, and associated findings help narrow the differential diagnosis. Common causes include pneumonia, pulmonary edema, interstitial lung disease, or malignancy. Further evaluation with high-resolution CT imaging and clinical correlation is often needed to determine the specific underlying pathology.'),
        
        'Enlarged Cardiomediastinum': _('Enlarged cardiomediastinum refers to an increase in the combined width of the cardiac silhouette and mediastinal contours. On a standard posteroanterior (PA) view, it is commonly identified when the cardiothoracic ratio (maximum horizontal cardiac diameter divided by maximal thoracic diameter) exceeds 0.50. On an anteroposterior (AP) projection—often performed in supine or portable studies—a mediastinal width greater than approximately 6–8 cm at the level of the aortic knob likewise suggests enlargement. Common causes include: Cardiomegaly (e.g., dilated cardiomyopathy, left ventricular hypertrophy), Pericardial effusion, which produces a globular ("water‐bottle") silhouette, Aortic pathology (aneurysm or dissection) leading to mediastinal widening, Mediastinal masses or lymphadenopathy. Recognition of an enlarged cardiomediastinum is pivotal, as it often prompts further evaluation—such as echocardiography for cardiac enlargement or contrast-enhanced CT to assess aortic and mediastinal pathology.'),
        
        'Lung Lesion': _('A lung lesion is any abnormal area or growth in lung tissue that differs from normal lung anatomy. Lesions can be benign or malignant and may include nodules, masses, cysts, or areas of inflammation. On chest X-rays, lesions appear as focal opacities that may vary in size, shape, and density. The characteristics of the lesion, including its borders, calcification pattern, and growth rate, help determine the likelihood of malignancy. Further evaluation with CT imaging, PET scanning, and possibly tissue sampling is often required to establish a definitive diagnosis and guide appropriate treatment.')
    }
    
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
    
    for viz in visualizations:
        viz_data = {
            'id': viz.id,
            'target_pathology': viz.target_pathology,
            'created_at': viz.created_at,
            'model_used': viz.model_used,
            'visualization_url': viz.visualization_url,
            'heatmap_url': viz.heatmap_url,
            'overlay_url': viz.overlay_url,
            'saliency_url': viz.saliency_url,
            'threshold': viz.threshold,
            'visualization_type': viz.visualization_type
        }
        
        if viz.visualization_type in ['gradcam', 'combined_gradcam']:
            gradcam_visualizations.append(viz_data)
        elif viz.visualization_type in ['pli', 'combined_pli']:
            pli_visualizations.append(viz_data)
    
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
def prediction_history(request):
    """View prediction history with advanced filtering and pagination"""
    form = PredictionHistoryFilterForm(request.GET)
    
    # Get user's hospital from profile
    user_hospital = request.user.profile.hospital
    
    # Initialize query with optimized select_related to avoid N+1 queries
    # Filter by users from the same hospital instead of just current user
    query = PredictionHistory.objects.filter(user__profile__hospital=user_hospital)\
                                    .select_related('xray', 'user', 'user__profile')\
                                    .prefetch_related('xray__user')\
                                    .order_by('-created_at')
    
    # Apply filters if the form is valid
    if form.is_valid():
        # Gender filter
        if form.cleaned_data.get('gender'):
            query = query.filter(xray__gender=form.cleaned_data['gender'])
        
        # Age range filter
        if form.cleaned_data.get('age_min') is not None:
            # Calculate date based on minimum age
            min_age_date = timezone.now().date() - relativedelta(years=form.cleaned_data['age_min'])
            query = query.filter(xray__date_of_birth__lte=min_age_date)
            
        if form.cleaned_data.get('age_max') is not None:
            # Calculate date based on maximum age
            max_age_date = timezone.now().date() - relativedelta(years=form.cleaned_data['age_max'] + 1)
            query = query.filter(xray__date_of_birth__gte=max_age_date)
        
        # Date range filter
        if form.cleaned_data.get('date_min'):
            query = query.filter(xray__date_of_xray__gte=form.cleaned_data['date_min'])
            
        if form.cleaned_data.get('date_max'):
            query = query.filter(xray__date_of_xray__lte=form.cleaned_data['date_max'])
        
        # Pathology filter
        if form.cleaned_data.get('pathology') and form.cleaned_data.get('pathology_threshold') is not None:
            threshold = form.cleaned_data['pathology_threshold']
            field_name = form.cleaned_data['pathology']
            
            # Dynamic field filtering
            filter_kwargs = {f"{field_name}__gte": threshold}
            query = query.filter(**filter_kwargs)
    
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
        prediction_history__in=[item.id for item in history_items]
    ).values_list('prediction_history_id', flat=True))
    
    context = {
        'form': form,
        'history_items': history_items,
        'total_count': total_count,
        'saved_record_ids': saved_record_ids,
    }
    
    return render(request, 'xrayapp/prediction_history.html', context)


@login_required
def delete_prediction_history(request, pk):
    """Delete a prediction history record"""
    try:
        # Get user's hospital from profile
        user_hospital = request.user.profile.hospital
        
        # Allow deletion of any record from the same hospital
        history_item = PredictionHistory.objects.get(pk=pk, user__profile__hospital=user_hospital)
        history_item.delete()
        messages.success(request, _('Prediction history record has been deleted.'))
    except PredictionHistory.DoesNotExist:
        messages.error(request, _('Prediction history record not found.'))
    
    return redirect('prediction_history')


@login_required
def delete_all_prediction_history(request):
    """Delete all prediction history records"""
    if request.method == 'POST':
        # Get user's hospital from profile
        user_hospital = request.user.profile.hospital
        
        # Count records before deletion for current hospital
        count = PredictionHistory.objects.filter(user__profile__hospital=user_hospital).count()
        
        # Delete all records for current hospital
        PredictionHistory.objects.filter(user__profile__hospital=user_hospital).delete()
        
        if count > 0:
            messages.success(request, _('All %(count)d prediction history records have been deleted.') % {'count': count})
        else:
            messages.info(request, _('No prediction history records to delete.'))
    
    return redirect('prediction_history')


@login_required
@require_POST
def delete_visualization(request, pk):
    """Delete a visualization result"""
    try:
        # Get the visualization
        visualization = VisualizationResult.objects.get(pk=pk)
        
        # Check if user has permission to delete (must be from same hospital)
        user_hospital = request.user.profile.hospital
        if visualization.xray.user.profile.hospital != user_hospital:
            return JsonResponse({'success': False, 'error': _('Permission denied')}, status=403)
        
        # Delete associated files
        import os
        from django.conf import settings
        
        file_paths = [
            visualization.visualization_path,
            visualization.heatmap_path,
            visualization.overlay_path,
            visualization.saliency_path,
        ]
        
        for file_path in file_paths:
            if file_path:
                try:
                    full_path = os.path.join(settings.MEDIA_ROOT, file_path)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        
        # Delete the visualization record
        visualization.delete()
        
        return JsonResponse({'success': True})
        
    except VisualizationResult.DoesNotExist:
        return JsonResponse({'success': False, 'error': _('Visualization not found')}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def edit_prediction_history(request, pk):
    """Edit a prediction history record"""
    try:
        # Get user's hospital from profile
        user_hospital = request.user.profile.hospital
        
        # Allow editing of any record from the same hospital
        history_item = PredictionHistory.objects.get(pk=pk, user__profile__hospital=user_hospital)
        
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
def generate_interpretability(request, pk):
    """Generate interpretability visualization for an X-ray image"""
    # Get user's hospital from profile
    user_hospital = request.user.profile.hospital
    
    # Allow access to any X-ray from the same hospital
    xray_instance = XRayImage.objects.get(pk=pk, user__profile__hospital=user_hospital)
    
    # Get parameters from request
    interpretation_method = request.GET.get('method', 'gradcam')  # Default to Grad-CAM
    model_type = request.GET.get('model_type', 'densenet')  # Default to DenseNet
    target_class = request.GET.get('target_class', None)  # Default to None (use highest probability class)
    
    # Reset progress to 0 and set status to processing
    xray_instance.progress = 0
    xray_instance.processing_status = 'processing'
    xray_instance.save()
    
    # Get the image path
    image_path = Path(settings.MEDIA_ROOT) / xray_instance.image.name
    
    # Start background processing
    thread = threading.Thread(
        target=process_with_interpretability_async,
        args=(image_path, xray_instance, model_type, interpretation_method, target_class)
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
def check_progress(request, pk):
    """AJAX endpoint to check processing progress"""
    try:
        # Get user's hospital from profile
        user_hospital = request.user.profile.hospital
        
        # Allow access to any X-ray from the same hospital
        xray_instance = XRayImage.objects.get(pk=pk, user__profile__hospital=user_hospital)
        
        response_data = {
            'status': xray_instance.processing_status,
            'progress': xray_instance.progress,
            'xray_id': xray_instance.pk
        }
        
        # If processing is complete, include visualization data
        if xray_instance.progress >= 100 and xray_instance.processing_status == 'complete':
            media_url = settings.MEDIA_URL
            
            # Get all visualizations for this X-ray
            visualizations = VisualizationResult.objects.filter(xray=xray_instance).order_by('-created_at')
            
            # Group visualizations by type
            gradcam_visualizations = []
            pli_visualizations = []
            
            for viz in visualizations:
                viz_data = {
                    'id': viz.id,
                    'target_pathology': viz.target_pathology,
                    'created_at': viz.created_at.isoformat(),
                    'model_used': viz.model_used,
                    'visualization_url': viz.visualization_url,
                    'heatmap_url': viz.heatmap_url,
                    'overlay_url': viz.overlay_url,
                    'saliency_url': viz.saliency_url,
                    'threshold': viz.threshold
                }
                
                if viz.visualization_type in ['gradcam', 'combined_gradcam']:
                    gradcam_visualizations.append(viz_data)
                elif viz.visualization_type in ['pli', 'combined_pli']:
                    pli_visualizations.append(viz_data)
            
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
        
        return JsonResponse(response_data)
    except XRayImage.DoesNotExist:
        return JsonResponse({'error': _('Image not found')}, status=404)


@login_required
def account_settings(request):
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


def logout_confirmation(request):
    """Display a confirmation page before logging out the user."""
    return render(request, 'registration/logout.html')


@require_POST
def set_language(request):
    """Custom language switching view that integrates with user profile"""
    language = request.POST.get('language')
    
    if language and language in [lang[0] for lang in settings.LANGUAGES]:
        # Activate the language for this session
        translation.activate(language)
        
        # Get the redirect URL before creating the response
        redirect_url = request.META.get('HTTP_REFERER', '/')
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
def toggle_save_record(request, pk):
    """Toggle save/unsave a prediction history record via AJAX"""
    try:
        # Get user's hospital from profile
        user_hospital = request.user.profile.hospital
        
        # Get the prediction history record (must be from same hospital)
        prediction_record = PredictionHistory.objects.get(pk=pk, user__profile__hospital=user_hospital)
        
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
def saved_records(request):
    """View saved prediction history records with advanced filtering"""
    form = PredictionHistoryFilterForm(request.GET)
    
    # Get user's saved records with optimized queries
    saved_records_query = SavedRecord.objects.filter(user=request.user)\
                                           .select_related('prediction_history__xray', 'prediction_history__user')\
                                           .prefetch_related('prediction_history__xray__user')\
                                           .order_by('-saved_at')
    
    # Apply filters if the form is valid
    if form.is_valid():
        # Gender filter
        if form.cleaned_data.get('gender'):
            saved_records_query = saved_records_query.filter(prediction_history__xray__gender=form.cleaned_data['gender'])
        
        # Age range filter
        if form.cleaned_data.get('age_min') is not None:
            # Calculate date based on minimum age
            min_age_date = timezone.now().date() - relativedelta(years=form.cleaned_data['age_min'])
            saved_records_query = saved_records_query.filter(prediction_history__xray__date_of_birth__lte=min_age_date)
            
        if form.cleaned_data.get('age_max') is not None:
            # Calculate date based on maximum age
            max_age_date = timezone.now().date() - relativedelta(years=form.cleaned_data['age_max'] + 1)
            saved_records_query = saved_records_query.filter(prediction_history__xray__date_of_birth__gte=max_age_date)
        
        # Date range filter
        if form.cleaned_data.get('date_min'):
            saved_records_query = saved_records_query.filter(prediction_history__xray__date_of_xray__gte=form.cleaned_data['date_min'])
            
        if form.cleaned_data.get('date_max'):
            saved_records_query = saved_records_query.filter(prediction_history__xray__date_of_xray__lte=form.cleaned_data['date_max'])
        
        # Pathology filter
        if form.cleaned_data.get('pathology') and form.cleaned_data.get('pathology_threshold') is not None:
            threshold = form.cleaned_data['pathology_threshold']
            field_name = f"prediction_history__{form.cleaned_data['pathology']}"
            
            # Dynamic field filtering
            filter_kwargs = {f"{field_name}__gte": threshold}
            saved_records_query = saved_records_query.filter(**filter_kwargs)
    
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
def handler400(request, exception=None):
    """400 Bad Request handler."""
    return render(request, 'errors/400.html', status=400)

def handler401(request, exception=None):
    """401 Unauthorized handler."""
    return render(request, 'errors/401.html', status=401)

def handler403(request, exception=None):
    """403 Forbidden handler."""
    return render(request, 'errors/403.html', status=403)

def handler404(request, exception=None):
    """404 Not Found handler."""
    return render(request, 'errors/404.html', status=404)

def handler408(request, exception=None):
    """408 Request Timeout handler."""
    return render(request, 'errors/408.html', status=408)

def handler429(request, exception=None):
    """429 Too Many Requests handler."""
    return render(request, 'errors/429.html', status=429)

def handler500(request):
    """500 Internal Server Error handler."""
    return render(request, 'errors/500.html', status=500)

def handler502(request):
    """502 Bad Gateway handler."""
    return render(request, 'errors/502.html', status=502)

def handler503(request):
    """503 Service Unavailable handler."""
    return render(request, 'errors/503.html', status=503)

def handler504(request):
    """504 Gateway Timeout handler."""
    return render(request, 'errors/504.html', status=504)

def terms_of_service(request):
    return render(request, 'xrayapp/terms_of_service.html')
