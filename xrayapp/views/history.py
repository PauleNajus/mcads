from __future__ import annotations

import logging
from typing import Any

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Case, F, FloatField, IntegerField, Value, When
from django.db.models.functions import Coalesce
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from xrayapp.forms import (
    PredictionHistoryFilterForm,
    XRayUploadForm,
)
from xrayapp.models import (
    PATHOLOGY_FIELDS,
    PredictionHistory,
    SavedRecord,
)
from .utils import (
    _get_user_hospital,
    _apply_history_filters,
)

logger = logging.getLogger(__name__)


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
            
            # List of all pathology fields
            pathology_fields = PATHOLOGY_FIELDS
            
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
        logger.exception("Error toggling saved record pk=%s user=%s", pk, request.user.pk)
        payload: dict[str, Any] = {
            'success': False,
            'error': _('An error occurred while updating saved records.'),
        }
        if settings.DEBUG:
            payload['details'] = str(e)
        return JsonResponse(payload, status=500)


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
