from __future__ import annotations

from django import template
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
import pytz
from typing import Any

register = template.Library()

@register.filter
def multiply(value: Any, arg: Any) -> float | str:
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return ''

@register.filter
def percentage(value: Any) -> str:
    """Convert a decimal to percentage format for CSS"""
    try:
        return f"{float(value) * 100:.1f}"
    except (ValueError, TypeError):
        return '0'

@register.filter
def get_top_pathology(prediction_history: Any) -> tuple[str, float]:
    """Get the top pathology (highest probability) from a prediction history item
    Returns a tuple (pathology_name, probability)"""
    pathology_fields = {
        'Atelectasis': prediction_history.atelectasis,
        'Cardiomegaly': prediction_history.cardiomegaly,
        'Consolidation': prediction_history.consolidation,
        'Edema': prediction_history.edema,
        'Effusion': prediction_history.effusion,
        'Emphysema': prediction_history.emphysema,
        'Fibrosis': prediction_history.fibrosis,
        'Hernia': prediction_history.hernia,
        'Infiltration': prediction_history.infiltration,
        'Mass': prediction_history.mass,
        'Nodule': prediction_history.nodule,
        'Pleural Thickening': prediction_history.pleural_thickening,
        'Pneumonia': prediction_history.pneumonia,
        'Pneumothorax': prediction_history.pneumothorax,
        'Fracture': prediction_history.fracture,
        'Lung Opacity': prediction_history.lung_opacity,
    }
    
    # Add DenseNet-only fields if they exist
    if prediction_history.enlarged_cardiomediastinum is not None:
        pathology_fields['Enlarged Cardiomediastinum'] = prediction_history.enlarged_cardiomediastinum
    if prediction_history.lung_lesion is not None:
        pathology_fields['Lung Lesion'] = prediction_history.lung_lesion
    
    # Filter out None values
    pathology_fields = {k: v for k, v in pathology_fields.items() if v is not None}
    
    # Find the max
    if not pathology_fields:
        return ('None', 0.0)
    
    top_pathology = max(pathology_fields.items(), key=lambda x: x[1])
    return top_pathology

@register.filter
def add_class(field: Any, css_class: str) -> Any:
    """Add a CSS class to a form field"""
    return field.as_widget(attrs={"class": css_class}) 

@register.filter
def get_severity_level(obj: Any) -> int | None:
    """Get the severity level (0-3) from a model instance (XRayImage or PredictionHistory)"""
    if hasattr(obj, 'severity_level') and obj.severity_level is not None:
        return obj.severity_level
    elif hasattr(obj, 'calculate_severity_level'):
        return obj.calculate_severity_level
    return None

@register.filter
def get_severity_label(obj: Any) -> str:
    """Get the severity label from a model instance (XRayImage or PredictionHistory)"""
    if hasattr(obj, 'severity_label'):
        return obj.severity_label
        
    # If no severity_label property, calculate manually
    severity_mapping = {
        1: _("Insignificant findings"),
        2: _("Moderate findings"),
        3: _("Significant findings"),
    }
    
    level = None
    if hasattr(obj, 'severity_level') and obj.severity_level is not None:
        level = obj.severity_level
    elif hasattr(obj, 'calculate_severity_level'):
        level = obj.calculate_severity_level
        
    return severity_mapping.get(level, _("Unknown"))

@register.filter
def get_severity_color(level: int | None) -> str:
    """Get appropriate color class based on severity level"""
    if level == 1:
        return "text-success"  # green for insignificant findings
    elif level == 2:
        return "text-warning"  # yellow for moderate findings
    elif level == 3:
        return "text-danger"   # red for significant findings
    return "" 

@register.simple_tag
def current_eest_time() -> str:
    """Get current server time in EEST timezone with seconds precision"""
    eest = pytz.timezone('Europe/Tallinn')  # EEST (UTC+3)
    current_time = timezone.now().astimezone(eest)
    return current_time.strftime('%Y-%m-%d %H:%M:%S EEST (UTC+3)') 