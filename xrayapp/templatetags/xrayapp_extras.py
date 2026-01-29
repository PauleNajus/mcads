from __future__ import annotations

from django import template
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
import pytz
from typing import Any
from ..models import PATHOLOGY_FIELDS, RESULT_KEY_TO_FIELD, SEVERITY_MAPPING

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
    
    # Invert the key mapping to get display names from field names
    FIELD_TO_DISPLAY = {v: k for k, v in RESULT_KEY_TO_FIELD.items()}
    
    pathology_values = {}
    for field in PATHOLOGY_FIELDS:
        val = getattr(prediction_history, field, None)
        if val is not None:
            # Use display name if available, else capitalize field name
            name = FIELD_TO_DISPLAY.get(field, field.replace('_', ' ').title())
            pathology_values[name] = val
    
    # Find the max
    if not pathology_values:
        return ('None', 0.0)
    
    top_pathology = max(pathology_values.items(), key=lambda x: x[1])
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
        
    level: int | None = None
    if hasattr(obj, 'severity_level') and obj.severity_level is not None:
        level = obj.severity_level
    elif hasattr(obj, 'calculate_severity_level'):
        level = obj.calculate_severity_level
        
    if level is None:
        return str(_("Unknown"))

    return str(SEVERITY_MAPPING.get(level, _("Unknown")))

@register.filter
def get_severity_color(level: int | None) -> str:
    """Get appropriate color class based on severity level (MTS)
    
    Mapping to MTS levels:
    1: Immediate (Red)
    2: Very Urgent (Orange)
    3: Urgent (Yellow)
    4: Standard (Green)
    5: Non-urgent (Blue)
    """
    if level == 1:
        return "text-danger"  # Red - Immediate
    elif level == 2:
        return "text-orange"  # Orange - Very Urgent (will need custom CSS class or inline style)
    elif level == 3:
        return "text-warning"  # Yellow - Urgent
    elif level == 4:
        return "text-success"  # Green - Standard
    elif level == 5:
        return "text-info"     # Blue - Non-urgent
    return "" 

@register.simple_tag
def current_eest_time() -> str:
    """Get current server time in EEST timezone with seconds precision"""
    eest = pytz.timezone('Europe/Tallinn')  # EEST (UTC+3)
    current_time = timezone.now().astimezone(eest)
    return current_time.strftime('%Y-%m-%d %H:%M:%S EEST (UTC+3)')
