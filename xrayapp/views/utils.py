from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.utils import timezone

from xrayapp.models import (
    XRayImage,
    VisualizationResult,
    PredictionHistory,
    SavedRecord,
    UserProfile,
)
from xrayapp.services import perform_inference, perform_interpretability, perform_segmentation

logger = logging.getLogger(__name__)


def _get_user_hospital(user: Any) -> str | None:
    """Return the user's hospital (from `UserProfile`) if available."""
    try:
        profile = user.profile  # type: ignore[attr-defined]
    except Exception:
        # Catch Exception for ObjectDoesNotExist or AttributeError without importing it explicitly
        return None
    return getattr(profile, "hospital", None)


def _serialize_visualization(viz: VisualizationResult) -> dict[str, Any]:
    """Serialize a VisualizationResult object for JSON/template response."""
    return {
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
        # Reload instance to ensure thread safety
        xray = XRayImage.objects.get(pk=xray_instance.pk)
        perform_inference(xray, model_type)
    except Exception:
        # services.perform_inference logs exceptions
        pass


def process_with_interpretability_async(
    image_path: Path,
    xray_instance: XRayImage,
    model_type: str,
    interpretation_method: str,
    target_class: str | None = None,
) -> None:
    """Process the image with interpretability visualization in a background thread"""
    try:
        # Reload instance to ensure thread safety
        xray = XRayImage.objects.get(pk=xray_instance.pk)
        perform_interpretability(xray, model_type, interpretation_method, target_class)
    except Exception:
        pass


def process_segmentation_async(xray_instance: XRayImage) -> None:
    """Process segmentation in a background thread"""
    try:
        # Reload instance to ensure thread safety
        xray = XRayImage.objects.get(pk=xray_instance.pk)
        perform_segmentation(xray)
    except Exception:
        pass
