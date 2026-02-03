from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from django.db import models
from django.utils.translation import gettext_lazy as _

logger = logging.getLogger(__name__)

# --- Domain constants ---------------------------------------------------------
#
# Keep shared domain lists/logic in one place. This reduces duplication across
# models, admin, views and template tags without changing business behavior.


class UserRole(models.TextChoices):
    """Supported user roles for RBAC checks."""

    ADMINISTRATOR = "Administrator", _("Administrator")
    RADIOGRAPHER = "Radiographer", _("Radiographer")
    TECHNOLOGIST = "Technologist", _("Technologist")
    RADIOLOGIST = "Radiologist", _("Radiologist")


# Normalized pathology field names used by both `XRayImage` and `PredictionHistory`.
# NOTE: These are the DB field names (snake_case), not the model output labels.
PATHOLOGY_FIELDS: tuple[str, ...] = (
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "fibrosis",
    "hernia",
    "infiltration",
    "mass",
    "nodule",
    "pleural_thickening",
    "pneumonia",
    "pneumothorax",
    "fracture",
    "lung_opacity",
    # DenseNet-only outputs (may be NULL depending on model)
    "enlarged_cardiomediastinum",
    "lung_lesion",
)


# Mapping from model output keys to our model field names.
# TorchXRayVision has historically used both spaces and underscores in labels;
# we support both to remain backwards compatible.
RESULT_KEY_TO_FIELD: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Effusion": "effusion",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Hernia": "hernia",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pleural_Thickening": "pleural_thickening",
    "Pleural Thickening": "pleural_thickening",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Fracture": "fracture",
    "Lung Opacity": "lung_opacity",
    "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
    "Lung Lesion": "lung_lesion",
}

# Centralized severity mapping for consistent display across the app (MTS)
SEVERITY_MAPPING = {
    1: _("Immediate (Red)"),
    2: _("Very Urgent (Orange)"),
    3: _("Urgent (Yellow)"),
    4: _("Standard (Green)"),
    5: _("Non-urgent (Blue)"),
}


def _iter_present_pathology_values(obj: object) -> list[float]:
    """Return all non-null pathology values from a model instance.

    This is intentionally Python-side. The DB-side equivalent is implemented
    where it matters (list views / sorting) using `annotate()`.
    """

    values: list[float] = []
    for field in PATHOLOGY_FIELDS:
        raw = getattr(obj, field, None)
        if raw is not None:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                # Defensive: the DB field is FloatField; unexpected types should be logged.
                logger.warning("Unexpected pathology value type for %s=%r", field, raw)
    return values


def _severity_from_values(values: Iterable[float]) -> int | None:
    """Compute severity level from pathology probability values using MTS logic.
    
    Manchester Triage System (MTS) approximation based on AI probabilities:
    1: Immediate (Red) - Very high probability of critical condition (>80%)
    2: Very Urgent (Orange) - High probability of significant pathology (>60%)
    3: Urgent (Yellow) - Moderate probability (>40%)
    4: Standard (Green) - Low but present probability (>20%)
    5: Non-urgent (Blue) - Very low probability (<20%)
    """

    values_list = list(values)
    if not values_list:
        return None
    
    # Use the maximum probability found as the primary risk indicator
    # (A single critical finding drives the triage urgency, not the average)
    max_probability = max(values_list)
    
    if max_probability >= 0.80:
        return 1  # Immediate
    elif max_probability >= 0.60:
        return 2  # Very Urgent
    elif max_probability >= 0.40:
        return 3  # Urgent
    elif max_probability >= 0.20:
        return 4  # Standard
    else:
        return 5  # Non-urgent


# --- QuerySets / Managers -----------------------------------------------------
#
# These keep the ORM usage centralized and consistent (select_related/prefetch,
# hospital scoping, etc.). Views should call these helpers instead of repeating
# filters in many places.


class XRayImageQuerySet(models.QuerySet):
    def for_hospital(self, hospital: str) -> "XRayImageQuerySet":
        return self.filter(user__profile__hospital=hospital)

    def with_user_profile(self) -> "XRayImageQuerySet":
        return self.select_related("user", "user__profile")


XRayImageManager = models.Manager.from_queryset(XRayImageQuerySet)


class PredictionHistoryQuerySet(models.QuerySet):
    def for_hospital(self, hospital: str) -> "PredictionHistoryQuerySet":
        return self.filter(user__profile__hospital=hospital)

    def with_related(self) -> "PredictionHistoryQuerySet":
        # Join everything templates/admin typically touch to avoid N+1 queries.
        return self.select_related(
            "xray",
            "user",
            "user__profile",
            "xray__user",
            "xray__user__profile",
        )


PredictionHistoryManager = models.Manager.from_queryset(PredictionHistoryQuerySet)


class VisualizationResultQuerySet(models.QuerySet):
    def with_xray_user_profile(self) -> "VisualizationResultQuerySet":
        return self.select_related("xray", "xray__user", "xray__user__profile")


VisualizationResultManager = models.Manager.from_queryset(VisualizationResultQuerySet)


class SavedRecordQuerySet(models.QuerySet):
    def for_user(self, user: Any) -> "SavedRecordQuerySet":
        # Accept Any to support custom User models without importing auth types.
        return self.filter(user=user)

    def with_related(self) -> "SavedRecordQuerySet":
        return self.select_related(
            "prediction_history",
            "prediction_history__user",
            "prediction_history__user__profile",
            "prediction_history__xray",
            "prediction_history__xray__user",
            "prediction_history__xray__user__profile",
        )


SavedRecordManager = models.Manager.from_queryset(SavedRecordQuerySet)

# Create your models here.

class XRayImage(models.Model):
    """Model to store X-ray images and analysis results"""
    # User who uploaded the image
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='xray_images', null=True, db_index=True)
    
    # Patient Information
    first_name = models.CharField(max_length=100, blank=True, db_index=True)
    last_name = models.CharField(max_length=100, blank=True, db_index=True)
    patient_id = models.CharField(max_length=100, blank=True, db_index=True)
    gender = models.CharField(max_length=10, blank=True, db_index=True)
    date_of_birth = models.DateField(null=True, blank=True, db_index=True)
    date_of_xray = models.DateField(null=True, blank=True, db_index=True)
    additional_info = models.TextField(blank=True)
    
    # Technologist Information
    technologist_first_name = models.CharField(max_length=100, blank=True, db_index=True)
    technologist_last_name = models.CharField(max_length=100, blank=True, db_index=True)
    
    # X-ray image and processing
    #
    # NOTE: We intentionally use FileField (not ImageField) so clinical uploads
    # can be stored and processed directly as DICOM (.dcm) without converting to
    # PNG/JPG. Browsers can't render DICOM in <img> tags; templates handle that
    # by showing a download link for DICOM uploads.
    image = models.FileField(upload_to='xrays/', max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True, db_index=True)
    processing_status = models.CharField(max_length=20, default='pending', db_index=True)
    progress = models.IntegerField(default=0)
    
    # Image metadata
    image_format = models.CharField(max_length=10, blank=True)  # e.g., 'JPEG', 'PNG'
    image_size = models.CharField(max_length=20, blank=True)    # e.g., '2.4 MB'
    image_resolution = models.CharField(max_length=20, blank=True)  # e.g., '1024x768'
    image_date_created = models.DateTimeField(null=True, blank=True)
    
    # Interpretability visualizations
    has_gradcam = models.BooleanField(default=False, db_index=True)
    gradcam_visualization = models.CharField(max_length=255, null=True, blank=True)
    gradcam_heatmap = models.CharField(max_length=255, null=True, blank=True)
    gradcam_overlay = models.CharField(max_length=255, null=True, blank=True)
    gradcam_target_class = models.CharField(max_length=50, null=True, blank=True)
    
    has_pli = models.BooleanField(default=False, db_index=True)
    pli_visualization = models.CharField(max_length=255, null=True, blank=True)
    pli_overlay_visualization = models.CharField(max_length=255, null=True, blank=True)
    pli_saliency_map = models.CharField(max_length=255, null=True, blank=True)
    pli_target_class = models.CharField(max_length=50, null=True, blank=True)
    
    # Predicted pathologies (values range from 0.0 to 1.0)
    atelectasis = models.FloatField(null=True, blank=True, db_index=True)
    cardiomegaly = models.FloatField(null=True, blank=True, db_index=True)
    consolidation = models.FloatField(null=True, blank=True, db_index=True)
    edema = models.FloatField(null=True, blank=True, db_index=True)
    effusion = models.FloatField(null=True, blank=True, db_index=True)
    emphysema = models.FloatField(null=True, blank=True, db_index=True)
    fibrosis = models.FloatField(null=True, blank=True, db_index=True)
    hernia = models.FloatField(null=True, blank=True, db_index=True)
    infiltration = models.FloatField(null=True, blank=True, db_index=True)
    mass = models.FloatField(null=True, blank=True, db_index=True)
    nodule = models.FloatField(null=True, blank=True, db_index=True)
    pleural_thickening = models.FloatField(null=True, blank=True, db_index=True)
    pneumonia = models.FloatField(null=True, blank=True, db_index=True)
    pneumothorax = models.FloatField(null=True, blank=True, db_index=True)
    fracture = models.FloatField(null=True, blank=True, db_index=True)
    lung_opacity = models.FloatField(null=True, blank=True, db_index=True)
    enlarged_cardiomediastinum = models.FloatField(null=True, blank=True, db_index=True)
    lung_lesion = models.FloatField(null=True, blank=True, db_index=True)
    
    # Severity level
    severity_level = models.IntegerField(null=True, blank=True, db_index=True)
    
    # Expert review requirement
    requires_expert_review = models.BooleanField(default=False, db_index=True)
    
    # Model used for analysis
    model_used = models.CharField(max_length=50, default='densenet', db_index=True)

    # Centralized queryset helpers (hospital scoping, select_related, etc.)
    objects = XRayImageManager()
    
    class Meta:
        # Add composite indexes for commonly queried combinations
        indexes = [
            models.Index(fields=['user', 'uploaded_at']),
            models.Index(fields=['processing_status', 'uploaded_at']),
            models.Index(fields=['patient_id', 'date_of_xray']),
            models.Index(fields=['gender', 'date_of_birth']),
            models.Index(fields=['severity_level', 'uploaded_at']),
        ]
        # Optimize database table order
        ordering = ['-uploaded_at']
    
    @property
    def calculate_severity_level(self) -> int | None:
        """Calculate severity level based on max pathology probability (MTS)"""
        return _severity_from_values(_iter_present_pathology_values(self))
    
    @property
    def severity_label(self) -> str:
        """Get severity level label"""
        level = self.severity_level
        if level is None:
            calculated = self.calculate_severity_level
            if calculated is None:
                return _("Unknown")
            return str(SEVERITY_MAPPING.get(calculated, _("Unknown")))
        return str(SEVERITY_MAPPING.get(level, _("Unknown")))
    
    def __str__(self) -> str:
        if self.patient_id and (self.first_name or self.last_name):
            return f"{self.first_name} {self.last_name} (ID: {self.patient_id}) - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
        return f"X-ray #{self.pk} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
        
    def get_patient_display(self) -> str:
        """Return formatted patient information"""
        if self.first_name or self.last_name:
            return f"{self.first_name} {self.last_name}".strip()
        return _("Unknown patient")

    def apply_predictions_from_results(self, results: Mapping[str, Any]) -> None:
        """Update pathology fields from an inference results mapping.

        The inference pipeline returns a dict-like structure keyed by pathology
        names. We normalize that to our DB field names using `RESULT_KEY_TO_FIELD`.
        """

        for key, field_name in RESULT_KEY_TO_FIELD.items():
            if key not in results:
                continue
            value = results.get(key)
            setattr(self, field_name, None if value is None else float(value))


class PredictionHistory(models.Model):
    """Model to store prediction history with filtering capabilities"""
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='prediction_history', null=True, db_index=True)
    xray = models.ForeignKey(XRayImage, on_delete=models.CASCADE, related_name='prediction_history', db_index=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    model_used = models.CharField(max_length=50, db_index=True)  # densenet, resnet, etc.
    
    # Predicted pathologies - copied from XRayImage for historical record
    atelectasis = models.FloatField(null=True, blank=True)
    cardiomegaly = models.FloatField(null=True, blank=True)
    consolidation = models.FloatField(null=True, blank=True)
    edema = models.FloatField(null=True, blank=True)
    effusion = models.FloatField(null=True, blank=True)
    emphysema = models.FloatField(null=True, blank=True)
    fibrosis = models.FloatField(null=True, blank=True)
    hernia = models.FloatField(null=True, blank=True)
    infiltration = models.FloatField(null=True, blank=True)
    mass = models.FloatField(null=True, blank=True)
    nodule = models.FloatField(null=True, blank=True)
    pleural_thickening = models.FloatField(null=True, blank=True)
    pneumonia = models.FloatField(null=True, blank=True)
    pneumothorax = models.FloatField(null=True, blank=True)
    fracture = models.FloatField(null=True, blank=True)
    lung_opacity = models.FloatField(null=True, blank=True)
    enlarged_cardiomediastinum = models.FloatField(null=True, blank=True)
    lung_lesion = models.FloatField(null=True, blank=True)
    
    # Severity level
    severity_level = models.IntegerField(null=True, blank=True, db_index=True)

    objects = PredictionHistoryManager()
    
    class Meta:
        # Add composite indexes for commonly queried combinations
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['model_used', 'created_at']),
            models.Index(fields=['severity_level', 'created_at']),
        ]
        # Optimize database table order
        ordering = ['-created_at']
    
    @property
    def calculate_severity_level(self) -> int | None:
        """Calculate severity level based on max pathology probability (MTS)"""
        return _severity_from_values(_iter_present_pathology_values(self))
    
    @property
    def severity_label(self) -> str:
        """Get severity level label"""
        level = self.severity_level
        if level is None:
            calculated = self.calculate_severity_level
            if calculated is None:
                return _("Unknown")
            return str(SEVERITY_MAPPING.get(calculated, _("Unknown")))
        return str(SEVERITY_MAPPING.get(level, _("Unknown")))
    
    def __str__(self) -> str:
        return f"Prediction #{self.pk} for {self.xray} using {self.model_used}"

    @classmethod
    def create_from_xray(cls, xray: XRayImage, model_used: str) -> "PredictionHistory | None":
        """Create a `PredictionHistory` snapshot from an `XRayImage`.

        This centralizes a historically duplicated mapping across views/tasks.
        """

        if xray.user is None:
            logger.warning("XRayImage %s has no user; skipping history creation", xray.pk)
            return None

        return cls.objects.create(
            user=xray.user,
            xray=xray,
            model_used=model_used,
            severity_level=xray.severity_level,
            **{field: getattr(xray, field) for field in PATHOLOGY_FIELDS},
        )

    def sync_from_xray(self, xray: XRayImage, model_used: str | None = None) -> None:
        """Update this record from an `XRayImage` (keeps history in sync).

        Note: This does not change the `xray` relation, only the copied fields.
        """

        for field in PATHOLOGY_FIELDS:
            setattr(self, field, getattr(xray, field))
        self.severity_level = xray.severity_level

        if model_used and self.model_used != model_used:
            self.model_used = f"{self.model_used}+{model_used}"

        self.save(update_fields=[*PATHOLOGY_FIELDS, "severity_level", "model_used"])

    @classmethod
    def update_latest_for_xray(cls, xray: XRayImage, model_used: str) -> "PredictionHistory | None":
        """Update the most recent history record for an XRayImage.

        Used when interpretability/segmentation runs after inference and should
        attach results to the existing entry shown in "History Records".
        """

        latest = cls.objects.filter(xray=xray).order_by("-created_at").first()
        if latest:
            latest.sync_from_xray(xray, model_used=model_used)
            return latest
        return cls.create_from_xray(xray, model_used=model_used)


class VisualizationResult(models.Model):
    """Model to store multiple interpretability visualizations for each X-ray image"""
    
    # Visualization type choices
    VISUALIZATION_TYPES = [
        ('gradcam', _('GRAD-CAM')),
        ('pli', _('Pixel-level Interpretability')),
        ('combined_gradcam', _('Combined')),
        ('combined_pli', _('Combined PLI')),
        ('segmentation', _('Anatomical Segmentation')),
        ('segmentation_combined', _('Combined Segmentation')),
    ]
    
    # Foreign key to X-ray image
    xray = models.ForeignKey(XRayImage, on_delete=models.CASCADE, related_name='visualizations', db_index=True)
    
    # Visualization details
    visualization_type = models.CharField(max_length=30, choices=VISUALIZATION_TYPES, db_index=True)
    target_pathology = models.CharField(max_length=50, db_index=True)  # The pathology this visualization targets
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    
    # File paths for visualization images
    visualization_path = models.CharField(max_length=255, null=True, blank=True)  # Main visualization
    heatmap_path = models.CharField(max_length=255, null=True, blank=True)      # Heatmap
    overlay_path = models.CharField(max_length=255, null=True, blank=True)      # Overlay image
    saliency_path = models.CharField(max_length=255, null=True, blank=True)     # Saliency map (PLI)
    
    # Additional metadata
    model_used = models.CharField(max_length=50, blank=True)  # Model used for visualization
    threshold = models.FloatField(null=True, blank=True)     # Threshold used (for PLI)
    confidence_score = models.FloatField(null=True, blank=True)  # Confidence score for segmentation
    metadata = models.JSONField(null=True, blank=True)  # Additional metadata as JSON

    objects = VisualizationResultManager()
    
    class Meta:
        # Unique constraint: prevent duplicate visualization type + pathology combinations
        constraints = [
            models.UniqueConstraint(
                fields=['xray', 'visualization_type', 'target_pathology'],
                name='unique_visualization_per_pathology'
            )
        ]
        
        # Add indexes for common queries
        indexes = [
            models.Index(fields=['xray', 'visualization_type']),
            models.Index(fields=['xray', 'target_pathology']),
            models.Index(fields=['created_at']),
        ]
        
        # Order by creation time (newest first)
        ordering = ['-created_at']
    
    def __str__(self) -> str:
        return f"{self.visualization_type_display} - {self.target_pathology} for X-ray #{self.xray.pk}"

    @property
    def visualization_type_display(self) -> str:
        """Return human-readable label for visualization_type."""
        choices_map = {key: label for key, label in self.VISUALIZATION_TYPES}
        return str(choices_map.get(self.visualization_type, self.visualization_type))
    
    @property
    def visualization_url(self) -> str | None:
        """Get the main visualization URL"""
        if self.visualization_path:
            from django.conf import settings
            return f"{settings.MEDIA_URL}{self.visualization_path}"
        return None
    
    @property
    def heatmap_url(self) -> str | None:
        """Get the heatmap URL"""
        if self.heatmap_path:
            from django.conf import settings
            return f"{settings.MEDIA_URL}{self.heatmap_path}"
        return None
    
    @property
    def overlay_url(self) -> str | None:
        """Get the overlay URL"""
        if self.overlay_path:
            from django.conf import settings
            return f"{settings.MEDIA_URL}{self.overlay_path}"
        return None
    
    @property
    def saliency_url(self) -> str | None:
        """Get the saliency map URL"""
        if self.saliency_path:
            from django.conf import settings
            return f"{settings.MEDIA_URL}{self.saliency_path}"
        return None


class UserProfile(models.Model):
    """Model to store additional user settings and preferences"""
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE, related_name='profile')
    
    # Role-based access control
    role = models.CharField(
        max_length=20,
        choices=UserRole.choices,
        default=UserRole.RADIOGRAPHER,
        db_index=True
    )
    
    # Hospital affiliation
    hospital = models.CharField(
        max_length=100,
        default='VULSK',
        db_index=True,
        help_text='Hospital name for shared prediction history'
    )
    
    # User preferences
    preferred_theme = models.CharField(
        max_length=10,
        choices=[('auto', _('System Default')), ('light', _('Light')), ('dark', _('Dark'))],
        default='auto'
    )
    preferred_language = models.CharField(
        max_length=10,
        choices=[('en', _('English')), ('lt', _('Lithuanian'))],
        default='en'
    )
    dashboard_view = models.CharField(
        max_length=10,
        choices=[('grid', _('Grid View')), ('list', _('List View'))],
        default='grid'
    )
    email_notifications = models.BooleanField(default=True)
    processing_complete_notification = models.BooleanField(default=True)
    two_factor_auth_enabled = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Role-based permission methods
    def can_access_admin(self) -> bool:
        """Check if user can access admin panel"""
        return self.role == UserRole.ADMINISTRATOR
    
    def can_upload_xrays(self) -> bool:
        """Check if user can upload X-ray images"""
        return self.role in {UserRole.ADMINISTRATOR, UserRole.RADIOGRAPHER, UserRole.TECHNOLOGIST}
    
    def can_view_all_patients(self) -> bool:
        """Check if user can view all patients' data"""
        return self.role in {UserRole.ADMINISTRATOR, UserRole.RADIOGRAPHER, UserRole.RADIOLOGIST}
    
    def can_edit_predictions(self) -> bool:
        """Check if user can edit prediction results"""
        return self.role in {UserRole.ADMINISTRATOR, UserRole.RADIOGRAPHER, UserRole.RADIOLOGIST}
    
    def can_delete_data(self) -> bool:
        """Check if user can delete X-ray data"""
        return self.role in {UserRole.ADMINISTRATOR, UserRole.RADIOGRAPHER}
    
    def can_generate_interpretability(self) -> bool:
        """Check if user can generate interpretability visualizations"""
        return self.role in {UserRole.ADMINISTRATOR, UserRole.RADIOGRAPHER, UserRole.RADIOLOGIST}
    
    def can_manage_users(self) -> bool:
        """Check if user can manage other users"""
        return self.role == UserRole.ADMINISTRATOR
    
    def can_view_prediction_history(self) -> bool:
        """Check if user can view prediction history"""
        return True  # All users can view their own history
    
    def can_change_settings(self) -> bool:
        """Check if user can change account settings"""
        return True  # All users can change their own settings

    def __str__(self) -> str:
        return f"{self.user.username} - {self.role}"


class SavedRecord(models.Model):
    """Model to store user-saved prediction history records"""
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='saved_records', db_index=True)
    prediction_history = models.ForeignKey(PredictionHistory, on_delete=models.CASCADE, related_name='saved_by_users', db_index=True)
    saved_at = models.DateTimeField(auto_now_add=True, db_index=True)

    objects = SavedRecordManager()
    
    class Meta:
        # Ensure a user can only save a record once
        unique_together = [('user', 'prediction_history')]
        # Order by most recently saved first
        ordering = ['-saved_at']
        indexes = [
            models.Index(fields=['user', 'saved_at']),
            models.Index(fields=['prediction_history', 'saved_at']),
        ]
    
    def __str__(self) -> str:
        return f"{self.user.username} saved #{self.prediction_history.pk}"