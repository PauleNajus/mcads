from __future__ import annotations

import logging

from datetime import date
from io import BytesIO

from django import forms
from django.conf import settings
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.utils.translation import gettext_lazy as _
from django.core.files.uploadedfile import UploadedFile
from django.contrib.auth.models import User

from .models import XRayImage, UserProfile
from .image_processing import (
    looks_like_dicom,
    convert_dicom_to_png,
    get_image_mime_type,
    MAGIC_AVAILABLE
)

logger = logging.getLogger(__name__)


class XRayUploadForm(forms.ModelForm):
    # Accept DICOM uploads and convert them to PNG during validation.
    # We use FileField (not ImageField) so `.dcm` can pass form parsing.
    image = forms.FileField(required=True)

    class Meta:
        model = XRayImage
        fields = ['image', 'first_name', 'last_name', 'patient_id', 'gender', 
                 'date_of_birth', 'date_of_xray', 'additional_info', 
                 'technologist_first_name', 'technologist_last_name']
        widgets = {
            'date_of_birth': forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}),
            'date_of_xray': forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD', 'value': timezone.now().strftime('%Y-%m-%d')}),
            'gender': forms.Select(choices=[('', _('--Select--')), ('male', _('Male')), ('female', _('Female')), ('other', _('Other'))]),
            'additional_info': forms.Textarea(attrs={'rows': 3, 'maxlength': 1000}),
        }
    
    def __init__(self, *args: object, **kwargs: object) -> None:
        user_obj = kwargs.pop('user', None)
        # Cast to correct type for mypy
        user: User | None = user_obj if isinstance(user_obj, User) else None
        
        super().__init__(*args, **kwargs)
        
        # Auto-populate technologist fields with current user's information
        if user and user.is_authenticated:
            # Only set initial values if the fields are not already populated
            if not self.data.get('technologist_first_name') and not self.initial.get('technologist_first_name'):
                self.fields['technologist_first_name'].initial = user.first_name or ''
            if not self.data.get('technologist_last_name') and not self.initial.get('technologist_last_name'):
                self.fields['technologist_last_name'].initial = user.last_name or ''
        
        # We validate file type in `clean_image()` using content-based checks.
        # This also allows extensionless DICOM uploads (common in PACS exports).
        # Add input length limits for security
        self.fields['first_name'].widget.attrs.update({'maxlength': 100})
        self.fields['last_name'].widget.attrs.update({'maxlength': 100})
        self.fields['patient_id'].widget.attrs.update({'maxlength': 100})
        self.fields['technologist_first_name'].widget.attrs.update({'maxlength': 100})
        self.fields['technologist_last_name'].widget.attrs.update({'maxlength': 100})
    
    def clean_image(self) -> UploadedFile | ContentFile | None:
        """Validate uploaded image/DICOM file.

        DICOM uploads are converted to PNG so the rest of the app can keep using
        PIL/skimage and render previews via `<img>`.
        """
        image = self.cleaned_data.get('image')
        if not image:
            return image
            
        # Check file size (keep consistent with Django + Nginx limits).
        max_bytes = int(getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", 10 * 1024 * 1024) or 10 * 1024 * 1024)
        if image.size > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            raise ValidationError(_('Image file too large. Maximum size is %(max_mb).0fMB.') % {'max_mb': max_mb})

        # Read a small header once (speed + memory); always reset the pointer.
        header = b""
        try:
            header = image.read(4096)
        finally:
            image.seek(0)

        is_dicom = looks_like_dicom(getattr(image, "name", None), header)
        
        # Check MIME type for security if magic is available
        file_mime: str | None = None
        if MAGIC_AVAILABLE and not is_dicom:
            file_mime = get_image_mime_type(header)
            
            # Common safe image MIME types.
            allowed_mimes = {
                'image/jpeg', 'image/jpg', 'image/png',
                'image/bmp', 'image/tiff', 'image/x-ms-bmp',
            }
            # Some libmagic installations may label DICOM explicitly.
            dicom_mimes = {'application/dicom'}

            if file_mime in dicom_mimes:
                is_dicom = True
            elif file_mime and file_mime not in allowed_mimes:
                # `application/octet-stream` is ambiguous and is often returned for
                # DICOM exports without the "DICM" marker. Don't reject it here;
                # we'll fall back to Pillow verification and then attempt DICOM decode
                # if the payload isn't a valid image.
                if file_mime != 'application/octet-stream':
                    raise ValidationError(_('Invalid file type. Only image files are allowed.'))

        # If this is a DICOM file, convert it to PNG before saving to the model.
        if is_dicom:
            # Persist the original upload "format" separately from the stored file format.
            # We convert DICOM → PNG for processing/display, but we still want user-facing
            # metadata to say "DICOM".
            self._mcads_source_format = "DICOM"
            return convert_dicom_to_png(image)

        # For non-DICOM uploads, we must validate that the content is a real image.
        # (We use FileField to accept .dcm, so we need to reintroduce ImageField-like verification.)
        try:
            from PIL import Image, UnidentifiedImageError
        except Exception as exc:  # pragma: no cover (dependency/runtime)
            raise ValidationError(_("Image validation is not available on this server.")) from exc

        # Quick extension sanity check (optional; content checks are the source of truth).
        # Only enforce this when MIME detection is unavailable/ambiguous.
        name = (getattr(image, "name", "") or "").lower()
        if (not MAGIC_AVAILABLE) or (file_mime in (None, 'application/octet-stream')):
            if "." in name:
                ext = name.rsplit(".", 1)[-1]
                allowed_exts = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
                if ext not in allowed_exts:
                    raise ValidationError(_('Invalid file type. Only image files are allowed.'))

        # Verify the image can be opened by Pillow.
        #
        # If Pillow cannot read it, try DICOM conversion as a fallback. This enables
        # extensionless/preamble-less DICOM uploads that don't include the "DICM" marker.
        try:
            image.seek(0)
            with Image.open(image) as img:
                img.verify()
        except (UnidentifiedImageError, OSError):
            try:
                self._mcads_source_format = "DICOM"
                return convert_dicom_to_png(image)
            except ValidationError as exc:
                raise ValidationError(_("Invalid image file.")) from exc
        finally:
            image.seek(0)

        return image
    
    def _clean_name_field(self, field_name: str, label: str) -> str:
        """Helper to sanitize name inputs."""
        value = self.cleaned_data.get(field_name, '').strip()
        if value and not value.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('%(label)s should only contain letters, spaces, hyphens, and apostrophes.') % {'label': label})
        return value

    def clean_first_name(self) -> str:
        return self._clean_name_field('first_name', _('First name'))
    
    def clean_last_name(self) -> str:
        return self._clean_name_field('last_name', _('Last name'))
    
    def clean_patient_id(self) -> str:
        """Validate patient ID format"""
        patient_id = self.cleaned_data.get('patient_id', '').strip()
        if patient_id and not patient_id.replace('-', '').replace('_', '').isalnum():
            raise ValidationError(_('Patient ID should only contain letters, numbers, hyphens, and underscores.'))
        return patient_id
    
    def clean_date_of_birth(self) -> date | None:
        """Validate date of birth"""
        dob = self.cleaned_data.get('date_of_birth')
        if dob and dob > timezone.now().date():
            raise ValidationError(_('Date of birth cannot be in the future.'))
        if dob and (timezone.now().date() - dob).days > 365 * 150:  # 150 years max
            raise ValidationError(_('Date of birth seems too old.'))
        return dob
    
    def clean_date_of_xray(self) -> date | None:
        """Validate X-ray date"""
        xray_date = self.cleaned_data.get('date_of_xray')
        if xray_date and xray_date > timezone.now().date():
            raise ValidationError(_('X-ray date cannot be in the future.'))
        return xray_date
    
    def clean_technologist_first_name(self) -> str:
        return self._clean_name_field('technologist_first_name', _('Technologist first name'))
    
    def clean_technologist_last_name(self) -> str:
        return self._clean_name_field('technologist_last_name', _('Technologist last name'))


class PredictionHistoryFilterForm(forms.Form):
    gender = forms.ChoiceField(
        choices=[('', _('All')), ('male', _('Male')), ('female', _('Female')), ('other', _('Other'))],
        required=False
    )
    age_min = forms.IntegerField(required=False, min_value=0, max_value=150, 
                                label=_("Minimum Age"),
                                widget=forms.NumberInput(attrs={'placeholder': _('Min Age')}))
    age_max = forms.IntegerField(required=False, min_value=0, max_value=150, 
                                label=_("Maximum Age"),
                                widget=forms.NumberInput(attrs={'placeholder': _('Max Age')}))
    date_min = forms.DateField(required=False, 
                              label=_("From Date"),
                              widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}))
    date_max = forms.DateField(required=False, 
                              label=_("To Date"),
                              widget=forms.TextInput(attrs={'placeholder': 'YYYY-MM-DD'}))
    pathology = forms.ChoiceField(
        choices=[], 
        required=False,
        label=_("Pathology")
    )
    pathology_threshold = forms.FloatField(
        required=False, 
        min_value=0.0, 
        max_value=1.0,
        label=_("Minimum Probability"),
        initial=0.5,
        widget=forms.NumberInput(attrs={'step': '0.01'})
    )
    records_per_page = forms.ChoiceField(
        choices=[
            ('25', '25'),
            ('50', '50'),
            ('100', '100'),
            ('200', '200')
        ],
        initial='25',
        required=False,
        label=_("Records per page")
    )
    # Sorting fields
    sort_by = forms.ChoiceField(
        choices=[
            ('', _('Default (Prediction date)')),
            ('severity', _('Severity')),
            ('xray_date', _('X-ray date'))
        ],
        initial='',
        required=False,
        label=_("Sort by")
    )
    sort_order = forms.ChoiceField(
        choices=[
            ('desc', _('Highest to lowest')),
            ('asc', _('Lowest to highest'))
        ],
        initial='desc',
        required=False,
        label=_("Sort order")
    )
    
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        # Populate pathology choices dynamically from model constant
        # This keeps the filter form in sync with PATHOLOGY_FIELDS automatically.
        from .models import PATHOLOGY_FIELDS, RESULT_KEY_TO_FIELD
        
        # Build choices with localized labels
        pathology_choices = [('', _('All'))]
        
        # Create reverse mapping for display labels (same logic as in views)
        FIELD_TO_LABEL = {v: k for k, v in RESULT_KEY_TO_FIELD.items()}
        
        for field in PATHOLOGY_FIELDS:
            # Use original key as label if available, else capitalize field name
            raw_label = FIELD_TO_LABEL.get(field, field.replace('_', ' ').title())
            # Translate the label
            label = _(raw_label)
            pathology_choices.append((field, label))
            
        self.fields['pathology'].choices = pathology_choices
    
    def clean(self) -> dict:
        """Cross-field validation"""
        cleaned_data = super().clean()
        age_min = cleaned_data.get('age_min')
        age_max = cleaned_data.get('age_max')
        date_min = cleaned_data.get('date_min')
        date_max = cleaned_data.get('date_max')
        
        if age_min is not None and age_max is not None and age_min > age_max:
            raise ValidationError(_('Minimum age cannot be greater than maximum age.'))
        
        if date_min and date_max and date_min > date_max:
            raise ValidationError(_('Start date cannot be after end date.'))
            
        return cleaned_data


class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'preferred_theme', 
            'preferred_language', 
            'dashboard_view', 
            'email_notifications',
            'processing_complete_notification',
        ]
        widgets = {
            'preferred_theme': forms.Select(attrs={'class': 'form-select'}),
            'preferred_language': forms.Select(attrs={'class': 'form-select'}),
            'dashboard_view': forms.Select(attrs={'class': 'form-select'}),
            'email_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'processing_complete_notification': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }


class UserInfoForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }
        labels = {
            'first_name': _('First Name'),
            'last_name': _('Last Name'), 
            'email': _('Email Address'),
        }
    
    def clean_email(self) -> str | None:
        """Validate email uniqueness"""
        email = self.cleaned_data.get('email')
        if email and User.objects.filter(email=email).exclude(id=self.instance.id).exists():
            raise ValidationError(_('This email address is already in use.'))
        return email


class ChangePasswordForm(forms.Form):
    current_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        label=_("Current Password")
    )
    new_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        label=_("New Password"),
        min_length=8
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class': 'form-control'}),
        label=_("Confirm New Password")
    )
    
    def __init__(self, user: User, *args: object, **kwargs: object) -> None:
        self.user: User = user
        super().__init__(*args, **kwargs)
    
    def clean_current_password(self) -> str:
        """Validate current password"""
        current_password = self.cleaned_data.get('current_password')
        if not self.user.check_password(current_password):
            raise ValidationError(_('Current password is incorrect.'))
        return current_password
    
    def clean_new_password(self) -> str:
        """Validate new password strength"""
        new_password = self.cleaned_data.get('new_password') or ""
        
        try:
            from django.contrib.auth.password_validation import validate_password
            validate_password(new_password, self.user)
        except ValidationError as e:
            raise ValidationError(e.messages)
            
        return new_password
    
    def clean(self) -> dict:
        """Cross-field validation"""
        cleaned_data = super().clean()
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')
        
        if new_password and confirm_password and new_password != confirm_password:
            raise ValidationError(_('New passwords do not match.'))
            
        return cleaned_data