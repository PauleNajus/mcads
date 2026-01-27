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
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
from .models import XRayImage, UserProfile
from django.contrib.auth.models import User

logger = logging.getLogger(__name__)

def _looks_like_dicom(filename: str | None, header: bytes) -> bool:
    """Heuristic DICOM detection.

    We intentionally keep this lightweight:
    - Standard DICOM files have a 128-byte preamble followed by b"DICM".
    - Many uploads also use the conventional .dcm extension.
    """
    if len(header) >= 132 and header[128:132] == b"DICM":
        return True
    if filename and filename.lower().endswith((".dcm", ".dicom")):
        return True
    return False


def _dicom_to_png(uploaded: UploadedFile) -> ContentFile:
    """Convert a DICOM upload into a PNG `ContentFile`.

    Why convert?
    - The rest of MCADS expects an image that PIL/skimage can read.
    - Templates use `<img src="{{ image_url }}">`, which won't render a `.dcm`.

    We keep the original name stem, but always return a `.png` file.
    """
    try:
        import numpy as np
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        from PIL import Image
    except Exception as exc:  # pragma: no cover (dependency/runtime)
        raise ValidationError(_("DICOM support is not available on this server.")) from exc

    uploaded.seek(0)
    try:
        ds = pydicom.dcmread(uploaded, force=True)
    except Exception as exc:
        raise ValidationError(_("Invalid DICOM file.")) from exc

    try:
        arr = np.asarray(ds.pixel_array)
    except Exception as exc:
        # This can happen for compressed pixel data without extra decoders.
        raise ValidationError(_("Unable to read DICOM pixel data.")) from exc

    # Handle common shapes:
    # - (H, W) grayscale
    # - (frames, H, W) multi-frame → take first frame
    # - (H, W, 3/4) RGB/RGBA → take first channel (rare for X-rays)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        else:
            arr = arr[0]
    elif arr.ndim > 3:
        # Fall back to the last 2 dimensions as an image plane.
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    arr = arr.astype("float32", copy=False)

    # Apply rescale slope/intercept when present.
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    # Apply VOI LUT / windowing when available.
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # MONOCHROME1 means the grayscale is inverted.
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = float(arr.max()) - arr

    # Robust normalization to 8-bit for downstream pipelines.
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        raise ValidationError(_("DICOM pixel data is empty or invalid."))

    finite_vals = arr[finite_mask]

    # Percentile computation on very large arrays can be noticeably slow.
    # A light, deterministic subsample preserves robustness while improving latency.
    sample = finite_vals
    if sample.size > 512_000:
        step = max(1, sample.size // 512_000)
        sample = sample[::step]

    low, high = np.percentile(sample, [1, 99]).astype("float32")
    if not (high > low):
        low = float(finite_vals.min())
        high = float(finite_vals.max())
    if not (high > low):
        high = low + 1.0

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    arr8 = (arr * 255.0).clip(0, 255).astype("uint8")

    img = Image.fromarray(arr8, mode="L")
    out = BytesIO()
    # `optimize=True` can add seconds for large PNGs; prefer fast compression.
    img.save(out, format="PNG", compress_level=1)

    # Keep name stable and safe.
    original_name = (uploaded.name or "dicom").rsplit("/", 1)[-1]
    stem = original_name.rsplit(".", 1)[0] or "dicom"
    return ContentFile(out.getvalue(), name=f"{stem}.png")


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
        user = kwargs.pop('user', None)
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

        is_dicom = _looks_like_dicom(getattr(image, "name", None), header)
        
        # Check MIME type for security if magic is available
        if MAGIC_AVAILABLE and not is_dicom:
            file_mime = None
            try:
                file_mime = magic.from_buffer(header, mime=True)
            except Exception as exc:
                # If magic fails, rely on Django's validation (do not block uploads).
                logger.warning("python-magic MIME detection failed: %s", exc)
            finally:
                image.seek(0)  # Always reset file pointer after read

                allowed_mimes = [
                    'image/jpeg', 'image/jpg', 'image/png', 
                    'image/bmp', 'image/tiff', 'image/x-ms-bmp'
                ]
                
                if file_mime and file_mime not in allowed_mimes:
                    raise ValidationError(_('Invalid file type. Only image files are allowed.'))

        # If this is a DICOM file, convert it to PNG before saving to the model.
        if is_dicom:
            # Persist the original upload "format" separately from the stored file format.
            # We convert DICOM → PNG for processing/display, but we still want user-facing
            # metadata to say "DICOM".
            self._mcads_source_format = "DICOM"
            return _dicom_to_png(image)

        # For non-DICOM uploads, we must validate that the content is a real image.
        # (We use FileField to accept .dcm, so we need to reintroduce ImageField-like verification.)
        try:
            from PIL import Image, UnidentifiedImageError
        except Exception as exc:  # pragma: no cover (dependency/runtime)
            raise ValidationError(_("Image validation is not available on this server.")) from exc

        # Quick extension sanity check (optional; content checks are the source of truth).
        # This prevents accidental uploads like "report.pdf" when MIME detection is unavailable.
        name = (getattr(image, "name", "") or "").lower()
        if "." in name:
            ext = name.rsplit(".", 1)[-1]
            allowed_exts = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
            if ext not in allowed_exts:
                raise ValidationError(_('Invalid file type. Only image files are allowed.'))

        # Verify the image can be opened by Pillow.
        try:
            image.seek(0)
            with Image.open(image) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValidationError(_("Invalid image file.")) from exc
        finally:
            image.seek(0)

        return image
    
    def clean_first_name(self) -> str:
        """Sanitize first name input"""
        first_name = self.cleaned_data.get('first_name', '').strip()
        if first_name and not first_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('First name should only contain letters, spaces, hyphens, and apostrophes.'))
        return first_name
    
    def clean_last_name(self) -> str:
        """Sanitize last name input"""
        last_name = self.cleaned_data.get('last_name', '').strip()
        if last_name and not last_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('Last name should only contain letters, spaces, hyphens, and apostrophes.'))
        return last_name
    
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
        """Sanitize technologist first name input"""
        technologist_first_name = self.cleaned_data.get('technologist_first_name', '').strip()
        if technologist_first_name and not technologist_first_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('Technologist first name should only contain letters, spaces, hyphens, and apostrophes.'))
        return technologist_first_name
    
    def clean_technologist_last_name(self) -> str:
        """Sanitize technologist last name input"""
        technologist_last_name = self.cleaned_data.get('technologist_last_name', '').strip()
        if technologist_last_name and not technologist_last_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('Technologist last name should only contain letters, spaces, hyphens, and apostrophes.'))
        return technologist_last_name


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
        # Populate pathology choices dynamically
        pathology_choices = [
            ('', _('All')),
            ('atelectasis', _('Atelectasis')),
            ('cardiomegaly', _('Cardiomegaly')),
            ('consolidation', _('Consolidation')),
            ('edema', _('Edema')),
            ('effusion', _('Effusion')),
            ('emphysema', _('Emphysema')),
            ('fibrosis', _('Fibrosis')),
            ('hernia', _('Hernia')),
            ('infiltration', _('Infiltration')),
            ('mass', _('Mass')),
            ('nodule', _('Nodule')),
            ('pleural_thickening', _('Pleural Thickening')),
            ('pneumonia', _('Pneumonia')),
            ('pneumothorax', _('Pneumothorax')),
            ('fracture', _('Fracture')),
            ('lung_opacity', _('Lung Opacity')),
            ('enlarged_cardiomediastinum', _('Enlarged Cardiomediastinum')),
            ('lung_lesion', _('Lung Lesion'))
        ]
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
            'two_factor_auth_enabled'
        ]
        widgets = {
            'preferred_theme': forms.Select(attrs={'class': 'form-select'}),
            'preferred_language': forms.Select(attrs={'class': 'form-select'}),
            'dashboard_view': forms.Select(attrs={'class': 'form-select'}),
            'email_notifications': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'processing_complete_notification': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'two_factor_auth_enabled': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
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
        
        # Basic password validation
        if len(new_password) < 8:
            raise ValidationError(_('Password must be at least 8 characters long.'))
        
        # Check for at least one digit
        if not any(char.isdigit() for char in new_password):
            raise ValidationError(_('Password must contain at least one digit.'))
        
        # Check for at least one uppercase letter
        if not any(char.isupper() for char in new_password):
            raise ValidationError(_('Password must contain at least one uppercase letter.'))
        
        # Check for at least one lowercase letter
        if not any(char.islower() for char in new_password):
            raise ValidationError(_('Password must contain at least one lowercase letter.'))
            
        return new_password
    
    def clean(self) -> dict:
        """Cross-field validation"""
        cleaned_data = super().clean()
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')
        
        if new_password and confirm_password and new_password != confirm_password:
            raise ValidationError(_('New passwords do not match.'))
            
        return cleaned_data 