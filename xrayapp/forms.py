from django import forms
from django.utils import timezone
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator
from django.utils.translation import gettext_lazy as _
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
from .models import XRayImage, UserProfile
from django.contrib.auth.models import User

class XRayUploadForm(forms.ModelForm):
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
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        
        # Auto-populate technologist fields with current user's information
        if user and user.is_authenticated:
            # Only set initial values if the fields are not already populated
            if not self.data.get('technologist_first_name') and not self.initial.get('technologist_first_name'):
                self.fields['technologist_first_name'].initial = user.first_name or ''
            if not self.data.get('technologist_last_name') and not self.initial.get('technologist_last_name'):
                self.fields['technologist_last_name'].initial = user.last_name or ''
        
        # Add file validation
        self.fields['image'].validators.append(
            FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])
        )
        # Add input length limits for security
        self.fields['first_name'].widget.attrs.update({'maxlength': 100})
        self.fields['last_name'].widget.attrs.update({'maxlength': 100})
        self.fields['patient_id'].widget.attrs.update({'maxlength': 100})
        self.fields['technologist_first_name'].widget.attrs.update({'maxlength': 100})
        self.fields['technologist_last_name'].widget.attrs.update({'maxlength': 100})
    
    def clean_image(self):
        """Validate uploaded image file"""
        image = self.cleaned_data.get('image')
        if not image:
            return image
            
        # Check file size (max 10MB)
        if image.size > 10 * 1024 * 1024:
            raise ValidationError(_('Image file too large. Maximum size is 10MB.'))
        
        # Check MIME type for security if magic is available
        if MAGIC_AVAILABLE:
            try:
                file_mime = magic.from_buffer(image.read(), mime=True)
                image.seek(0)  # Reset file pointer
                
                allowed_mimes = [
                    'image/jpeg', 'image/jpg', 'image/png', 
                    'image/bmp', 'image/tiff', 'image/x-ms-bmp'
                ]
                
                if file_mime not in allowed_mimes:
                    raise ValidationError(_('Invalid file type. Only image files are allowed.'))
            except Exception:
                # If magic fails, rely on Django's validation
                pass
            
        return image
    
    def clean_first_name(self):
        """Sanitize first name input"""
        first_name = self.cleaned_data.get('first_name', '').strip()
        if first_name and not first_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('First name should only contain letters, spaces, hyphens, and apostrophes.'))
        return first_name
    
    def clean_last_name(self):
        """Sanitize last name input"""
        last_name = self.cleaned_data.get('last_name', '').strip()
        if last_name and not last_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('Last name should only contain letters, spaces, hyphens, and apostrophes.'))
        return last_name
    
    def clean_patient_id(self):
        """Validate patient ID format"""
        patient_id = self.cleaned_data.get('patient_id', '').strip()
        if patient_id and not patient_id.replace('-', '').replace('_', '').isalnum():
            raise ValidationError(_('Patient ID should only contain letters, numbers, hyphens, and underscores.'))
        return patient_id
    
    def clean_date_of_birth(self):
        """Validate date of birth"""
        dob = self.cleaned_data.get('date_of_birth')
        if dob and dob > timezone.now().date():
            raise ValidationError(_('Date of birth cannot be in the future.'))
        if dob and (timezone.now().date() - dob).days > 365 * 150:  # 150 years max
            raise ValidationError(_('Date of birth seems too old.'))
        return dob
    
    def clean_date_of_xray(self):
        """Validate X-ray date"""
        xray_date = self.cleaned_data.get('date_of_xray')
        if xray_date and xray_date > timezone.now().date():
            raise ValidationError(_('X-ray date cannot be in the future.'))
        return xray_date
    
    def clean_technologist_first_name(self):
        """Sanitize technologist first name input"""
        technologist_first_name = self.cleaned_data.get('technologist_first_name', '').strip()
        if technologist_first_name and not technologist_first_name.replace(' ', '').replace('-', '').replace("'", '').isalpha():
            raise ValidationError(_('Technologist first name should only contain letters, spaces, hyphens, and apostrophes.'))
        return technologist_first_name
    
    def clean_technologist_last_name(self):
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
    
    def __init__(self, *args, **kwargs):
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
    
    def clean(self):
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
    
    def clean_email(self):
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
    
    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)
    
    def clean_current_password(self):
        """Validate current password"""
        current_password = self.cleaned_data.get('current_password')
        if not self.user.check_password(current_password):
            raise ValidationError(_('Current password is incorrect.'))
        return current_password
    
    def clean_new_password(self):
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
    
    def clean(self):
        """Cross-field validation"""
        cleaned_data = super().clean()
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')
        
        if new_password and confirm_password and new_password != confirm_password:
            raise ValidationError(_('New passwords do not match.'))
            
        return cleaned_data 