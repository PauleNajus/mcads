from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.utils.html import format_html
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from .models import XRayImage, PredictionHistory, UserProfile, VisualizationResult


# Unregister the default User admin
admin.site.unregister(User)


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    """Enhanced User admin with role information"""
    list_display = ('username', 'email', 'first_name', 'last_name', 'get_role', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_superuser', 'is_active', 'date_joined', 'profile__role')
    search_fields = ('username', 'first_name', 'last_name', 'email')
    ordering = ('username',)
    
    def get_role(self, obj):
        """Get user role with color coding"""
        try:
            role = obj.profile.role
            colors = {
                'Administrator': '#dc3545',  # Red
                'Radiographer': '#28a745',   # Green
                'Technologist': '#007bff',   # Blue
                'Radiologist': '#6f42c1'     # Purple
            }
            color = colors.get(role, '#6c757d')
            return format_html(
                '<span style="color: {}; font-weight: bold;">{}</span>',
                color, role
            )
        except:
            return format_html('<span style="color: #dc3545;">{}</span>', _('No Profile'))
    get_role.short_description = _('Role')
    get_role.admin_order_field = 'profile__role'

    def get_fieldsets(self, request, obj=None):
        """Override fieldsets to include role management"""
        fieldsets = super().get_fieldsets(request, obj)
        if obj and hasattr(obj, 'profile'):
            # Add role information to personal info
            fieldsets = list(fieldsets)
            fieldsets[1] = (
                _('Personal info'),
                {'fields': ('first_name', 'last_name', 'email', 'get_role_info')}
            )
        return fieldsets

    def get_role_info(self, obj):
        """Display role information in user edit form"""
        try:
            profile = obj.profile
            return format_html(
                '<strong>{}:</strong> {} <br>'
                '<a href="{}" class="button">{}</a>',
                _('Current Role'),
                profile.role,
                reverse('admin:xrayapp_userprofile_change', args=[profile.pk]),
                _('Edit Profile & Role')
            )
        except:
            return format_html(
                '<span style="color: red;">{}</span><br>'
                '<a href="{}" class="button">{}</a>',
                _('No profile found'),
                reverse('admin:xrayapp_userprofile_add') + f'?user={obj.pk}',
                _('Create Profile')
            )
    get_role_info.short_description = _('Role Information')

    readonly_fields = (*UserAdmin.readonly_fields, 'get_role_info')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """Enhanced UserProfile admin with role management"""
    list_display = (
        'user', 'get_user_full_name', 'role', 'get_role_permissions', 
        'preferred_theme', 'email_notifications', 'created_at'
    )
    list_filter = ('role', 'preferred_theme', 'preferred_language', 'email_notifications', 'created_at')
    search_fields = ('user__username', 'user__email', 'user__first_name', 'user__last_name')
    ordering = ('user__username',)
    
    fieldsets = (
        (_('User'), {
            'fields': ('user', 'get_user_info')
        }),
        (_('Role & Permissions'), {
            'fields': ('role', 'get_permissions_display'),
            'description': _('Role determines what actions the user can perform in the system.')
        }),
        (_('Preferences'), {
            'fields': ('preferred_theme', 'preferred_language', 'dashboard_view'),
            'classes': ('collapse',)
        }),
        (_('Notifications'), {
            'fields': ('email_notifications', 'processing_complete_notification'),
            'classes': ('collapse',)
        }),
        (_('Security'), {
            'fields': ('two_factor_auth_enabled',),
            'classes': ('collapse',)
        }),
        (_('Timestamps'), {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )
    
    readonly_fields = ('get_user_info', 'get_permissions_display', 'created_at', 'updated_at')

    def get_user_full_name(self, obj):
        """Get user's full name"""
        return f"{obj.user.first_name} {obj.user.last_name}".strip() or obj.user.username
    get_user_full_name.short_description = _('Full Name')
    get_user_full_name.admin_order_field = 'user__first_name'

    def get_user_info(self, obj):
        """Display user information"""
        user = obj.user
        return format_html(
            '<strong>{}:</strong> {}<br>'
            '<strong>{}:</strong> {}<br>'
            '<strong>{}:</strong> {}<br>'
            '<strong>{}:</strong> {}<br>'
            '<a href="{}" class="button">{}</a>',
            _('Username'), user.username,
            _('Email'), user.email,
            _('Staff'), '✅' if user.is_staff else '❌',
            _('Superuser'), '✅' if user.is_superuser else '❌',
            reverse('admin:auth_user_change', args=[user.pk]),
            _('Edit User Details')
        )
    get_user_info.short_description = _('User Information')

    def get_role_permissions(self, obj):
        """Show key permissions for the role"""
        perms = []
        if obj.can_access_admin():
            perms.append(_('Admin'))
        if obj.can_upload_xrays():
            perms.append(_('Upload'))
        if obj.can_edit_predictions():
            perms.append(_('Edit'))
        if obj.can_delete_data():
            perms.append(_('Delete'))
        
        return ', '.join(perms) if perms else _('View Only')
    get_role_permissions.short_description = _('Key Permissions')

    def get_permissions_display(self, obj):
        """Display all permissions for the role"""
        permissions = [
            (_('Access Admin Panel'), obj.can_access_admin()),
            (_('Upload X-rays'), obj.can_upload_xrays()),
            (_('View All Patients'), obj.can_view_all_patients()),
            (_('Edit Predictions'), obj.can_edit_predictions()),
            (_('Delete Data'), obj.can_delete_data()),
            (_('Generate Interpretability'), obj.can_generate_interpretability()),
            (_('Manage Users'), obj.can_manage_users()),
        ]
        
        html = '<table style="width: 100%;">'
        for perm_name, has_perm in permissions:
            icon = '✅' if has_perm else '❌'
            html += f'<tr><td>{perm_name}</td><td>{icon}</td></tr>'
        html += '</table>'
        
        return format_html(html)
    get_permissions_display.short_description = _('Role Permissions')


@admin.register(XRayImage)
class XRayImageAdmin(admin.ModelAdmin):
    """Enhanced XRayImage admin with role-based features"""
    list_display = (
        'id', 'get_user_with_role', 'get_patient_display', 'patient_id', 
        'gender', 'uploaded_at', 'processing_status', 'get_severity_display'
    )
    list_filter = (
        'user__profile__role', 'processing_status', 'gender', 'uploaded_at', 
        'severity_level', 'has_gradcam', 'has_pli'
    )
    search_fields = ('user__username', 'patient_id', 'first_name', 'last_name', 'technologist_first_name', 'technologist_last_name')
    readonly_fields = (
        'uploaded_at', 'progress', 'image_format', 'image_size', 'image_resolution',
        'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 
        'effusion', 'emphysema', 'fibrosis', 'hernia', 'infiltration',
        'mass', 'nodule', 'pleural_thickening', 'pneumonia', 'pneumothorax',
        'fracture', 'lung_opacity', 'enlarged_cardiomediastinum', 'lung_lesion',
        'severity_level', 'has_gradcam', 'gradcam_visualization', 'gradcam_heatmap',
        'gradcam_overlay', 'gradcam_target_class', 'has_pli', 'pli_visualization',
        'pli_overlay_visualization', 'pli_saliency_map', 'pli_target_class'
    )
    
    fieldsets = (
        (_('User Information'), {
            'fields': ('user', 'get_user_role_info')
        }),
        (_('Patient Information'), {
            'fields': ('first_name', 'last_name', 'patient_id', 'gender', 'date_of_birth', 'date_of_xray', 'additional_info', 'technologist_first_name', 'technologist_last_name')
        }),
        (_('Image Processing'), {
            'fields': ('image', 'uploaded_at', 'processing_status', 'progress', 'image_format', 'image_size', 'image_resolution')
        }),
        (_('Pathology Predictions'), {
            'fields': (
                'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 
                'effusion', 'emphysema', 'fibrosis', 'hernia', 'infiltration',
                'mass', 'nodule', 'pleural_thickening', 'pneumonia', 'pneumothorax',
                'fracture', 'lung_opacity', 'enlarged_cardiomediastinum', 'lung_lesion',
                'severity_level'
            ),
            'classes': ('collapse',)
        }),
        (_('Interpretability Visualizations'), {
            'fields': (
                'has_gradcam', 'gradcam_visualization', 'gradcam_heatmap',
                'gradcam_overlay', 'gradcam_target_class', 'has_pli', 'pli_visualization',
                'pli_overlay_visualization', 'pli_saliency_map', 'pli_target_class'
            ),
            'classes': ('collapse',)
        })
    )

    def get_user_with_role(self, obj):
        """Display user with their role"""
        try:
            role = obj.user.profile.role
            colors = {
                'Administrator': '#dc3545',
                'Radiographer': '#28a745',
                'Technologist': '#007bff',
                'Radiologist': '#6f42c1'
            }
            color = colors.get(role, '#6c757d')
            return format_html(
                '{} <span style="color: {}; font-size: 0.8em;">({})</span>',
                obj.user.username, color, role
            )
        except:
            return obj.user.username
    get_user_with_role.short_description = _('User (Role)')
    get_user_with_role.admin_order_field = 'user__username'

    def get_user_role_info(self, obj):
        """Display user role information in detail view"""
        try:
            profile = obj.user.profile
            return format_html(
                '<strong>{}:</strong> {}<br>'
                '<strong>{}:</strong> {}<br>'
                '<strong>{}:</strong> {}',
                _('Role'), profile.role,
                _('Can Edit Predictions'), '✅' if profile.can_edit_predictions() else '❌',
                _('Can Delete'), '✅' if profile.can_delete_data() else '❌'
            )
        except:
            return _('No profile information available')
    get_user_role_info.short_description = _('User Role Info')

    def get_severity_display(self, obj):
        """Display severity with color coding"""
        level = obj.severity_level or obj.calculate_severity_level
        if level == 1:
            return format_html('<span style="color: #28a745;">{}</span>', _('Insignificant'))
        elif level == 2:
            return format_html('<span style="color: #ffc107;">{}</span>', _('Moderate'))
        elif level == 3:
            return format_html('<span style="color: #dc3545;">{}</span>', _('Significant'))
        return _('Unknown')
    get_severity_display.short_description = _('Severity')


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    """Enhanced PredictionHistory admin"""
    list_display = ('id', 'get_user_with_role', 'xray', 'created_at', 'model_used', 'get_severity_display')
    list_filter = ('user__profile__role', 'model_used', 'created_at', 'severity_level')
    search_fields = ('user__username', 'xray__patient_id', 'xray__first_name', 'xray__last_name')
    readonly_fields = (
        'user', 'xray', 'created_at', 'model_used',
        'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 
        'effusion', 'emphysema', 'fibrosis', 'hernia', 'infiltration',
        'mass', 'nodule', 'pleural_thickening', 'pneumonia', 'pneumothorax',
        'fracture', 'lung_opacity', 'enlarged_cardiomediastinum', 'lung_lesion',
        'severity_level'
    )

    def get_user_with_role(self, obj):
        """Display user with their role"""
        try:
            role = obj.user.profile.role
            colors = {
                'Administrator': '#dc3545',
                'Radiographer': '#28a745',
                'Technologist': '#007bff',
                'Radiologist': '#6f42c1'
            }
            color = colors.get(role, '#6c757d')
            return format_html(
                '{} <span style="color: {}; font-size: 0.8em;">({})</span>',
                obj.user.username, color, role
            )
        except:
            return obj.user.username
    get_user_with_role.short_description = _('User (Role)')
    get_user_with_role.admin_order_field = 'user__username'

    def get_severity_display(self, obj):
        """Display severity with color coding"""
        level = obj.severity_level or obj.calculate_severity_level
        if level == 1:
            return format_html('<span style="color: #28a745;">{}</span>', _('Insignificant'))
        elif level == 2:
            return format_html('<span style="color: #ffc107;">{}</span>', _('Moderate'))
        elif level == 3:
            return format_html('<span style="color: #dc3545;">{}</span>', _('Significant'))
        return _('Unknown')
    get_severity_display.short_description = _('Severity')


@admin.register(VisualizationResult)
class VisualizationResultAdmin(admin.ModelAdmin):
    """Admin interface for VisualizationResult model"""
    list_display = ('id', 'get_xray_info', 'visualization_type', 'target_pathology', 'model_used', 'created_at', 'get_preview')
    list_filter = ('visualization_type', 'model_used', 'created_at', 'target_pathology')
    search_fields = ('xray__patient_id', 'xray__first_name', 'xray__last_name', 'target_pathology', 'model_used')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'created_at', 'get_preview', 'get_file_links')
    
    fieldsets = (
        (_('Basic Information'), {
            'fields': ('id', 'xray', 'visualization_type', 'target_pathology', 'created_at')
        }),
        (_('File Paths'), {
            'fields': ('visualization_path', 'heatmap_path', 'overlay_path', 'saliency_path')
        }),
        (_('Metadata'), {
            'fields': ('model_used', 'threshold')
        }),
        (_('Preview'), {
            'fields': ('get_preview', 'get_file_links'),
            'classes': ('collapse',)
        })
    )
    
    def get_xray_info(self, obj):
        """Get X-ray information with link"""
        xray = obj.xray
        url = reverse('admin:xrayapp_xrayimage_change', args=[xray.pk])
        patient_info = f"{xray.first_name} {xray.last_name}".strip() if xray.first_name or xray.last_name else f"X-ray #{xray.id}"
        return format_html('<a href="{}">{}</a>', url, patient_info)
    get_xray_info.short_description = _('X-ray')
    get_xray_info.admin_order_field = 'xray'
    
    def get_preview(self, obj):
        """Get image preview"""
        if obj.visualization_url:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px;" />',
                obj.visualization_url
            )
        return _('No visualization available')
    get_preview.short_description = _('Preview')
    
    def get_file_links(self, obj):
        """Get links to all visualization files"""
        links = []
        
        if obj.visualization_url:
            links.append(f'<a href="{obj.visualization_url}" target="_blank">{_("Main Visualization")}</a>')
        if obj.heatmap_url:
            links.append(f'<a href="{obj.heatmap_url}" target="_blank">{_("Heatmap")}</a>')
        if obj.overlay_url:
            links.append(f'<a href="{obj.overlay_url}" target="_blank">{_("Overlay")}</a>')
        if obj.saliency_url:
            links.append(f'<a href="{obj.saliency_url}" target="_blank">{_("Saliency Map")}</a>')
        
        if links:
            return format_html(' | '.join(links))
        return _('No files available')
    get_file_links.short_description = _('File Links')


# Customize admin site headers
admin.site.site_header = _("MCADS Administration")
admin.site.site_title = _("MCADS Admin")
admin.site.index_title = _("Multi-label Chest Abnormality Detection System")
