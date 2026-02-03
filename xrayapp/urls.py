from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('health/', views.health, name='health'),
    path('', views.home, name='home'),
    path('xray/<int:pk>/', views.xray_results, name='xray_results'),
    path('xray/<int:pk>/preview.png', views.dicom_preview, name='dicom_preview'),
    path('progress/<int:pk>/', views.check_progress, name='check_progress'),
    path('interpretability/<int:pk>/generate/', views.generate_interpretability, name='generate_interpretability'),
    path('segmentation/<int:pk>/generate/', views.generate_segmentation, name='generate_segmentation'),
    path('visualization/<int:pk>/delete/', views.delete_visualization, name='delete_visualization'),
    path('prediction-history/', views.prediction_history, name='prediction_history'),
    path('prediction-history/<int:pk>/delete/', views.delete_prediction_history, name='delete_prediction_history'),
    path('prediction-history/<int:pk>/edit/', views.edit_prediction_history, name='edit_prediction_history'),
    path('prediction-history/delete-all/', views.delete_all_prediction_history, name='delete_all_prediction_history'),
    path('prediction-history/<int:pk>/toggle-save/', views.toggle_save_record, name='toggle_save_record'),
    path('saved-records/', views.saved_records, name='saved_records'),
    path('account/settings/', views.account_settings, name='account_settings'),
    path('accounts/logout-confirmation/', views.logout_confirmation, name='logout_confirmation'),
    path('set-language/', views.set_language, name='set_language'),
    path('terms/', views.terms_of_service, name='terms_of_service'),
]

# Add media URL patterns for development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 