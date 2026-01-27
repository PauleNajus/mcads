from __future__ import annotations

import logging

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.utils import translation
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from xrayapp.forms import (
    ChangePasswordForm,
    UserInfoForm,
    UserProfileForm,
)
from xrayapp.models import UserProfile

logger = logging.getLogger(__name__)


@login_required
def account_settings(request: HttpRequest) -> HttpResponse:
    """View for managing user account settings"""
    # Ensure user has a profile
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    
    # Handle form submissions
    active_tab = request.GET.get('tab', 'profile')
    
    # Process profile info form
    if request.method == 'POST' and 'update_profile' in request.POST:
        user_form = UserInfoForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()
            messages.success(request, _('Your profile information has been updated successfully.'))
            active_tab = 'profile'
        else:
            messages.error(request, _('Please correct the errors below.'))
            active_tab = 'profile'
    else:
        user_form = UserInfoForm(instance=request.user)
    
    # Process settings form
    if request.method == 'POST' and 'update_settings' in request.POST:
        settings_form = UserProfileForm(request.POST, instance=profile)
        if settings_form.is_valid():
            settings_form.save()
            messages.success(request, _('Your preferences have been updated successfully. Please refresh the page to see theme changes.'))
            active_tab = 'settings'
        else:
            messages.error(request, _('Please correct the errors in your preferences.'))
            active_tab = 'settings'
    else:
        settings_form = UserProfileForm(instance=profile)
    
    # Process password change form
    if request.method == 'POST' and 'change_password' in request.POST:
        password_form = ChangePasswordForm(request.user, request.POST)
        if password_form.is_valid():
            user = request.user
            new_password = password_form.cleaned_data['new_password']
            user.set_password(new_password)
            user.save()
            # Update the session to prevent the user from being logged out
            update_session_auth_hash(request, user)
            messages.success(request, _('Your password has been changed successfully.'))
            active_tab = 'security'
        else:
            messages.error(request, _('Please correct the errors in your password form.'))
            active_tab = 'security'
    else:
        password_form = ChangePasswordForm(request.user)
    
    context = {
        'user_form': user_form,
        'settings_form': settings_form,
        'password_form': password_form,
        'active_tab': active_tab,
    }
    
    return render(request, 'xrayapp/account_settings.html', context)


def logout_confirmation(request: HttpRequest) -> HttpResponse:
    """Display a confirmation page before logging out the user."""
    return render(request, 'registration/logout.html')


@require_POST
def set_language(request: HttpRequest) -> HttpResponse:
    """Custom language switching view that integrates with user profile"""
    language = request.POST.get('language')
    
    if language and language in [lang[0] for lang in settings.LANGUAGES]:
        # Activate the language for this session
        translation.activate(language)
        
        # Get the redirect URL before creating the response
        redirect_url = request.META.get('HTTP_REFERER') or '/'
        if not url_has_allowed_host_and_scheme(
            url=redirect_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            redirect_url = '/'
        response = redirect(redirect_url)
        
        # Set language in session using Django's standard session key
        # Django's LocaleMiddleware looks for 'django_language' in the session
        request.session['django_language'] = language
        
        # Also set the language cookie for immediate effect
        response.set_cookie(
            settings.LANGUAGE_COOKIE_NAME,
            language,
            max_age=settings.LANGUAGE_COOKIE_AGE,
        )
        
        # Update user profile if user is authenticated
        if request.user.is_authenticated:
            try:
                profile, created = UserProfile.objects.get_or_create(user=request.user)
                profile.preferred_language = language
                profile.save()
            except Exception:
                logger.exception("Error updating user language preference for user=%s", request.user.pk)
        
        return response
    
    # If invalid language, redirect to home
    return redirect('/')
