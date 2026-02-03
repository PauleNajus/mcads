from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

from django.conf import settings
from django.core.cache import cache
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden, JsonResponse
from django.http.response import HttpResponseRedirectBase
from django.template.loader import render_to_string
from django.utils.translation import gettext_lazy as _
logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware to prevent brute force attacks"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.max_attempts = getattr(settings, 'RATELIMIT_MAX_ATTEMPTS', 5)
        self.lockout_duration = getattr(settings, 'RATELIMIT_LOCKOUT_DURATION', 300)
        
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Only apply rate limiting to login attempts
        if request.path == '/accounts/login/' and request.method == 'POST':
            if not self._check_rate_limit(request):
                response = HttpResponse(
                    "Too many login attempts. Please try again later.",
                    status=429,
                    content_type="text/plain"
                )
                return response
        
        response = self.get_response(request)

        # Successful login should reset the failure counter.
        #
        # Django's default login view returns a redirect (3xx) on success.
        # Without clearing, users who previously failed a few times can get
        # unexpectedly locked out again after a single typo.
        if (
            request.path == '/accounts/login/'
            and request.method == 'POST'
            and 300 <= int(response.status_code) < 400
        ):
            cache.delete(self._get_cache_key(request))
        
        # Track failed login attempts.
        #
        # Django's login view returns:
        # - 200 for invalid credentials (form re-render)
        # - 302 for successful login (redirect)
        #
        # We must NOT count 302 as a failure; `request.user` may still be
        # unauthenticated in middleware even though the session was updated.
        if (request.path == '/accounts/login/' and
            request.method == 'POST' and
            response.status_code == 200 and
            not request.user.is_authenticated):
            self._record_failed_attempt(request)
            
        return response
    
    def _get_cache_key(self, request: HttpRequest) -> str:
        """Generate cache key based on IP address"""
        ip = self._get_client_ip(request)
        return f"login_attempts:{hashlib.sha256(ip.encode()).hexdigest()}"
    
    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address, preferring X-Real-IP set by Nginx"""
        x_real_ip = request.META.get('HTTP_X_REAL_IP')
        if x_real_ip:
            return x_real_ip.strip()
            
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            # Proxy headers may include multiple IPs; use the left-most one.
            # Split by comma and strip whitespace from the first element
            return x_forwarded_for.split(',')[0].strip()
        return request.META.get('REMOTE_ADDR', '0.0.0.0')
    
    def _check_rate_limit(self, request: HttpRequest) -> bool:
        """Check if request is within rate limit"""
        cache_key = self._get_cache_key(request)
        attempts = cache.get(cache_key, 0)
        return attempts < self.max_attempts
    
    def _record_failed_attempt(self, request: HttpRequest) -> None:
        """Record a failed login attempt"""
        cache_key = self._get_cache_key(request)
        attempts = cache.get(cache_key, 0)
        cache.set(cache_key, attempts + 1, self.lockout_duration)


class RoleBasedAccessMiddleware:
    """Middleware to enforce role-based access control"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        # Use setting for admin URL to avoid duplication
        admin_url = '/' + getattr(settings, 'ADMIN_URL', 'admin/').lstrip('/')
        self.protected_patterns = {
            admin_url: 'can_access_admin',
            '/admin/': 'can_access_admin', # Keep legacy just in case
            '/interpretability/': 'can_generate_interpretability',
            '/segmentation/': 'can_generate_interpretability',
        }
        
    def __call__(self, request: HttpRequest) -> HttpResponse:
        # Skip check for public paths
        public_paths = [
            '/accounts/login/',
            '/accounts/logout/',
            '/favicon.ico',
            '/static/',
            '/media/',
        ]
        
        is_public = any(request.path.startswith(path) for path in public_paths)
        
        # Only check authenticated users for non-public paths
        if not is_public and request.user.is_authenticated:
            if not self._check_access(request):
                # Return 403 Forbidden with custom error page
                return HttpResponseForbidden(
                    render_to_string('errors/403.html', {'request': request})
                )
        
        response = self.get_response(request)
        return response
    
    def _check_access(self, request: HttpRequest) -> bool:
        """Check if user has access to the requested resource"""
        try:
            # Get or create user profile.
            #
            # Some flows in this codebase create `UserProfile` lazily (e.g. on first upload).
            # Middleware must not lock those users out with a 403 before the view can run.
            try:
                profile = request.user.profile  # type: ignore[attr-defined]
            except ObjectDoesNotExist:
                from .models import UserProfile, UserRole  # local import to avoid import-time side effects

                default_role = UserRole.ADMINISTRATOR if request.user.is_superuser else UserRole.RADIOGRAPHER
                profile, _ = UserProfile.objects.get_or_create(user=request.user, defaults={'role': default_role})

            permission = self._required_permission_for_path(request.path)
            if permission:
                return getattr(profile, permission, lambda: False)()
            
            # Allow access if no specific protection is defined
            return True
            
        except Exception:
            # If there's any error, deny access and log for audit/debugging.
            logger.exception("RoleBasedAccessMiddleware error for path=%s", request.path)
            return False 

    def _required_permission_for_path(self, path: str) -> str | None:
        """Return the permission method name required for a path (or None).

        Keep this intentionally simple and fast: pure string matching.
        """
        # Dynamic prediction-history actions
        if path.startswith('/prediction-history/'):
            if path.endswith('/delete/') or path.startswith('/prediction-history/delete-all/'):
                return 'can_delete_data'
            if path.endswith('/edit/'):
                return 'can_edit_predictions'

        # Dynamic visualization deletion
        if path.startswith('/visualization/') and path.endswith('/delete/'):
            return 'can_delete_data'

        # Static prefix patterns
        for pattern, permission in self.protected_patterns.items():
            if path.startswith(pattern):
                return permission

        return None


class AjaxRedirectToJsonMiddleware:
    """Return JSON instead of redirects for XHR/fetch requests.

    Browsers follow redirects automatically for `fetch()`. If a view protected by
    `@login_required` redirects to the login page, the frontend ends up with a
    200 HTML response (login form) and JSON parsing fails with a confusing
    "Response is not JSON".

    This middleware converts redirects for requests that *expect JSON* into a
    JSON payload with a non-2xx status so frontend code can handle it explicitly.
    """

    def __init__(self, get_response):
        self.get_response = get_response

        # LOGIN_URL can be a named URL ("login") or an absolute/relative path.
        # Resolve it once to a concrete path for reliable comparisons.
        from django.shortcuts import resolve_url

        login_url = getattr(settings, "LOGIN_URL", None)
        try:
            resolved_login_url = resolve_url(login_url) if login_url else "/accounts/login/"
        except Exception:
            resolved_login_url = str(login_url or "/accounts/login/")
        self._login_path_prefix = urlparse(str(resolved_login_url)).path or "/accounts/login/"

    @staticmethod
    def _wants_json(request: HttpRequest) -> bool:
        """Heuristic: treat requests as AJAX when they expect JSON back."""
        accept = request.headers.get("Accept", "")
        return (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or "application/json" in accept
            # Our frontend sets this header on form uploads; it helps in cases
            # where proxies strip non-standard headers.
            or bool(request.headers.get("X-CSRFToken"))
        )

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)

        if not self._wants_json(request):
            return response

        # Only convert actual redirects (3xx with a Location header).
        if not isinstance(response, HttpResponseRedirectBase):
            return response

        location = response.get("Location")
        if not location:
            return response

        # Normalize paths for comparison.
        try:
            location_path = urlparse(str(location)).path or ""
        except Exception:
            location_path = str(location)

        if location_path.startswith(self._login_path_prefix):
            return JsonResponse(
                {"error": _("Authentication required"), "login_url": location},
                status=401,
            )

        # Generic redirect for JSON callers (e.g., HTTPS enforcement).
        return JsonResponse(
            {"error": _("Unexpected redirect"), "redirect_url": location},
            status=409,
        )