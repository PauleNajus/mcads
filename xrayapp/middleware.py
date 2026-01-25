from __future__ import annotations

import logging

from django.core.cache import cache
from django.http import HttpRequest, HttpResponse, HttpResponseForbidden
from django.template.loader import render_to_string
import hashlib
logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Rate limiting middleware to prevent brute force attacks"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes in seconds
        
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
        
        # Track failed login attempts
        if (request.path == '/accounts/login/' and 
            request.method == 'POST' and 
            response.status_code in [200, 302] and 
            not request.user.is_authenticated):
            self._record_failed_attempt(request)
            
        return response
    
    def _get_cache_key(self, request: HttpRequest) -> str:
        """Generate cache key based on IP address"""
        ip = self._get_client_ip(request)
        return f"login_attempts:{hashlib.sha256(ip.encode()).hexdigest()}"
    
    def _get_client_ip(self, request: HttpRequest) -> str:
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
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
        # Define URL patterns and required permissions
        self.protected_patterns = {
            '/secure-admin-mcads-2024/': 'can_access_admin',
            '/admin/': 'can_access_admin',
            '/prediction-history/delete': 'can_delete_data',
            '/interpretability/': 'can_generate_interpretability',
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
            # Get user profile
            profile = getattr(request.user, 'profile', None)
            if not profile:
                return False
            
            # Check against protected patterns
            for pattern, permission in self.protected_patterns.items():
                if request.path.startswith(pattern):
                    return getattr(profile, permission, lambda: False)()
            
            # Allow access if no specific protection is defined
            return True
            
        except Exception:
            # If there's any error, deny access and log for audit/debugging.
            logger.exception("RoleBasedAccessMiddleware error for path=%s", request.path)
            return False 