from __future__ import annotations

from django.http import HttpRequest, HttpResponse
from django.shortcuts import render


def handler400(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """400 Bad Request handler."""
    return render(request, 'errors/400.html', status=400)

def handler401(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """401 Unauthorized handler."""
    return render(request, 'errors/401.html', status=401)

def handler403(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """403 Forbidden handler."""
    return render(request, 'errors/403.html', status=403)

def handler404(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """404 Not Found handler."""
    return render(request, 'errors/404.html', status=404)

def handler408(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """408 Request Timeout handler."""
    return render(request, 'errors/408.html', status=408)

def handler429(request: HttpRequest, exception: Exception | None = None) -> HttpResponse:
    """429 Too Many Requests handler."""
    return render(request, 'errors/429.html', status=429)

def handler500(request: HttpRequest) -> HttpResponse:
    """500 Internal Server Error handler."""
    return render(request, 'errors/500.html', status=500)

def handler502(request: HttpRequest) -> HttpResponse:
    """502 Bad Gateway handler."""
    return render(request, 'errors/502.html', status=502)

def handler503(request: HttpRequest) -> HttpResponse:
    """503 Service Unavailable handler."""
    return render(request, 'errors/503.html', status=503)

def handler504(request: HttpRequest) -> HttpResponse:
    """504 Gateway Timeout handler."""
    return render(request, 'errors/504.html', status=504)

def terms_of_service(request: HttpRequest) -> HttpResponse:
    return render(request, 'xrayapp/terms_of_service.html')
