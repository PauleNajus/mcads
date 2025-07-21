# Emergency CSS fix patch
import os
from pathlib import Path

# Ensure static files work with minimal configuration
STATIC_URL = '/static/'
STATIC_ROOT = '/opt/mcads/app/staticfiles'

# Disable all potentially problematic middleware temporarily
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Simplified static files
STATICFILES_DIRS = [
    '/opt/mcads/app/static',
]

# Temporarily set DEBUG=True for static file serving
DEBUG = True

print("Emergency settings applied!")
