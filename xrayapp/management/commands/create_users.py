from __future__ import annotations

import os
import secrets

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.db import transaction
from xrayapp.models import UserProfile


class Command(BaseCommand):
    help = 'Creates default users for the system with roles'

    def handle(self, *args, **kwargs):
        """Create a small set of default users for bootstrapping.

        Security:
        - Never hardcode passwords in source control.
        - Passwords are read from environment variables if provided; otherwise a
          strong random password is generated and printed once to stdout.
        """

        users_data = [
            {
                'username': 'admin',
                'email': 'admin@mcads.casa',
                'first_name': 'Adminfirst',
                'last_name': 'Adminlast',
                'role': 'Administrator',
                'hospital': 'VULSK',
                'is_staff': True,
                'is_superuser': True
            },
            {
                'username': 'paubun',
                'email': 'paubun@mcads.casa',
                'first_name': 'Paulius',
                'last_name': 'Bundza',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'justri',
                'email': 'justri@mcads.casa',
                'first_name': 'Justas',
                'last_name': 'Trinkūnas',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'rolber',
                'email': 'rolber@mcads.casa',
                'first_name': 'Rolandas',
                'last_name': 'Bėrontas',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'technologist',
                'email': 'technologist@mcads.casa',
                'first_name': 'Tech',
                'last_name': 'Nologist',
                'role': 'Technologist',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'radiologist',
                'email': 'radiologist@mcads.casa',
                'first_name': 'Radio',
                'last_name': 'Logist',
                'role': 'Radiologist',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'guest',
                'email': 'guestuser@mcads.casa',
                'first_name': 'Guest',
                'last_name': 'User',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'gabryl',
                'email': 'gabryl@mcads.casa',
                'first_name': 'Gabija',
                'last_name': 'Ryliškytė',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            },
            {
                'username': 'augbun',
                'email': 'augbun@mcads.casa',
                'first_name': 'Augustė',
                'last_name': 'Bundzaitė',
                'role': 'Radiographer',
                'hospital': 'VULSK',
                'is_staff': False,
                'is_superuser': False
            }
        ]

        default_password = os.environ.get("MCADS_DEFAULT_USER_PASSWORD", "").strip() or None

        with transaction.atomic():
            for user_data in users_data:
                username = user_data.pop('username')
                role = user_data.pop('role')
                hospital = user_data.pop('hospital')

                # Password resolution order:
                # 1) Per-user env var: MCADS_USER_PASSWORD_<USERNAME_UPPER>
                # 2) Shared env var: MCADS_DEFAULT_USER_PASSWORD
                # 3) Generated strong password (printed once)
                password_env_key = f"MCADS_USER_PASSWORD_{username.upper()}"
                password = os.environ.get(password_env_key, "").strip() or default_password
                generated = False
                if not password:
                    password = secrets.token_urlsafe(18)
                    generated = True
                
                # Check if user already exists
                if User.objects.filter(username=username).exists():
                    self.stdout.write(self.style.WARNING(f'User {username} already exists. Skipping.'))
                    continue
                
                # Create the user
                user = User.objects.create_user(username=username, password=password, **user_data)

                if generated:
                    self.stdout.write(
                        self.style.WARNING(
                            f'Generated password for {username}: {password} (set {password_env_key} to control this)'
                        )
                    )
                
                # Create user profile with role and hospital
                profile, created = UserProfile.objects.get_or_create(
                    user=user,
                    defaults={'role': role, 'hospital': hospital}
                )
                if not created:
                    profile.role = role
                    profile.hospital = hospital
                    profile.save()
                
                self.stdout.write(
                    self.style.SUCCESS(f'User {username} created successfully with role: {role}, hospital: {hospital}')
                )
        
        self.stdout.write(self.style.SUCCESS('All users created successfully.')) 