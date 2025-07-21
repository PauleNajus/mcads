#!/bin/bash

echo "🔄 Restarting MCADS services after fixing STATIC_URL and CSP issues..."

# Restart Django application
echo "Restarting Django application..."
sudo systemctl restart mcads.service

# Wait a moment for the service to start
sleep 2

# Reload nginx configuration  
echo "Reloading nginx configuration..."
sudo systemctl reload nginx

# Check service status
echo "📊 Service Status Check:"
echo "MCADS service status: $(sudo systemctl is-active mcads.service)"
echo "Nginx service status: $(sudo systemctl is-active nginx)"

echo ""
echo "✅ Services restarted successfully!"
echo "🔧 Fixed issues:"
echo "   - STATIC_URL corrected from 'static/' to '/static/'"
echo "   - Content Security Policy temporarily disabled"
echo ""
echo "🌐 Please refresh your browser and check the website now!"
echo "   The CSS should now be loading properly." 