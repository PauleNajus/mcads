#!/bin/bash

echo "🔧 Fixing PyTorch CPU backend issues and restarting MCADS services..."

# Reload systemd configuration
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Restart the MCADS service
echo "Restarting MCADS service..."
sudo systemctl restart mcads.service

# Check service status
echo "Checking service status..."
sudo systemctl status mcads.service --no-pager

echo "✅ Service restart completed!"

# Check if the service is running
if sudo systemctl is-active mcads.service >/dev/null 2>&1; then
    echo "🎉 MCADS service is running successfully!"
else
    echo "❌ MCADS service failed to start. Check logs with: journalctl -u mcads.service -f"
fi 