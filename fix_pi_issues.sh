#!/bin/bash
# Quick fix script for Pi issues

cd ~/rpi-dobot

echo "=== Fixing git merge conflict ==="
# Remove Python cache files that are blocking merge
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Now try git pull again
git pull

echo ""
echo "=== Installing missing Python modules ==="
cd pwa-dobot-plc/backend

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Using virtual environment"
else
    echo "No venv found, using system Python"
fi

# Install required modules
pip3 install flask-socketio flask-cors python-snap7 opencv-python

echo ""
echo "=== Starting backend on port 8080 ==="
pkill -f app.py
sleep 2

PORT=8080 python3 app.py
