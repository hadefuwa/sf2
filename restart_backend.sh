#!/bin/bash
cd ~/rpi-dobot
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
git pull
cd pwa-dobot-plc/backend
pip3 install flask-socketio 2>/dev/null
pkill -f app.py
sleep 2
PORT=8080 python3 app.py
