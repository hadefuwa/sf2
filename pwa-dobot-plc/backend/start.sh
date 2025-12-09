#!/bin/bash
# Startup script for PWA Dobot-PLC Backend
# Ensures backend runs on port 8080

cd "$(dirname "$0")"

# Set port to 8080
export PORT=8080

# Kill any existing instances
pkill -f "python.*app.py"
sleep 1

# Start backend
echo "Starting backend on port $PORT..."
python3 app.py 2>&1 | tee backend.log
