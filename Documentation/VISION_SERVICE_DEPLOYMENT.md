# Vision Service Deployment Guide

## Overview

The vision service has been split into a separate microservice to isolate YOLO crashes from the main Flask application. This follows the industrial-grade pattern recommended by your boss.

## Architecture

- **Main App** (`app.py`): Flask app on port 8080 - handles PLC, Dobot, camera, and API
- **Vision Service** (`vision_service.py`): Flask service on port 5001 - handles YOLO detection only

## Deployment Steps

### 1. Install Dependencies

On the Raspberry Pi:

```bash
cd ~/rpi-dobot/pwa-dobot-plc/backend
source venv/bin/activate
pip install requests
```

### 2. Deploy Code

```bash
cd ~/rpi-dobot
git pull
```

### 3. Update PM2 Configuration

The PM2 ecosystem config has been updated to include both services. Restart PM2:

```bash
cd ~/rpi-dobot/pwa-dobot-plc/deploy
pm2 delete all
pm2 start ecosystem.config.js
pm2 save
```

### 4. Verify Services

Check that both services are running:

```bash
pm2 list
```

You should see:
- `pwa-dobot-plc` (main app)
- `vision-service` (YOLO service)

### 5. Test Vision Service

Check health endpoint:

```bash
curl http://127.0.0.1:5001/health
```

Expected response:
```json
{
  "status": "healthy",
  "yolo_available": true,
  "model_loaded": true,
  "model_path": "/home/pi/counter_detector.pt"
}
```

## How It Works

### Main App (`app.py`)
- When YOLO detection is needed, calls `call_vision_service()`
- Sends frame as base64-encoded JPEG to vision service
- Receives detection results back
- If vision service is down, returns graceful error (doesn't crash)

### Vision Service (`vision_service.py`)
- Loads YOLO model once at startup
- Exposes `/detect` endpoint for detection requests
- Exposes `/health` endpoint for status checks
- Runs in separate process - crashes don't affect main app

### PM2 Management
- Both services managed by PM2
- If vision service crashes, PM2 automatically restarts it
- Main app continues running even if vision service is down

## Error Handling

The main app handles vision service failures gracefully:

- **Connection Error**: Returns `{"error": "Vision service unavailable"}`
- **Timeout**: Returns `{"error": "Vision service timeout"}` (5 second timeout)
- **Service Error**: Returns error message from vision service

The frontend can display "Vision temporarily unavailable" messages.

## Monitoring

### Check Logs

Main app logs:
```bash
pm2 logs pwa-dobot-plc
```

Vision service logs:
```bash
pm2 logs vision-service
```

### Check Status

```bash
pm2 status
```

## Benefits

1. **Isolation**: YOLO crashes don't kill the main app
2. **Automatic Recovery**: PM2 restarts vision service on crash
3. **Graceful Degradation**: Main app continues serving other endpoints
4. **Easy Debugging**: Separate logs for each service
5. **Scalability**: Can scale vision service independently if needed

## Troubleshooting

### Vision Service Won't Start

1. Check if YOLO model exists:
   ```bash
   ls -la ~/counter_detector.pt
   ```

2. Check vision service logs:
   ```bash
   pm2 logs vision-service --lines 50
   ```

3. Test manually:
   ```bash
   cd ~/rpi-dobot/pwa-dobot-plc/backend
   source venv/bin/activate
   python vision_service.py
   ```

### Main App Can't Connect to Vision Service

1. Check if vision service is running:
   ```bash
   pm2 list
   curl http://127.0.0.1:5001/health
   ```

2. Check firewall/port binding:
   ```bash
   netstat -tlnp | grep 5001
   ```

3. Check main app logs for connection errors:
   ```bash
   pm2 logs pwa-dobot-plc | grep vision
   ```

## Configuration

### Environment Variables

- `VISION_SERVICE_URL`: URL of vision service (default: `http://127.0.0.1:5001`)
- `VISION_PORT`: Port for vision service (default: `5001`)
- `PORT`: Port for main app (default: `8080`)

Set in PM2 ecosystem config or environment.

