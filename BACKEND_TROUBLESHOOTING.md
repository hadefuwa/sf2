# Backend Troubleshooting Guide

## ✅ Changes Pushed to Git

The following fixes have been committed and pushed:

1. **Commit cd3e481**: Fix PLC stability: remove duplicate locks and simplify retry logic
   - Fixed duplicate `with self.plc_lock` in write_db_bool
   - Removed duplicate write_vision_fault_bit method
   - Simplified all retry logic (no more aggressive 3-5 retry loops)
   - Reduced delays from 200-500ms to 20ms

2. **Commit d93f137**: Add better error handling for PLC status and DB123 read endpoints
   - Better error responses when PLC not connected
   - Improved /api/plc/status endpoint

---

## 🔧 Steps to Test on Raspberry Pi

### Step 1: SSH to Pi and Pull Changes

```bash
# On your Windows PC, open Git Bash or PowerShell:
ssh pi@rpi
# Password: 1

# Once connected to Pi:
cd ~/rpi-dobot
git pull
```

### Step 2: Stop Any Running Backend

```bash
# Find and kill any running backend process
pkill -f "python.*app.py"

# Or if you know the process ID:
ps aux | grep app.py
kill <PID>
```

### Step 3: Test Import (Check for Syntax Errors)

```bash
cd ~/rpi-dobot/pwa-dobot-plc/backend

# Test plc_client.py syntax
python3 -m py_compile plc_client.py && echo "✓ plc_client.py: No syntax errors"

# Test app.py syntax
python3 -m py_compile app.py && echo "✓ app.py: No syntax errors"
```

### Step 4: Start Backend with Debug Output

```bash
cd ~/rpi-dobot/pwa-dobot-plc/backend

# Start backend with full logging
python3 app.py 2>&1 | tee backend.log
```

**Watch for errors in the output!** Common issues:

- ❌ `ModuleNotFoundError`: Missing Python package
- ❌ `SyntaxError`: Code still has syntax issues
- ❌ `PermissionError`: Port 5000 already in use
- ❌ Snap7 errors: PLC connection issues (these are OK - backend should still start)

### Step 5: Test API Endpoints (From Another Terminal)

Open a **second SSH session** or use `curl` in background:

```bash
# Test if backend is responding
curl http://localhost:5000/api/config
# Should return: {"camera": {...}, "dobot": {...}, "plc": {...}}

# Test PLC status endpoint
curl http://localhost:5000/api/plc/status
# Should return: {"connected": false, "ip": "192.168.1.150", ...}

# Test DB123 read endpoint
curl http://localhost:5000/api/plc/db123/read
# Should return: {"plc_connected": false, ...} or actual tag values if PLC connected
```

---

## 🐛 Common Issues and Solutions

### Issue 1: "Backend server not responding"

**Symptoms**: Frontend shows "⚠️ Error: Backend server not responding"

**Causes**:
1. Backend not running at all
2. Backend crashed during startup
3. Backend running on wrong port
4. Firewall blocking connection

**Solution**:
```bash
# Check if backend is running
ps aux | grep app.py

# Check if port 5000 is in use
netstat -tlnp | grep 5000

# Check backend logs
cat ~/rpi-dobot/pwa-dobot-plc/backend/backend.log
```

### Issue 2: "Failed to fetch" Errors

**Symptoms**: Frontend shows "⚠️ Error: Failed to fetch"

**Causes**:
1. CORS issues (fixed in our code)
2. Network connectivity
3. Backend endpoint returning 500 error

**Solution**:
```bash
# Test endpoints directly with curl
curl -v http://localhost:5000/api/config

# Check for 500 errors in backend logs
grep "ERROR" backend.log
grep "500" backend.log
```

### Issue 3: Import Errors (snap7, flask, etc.)

**Symptoms**: `ModuleNotFoundError: No module named 'snap7'`

**Solution**:
```bash
# Install missing packages
pip3 install python-snap7 flask flask-socketio flask-cors opencv-python

# Or install all requirements
pip3 install -r ~/rpi-dobot/pwa-dobot-plc/backend/requirements.txt
```

### Issue 4: PLC Connection Errors (These are OK!)

**Symptoms**: Backend logs show PLC connection errors

**Note**: The backend **should still start** even if PLC is not connected. Snap7 errors are expected if:
- PLC is not powered on
- PLC IP is wrong
- Network cable unplugged

**Verify**: Check if backend APIs respond even with PLC disconnected:
```bash
curl http://localhost:5000/api/plc/status
# Should return: {"connected": false, "ip": "192.168.1.150", "last_error": "..."}
```

---

## 📊 Expected Startup Output

A **successful** backend startup should look like:

```
Counter images will be saved to: /home/pi/counter_images
No counter images found - starting fresh
Initialized counter tracker: starting from 0
INFO - Starting backend initialization...
INFO - Initializing PLC client...
WARNING - snap7 library not available or failed to import: ...
INFO - PLC client initialized (snap7 not available)
INFO - Camera service initialized
INFO - All clients initialized
 * Running on http://0.0.0.0:5000
```

**Key points**:
- ✅ Backend starts even if snap7 is not available
- ✅ "Running on http://0.0.0.0:5000" means server is up
- ⚠️ Snap7 warnings are OK - backend still works

---

## 🔍 Debugging Steps

### 1. Check Backend Status

```bash
# Is backend process running?
ps aux | grep -i "python.*app.py"

# Is port 5000 listening?
sudo netstat -tlnp | grep 5000

# Recent backend logs
tail -50 backend.log
```

### 2. Test Each Component

```bash
cd ~/rpi-dobot/pwa-dobot-plc/backend

# Test plc_client import
python3 -c "from plc_client import PLCClient; print('plc_client OK')"

# Test app.py import (don't run, just import)
python3 -c "import sys; sys.argv=['app.py']; from app import app; print('app.py OK')"
```

### 3. Run Backend in Foreground

```bash
# Stop any background processes
pkill -f app.py

# Run in foreground to see all output
cd ~/rpi-dobot/pwa-dobot-plc/backend
python3 app.py
```

Watch for the line: `* Running on http://0.0.0.0:5000`

---

## 📝 What to Report Back

If backend still doesn't work, please provide:

1. **Output of startup command**:
   ```bash
   cd ~/rpi-dobot/pwa-dobot-plc/backend
   python3 app.py 2>&1 | head -100
   ```

2. **Process status**:
   ```bash
   ps aux | grep app.py
   netstat -tlnp | grep 5000
   ```

3. **Test API call**:
   ```bash
   curl -v http://localhost:5000/api/config
   ```

4. **Error logs**:
   ```bash
   grep -i error backend.log | tail -20
   ```

---

## ✅ Quick Health Check

Run this one-liner to check everything:

```bash
cd ~/rpi-dobot/pwa-dobot-plc/backend && \
echo "=== Syntax Check ===" && \
python3 -m py_compile plc_client.py app.py && \
echo "✓ No syntax errors" && \
echo "" && \
echo "=== Backend Process ===" && \
ps aux | grep -v grep | grep app.py && \
echo "" && \
echo "=== Port Status ===" && \
netstat -tlnp 2>/dev/null | grep 5000 && \
echo "" && \
echo "=== API Test ===" && \
curl -s http://localhost:5000/api/config | python3 -m json.tool | head -20
```

This will show you at a glance if everything is working.
