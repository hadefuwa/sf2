# 🏭 Smart Factory

A comprehensive smart factory automation system featuring Dobot Magician robot control with PLC integration, real-time monitoring, and a modern web-based interface. Perfect for Industry 4.0 applications with automatic alarm clearing and seamless PLC communication.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [What This Project Does](#-what-this-project-does)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Features](#-key-features)
- [Documentation](#-documentation)
- [Testing](#-testing)
- [Troubleshooting](#-troubleshooting)
- [Deployment](#-deployment)
- [Support](#-support)

---

## 🚀 Quick Start

### On Raspberry Pi:

```bash
# Navigate to project directory
cd ~/smart-factory

# Pull latest changes (if using git)
git pull origin main

# Go to backend directory
cd pwa-dobot-plc/backend

# Activate virtual environment (if using one)
source venv/bin/activate

# Run the application
python3 app.py
```

**Access the web interface:** Open your browser and visit `http://your-pi-ip-address:8080`

---

## 🎯 What This Project Does

This project allows you to:

- **Control a Dobot Magician robot arm** through a web interface
- **Integrate with Siemens S7-1200 PLC** for automated control
- **Monitor robot position** in real-time
- **Clear robot alarms automatically** (this was a key fix!)
- **Control robot movements** manually or via PLC commands
- **Use as a Progressive Web App (PWA)** - install it on your phone or desktop

---

## 📁 Project Structure

```
smart-factory/
├── pwa-dobot-plc/              # Main application (core working code)
├── WinCC_Camera_Control/       # Siemens WinCC Unified Custom Web Control
│   ├── backend/                 # Flask server and robot control
│   │   ├── app.py              # Main Flask application
│   │   ├── dobot_client.py     # Dobot robot control (FIXED)
│   │   ├── plc_client.py       # PLC communication
│   │   ├── camera_service.py   # Camera functionality
│   │   └── config.json         # Configuration file
│   ├── frontend/               # Web interface files
│   │   ├── index.html         # Main control page
│   │   ├── robot-arm.html     # Robot control interface
│   │   └── ...                # Other pages
│   └── deploy/                # Deployment scripts
│       └── ecosystem.config.js # PM2 configuration
├── docs/                       # All documentation
│   ├── guides/                # Setup and usage guides
│   ├── solutions/             # Problem resolution docs
│   └── api/                   # API documentation
├── scripts/                    # Executable scripts
│   ├── deployment/            # Deployment scripts
│   └── testing/               # Test scripts
├── lib/                       # External libraries
│   └── DobotAPI/             # Official Dobot API files
├── tests/                     # Test files
│   ├── pydobot/              # pydobot library tests
│   └── official_api/         # Official API tests
└── README.md                  # This file
```

---

## 💻 Installation

### Prerequisites

- Raspberry Pi (3 or 4 recommended)
- Dobot Magician robot connected via USB
- Siemens S7-1200 PLC (optional, for PLC integration)
- Python 3.7 or higher
- Internet connection (for initial setup)

### Step-by-Step Installation

#### 1. Clone or Download the Project

```bash
cd ~
git clone https://github.com/hadefuwa/rpi-dobot.git
cd rpi-dobot
```

#### 2. Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install Python and build tools
sudo apt-get install -y python3-pip python3-venv build-essential

# Install Snap7 library for PLC communication (if using PLC)
cd ~
wget https://sourceforge.net/projects/snap7/files/1.4.2/snap7-full-1.4.2.tar.gz
tar -zxvf snap7-full-1.4.2.tar.gz
cd snap7-full-1.4.2/build/unix
make -f arm_v7_linux.mk
sudo cp ../bin/arm_v7-linux/libsnap7.so /usr/lib/
sudo ldconfig
```

#### 3. Set Up Python Virtual Environment

```bash
cd ~/smart-factory/pwa-dobot-plc/backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python packages
pip install -r requirements.txt
```

#### 4. Configure USB Permissions (for Dobot)

```bash
# Add your user to dialout group (allows USB access)
sudo usermod -a -G dialout $USER

# Log out and back in, or run:
newgrp dialout

# Find your Dobot device
ls -la /dev/ttyACM*
```

#### 5. Configure Settings

Edit `pwa-dobot-plc/backend/config.json`:

```json
{
  "dobot": {
    "port": "/dev/ttyACM0",
    "baudrate": 115200
  },
  "plc": {
    "ip": "192.168.1.150",
    "rack": 0,
    "slot": 1
  },
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  }
}
```

**Important:** Update the `dobot.port` to match your device (usually `/dev/ttyACM0` or `/dev/ttyACM1`)

#### 6. Test the Installation

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the application
python3 app.py
```

You should see:
```
INFO - Starting Flask server on 0.0.0.0:8080
INFO - PLC client initialized
INFO - Dobot client initialized
```

---

## 🎮 Usage

### Starting the Application

```bash
cd ~/smart-factory/pwa-dobot-plc/backend
source venv/bin/activate
python3 app.py
```

### Using the Web Interface

1. **Open your browser** and go to `http://your-pi-ip:8080`
2. **Check connections** - Green indicators show PLC and Dobot are connected
3. **Monitor data** - See real-time robot position and PLC status
4. **Control the robot**:
   - 🏠 **Home** - Send robot to home position
   - ▶️ **Move to Target** - Move robot to PLC target coordinates
   - 🛑 **Emergency Stop** - Immediately stop all movement
   - Manual position control via web interface

### Installing as PWA (Progressive Web App)

**On Mobile (iOS/Android):**
1. Open the app in Safari (iOS) or Chrome (Android)
2. Tap "Share" → "Add to Home Screen"
3. Launch like a native app!

**On Desktop:**
1. Open in Chrome browser
2. Click the install icon in the address bar
3. Use as a standalone app!

---

## ✨ Key Features

- ✅ **Dobot Movement Control** - Full robot arm control via web interface
- ✅ **Automatic Alarm Clearing** - Robot alarms are cleared automatically on startup (key fix!)
- ✅ **PLC Integration** - Siemens S7-1200 communication for automated control
- ✅ **Real-time Monitoring** - Live position and status updates via WebSocket
- ✅ **Settings Management** - Web-based configuration interface
- ✅ **Emergency Stop** - Safety controls for immediate shutdown
- ✅ **Progressive Web App** - Install and use offline
- ✅ **Camera Support** - Optional camera integration for vision systems
- ✅ **WinCC HMI Support** - Custom Web Control to view camera streams on Siemens Unified Panels

---

## 📚 Documentation

### Quick Start Guides

- **[Quick Start Guide](docs/guides/QUICK_START_ON_PI.md)** - Get started quickly on Raspberry Pi
- **[Deployment Guide](docs/guides/DEPLOY_TO_PI.md)** - Full deployment instructions
- **[PLC Setup Guide](docs/guides/PLC_DB1_Setup_Guide.md)** - Setting up PLC communication
- **[PLC Robot Control](docs/guides/PLC_Robot_Control_Guide.md)** - Using PLC to control robot
- **[PLC Settings Guide](docs/guides/PLC_Settings_Guide.md)** - Configuring PLC settings

### Problem Solutions

- **[Solution Summary](docs/solutions/SOLUTION_SUMMARY.md)** - **Main fix documentation** (read this first!)
- **[Bugfix Summary](docs/solutions/BUGFIX_SUMMARY.md)** - Summary of bugs fixed
- **[Implementation Summary](docs/solutions/IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[Complete Documentation](docs/solutions/DOBOT_FIX_COMPLETE_DOCUMENTATION.md)** - Full technical documentation

### API Documentation

- **[API Migration Plan](docs/api/DOBOT_API_MIGRATION_PLAN.md)** - API migration information
- **[Official API Migration Guide](docs/api/OFFICIAL_API_MIGRATION_GUIDE.md)** - Official API guide
- **[API Quick Reference](docs/api/OFFICIAL_API_QUICK_REFERENCE.md)** - Quick API reference
- **[API Commands Reference](docs/api/DOBOT_API_COMMANDS_REFERENCE.md)** - Complete command reference

### Documentation Index

For a complete overview, see **[Documentation Index](DOCUMENTATION_INDEX.md)**

---

## 🧪 Testing

### Main Test (Alarm Clearing Fix)

Test the improved Dobot client with alarm clearing:

```bash
python3 scripts/testing/test_improved_client.py
```

### pydobot Library Tests

Test basic Dobot functionality:

```bash
python3 tests/pydobot/test_dobot_simple.py
python3 tests/pydobot/test_dobot_speed.py
python3 tests/pydobot/test_dobot_home.py
python3 tests/pydobot/test_dobot_ptp_params.py
python3 tests/pydobot/test_dobot_go_lock.py
```

### Official API Tests

Test official Dobot API (if using):

```bash
python3 tests/official_api/test_official_api_connection.py
python3 tests/official_api/test_official_api_movement.py
python3 tests/official_api/test_official_api_peripherals.py
```

### Alarm Clearing Test

Test alarm clearing functionality:

```bash
python3 scripts/testing/test_alarm_clear.py
```

---

## 🔧 Troubleshooting

### Dobot Not Connecting

**Problem:** Robot doesn't connect or shows as disconnected

**Solutions:**

```bash
# Check USB device exists
ls -la /dev/ttyACM*

# Check permissions (should include 'dialout')
groups

# If not in dialout group:
sudo usermod -a -G dialout $USER
newgrp dialout

# Try different device path in config.json
# Common paths: /dev/ttyACM0, /dev/ttyACM1, /dev/ttyUSB0
```

**Check config.json:**
- Make sure `dobot.port` matches your actual device
- Verify `baudrate` is set to `115200`

### Robot Not Moving

**Problem:** Robot connects but doesn't move when commanded

**Solution:** This was the main issue fixed! The robot needs alarms cleared on startup. The fixed code (`dobot_client.py`) now does this automatically. Make sure you're using the updated version.

**Verify fix is applied:**
- Check that `dobot_client.py` includes alarm clearing in the `connect()` method
- See [Solution Summary](docs/solutions/SOLUTION_SUMMARY.md) for details

### PLC Not Connecting

**Problem:** PLC shows as disconnected

**Solutions:**

```bash
# Test network connection
ping 192.168.1.150

# Check PLC IP in config.json
# Make sure IP matches your PLC's actual IP address

# Verify Snap7 library is installed
ldconfig -p | grep snap7
```

**Check config.json:**
- Verify `plc.ip` matches your PLC's IP address
- Check `plc.rack` and `plc.slot` are correct (usually 0 and 1)

### Port Already in Use

**Problem:** Error "Address already in use" on port 8080

**Solutions:**

```bash
# Find process using port 8080
sudo lsof -ti:8080

# Kill the process
sudo lsof -ti:8080 | xargs -r sudo kill -9

# Or change port in config.json
# Set "server.port" to a different number (e.g., 8081)
```

### Import Errors

**Problem:** Python import errors when running app.py

**Solutions:**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# Check Python version (needs 3.7+)
python3 --version
```

### Permission Denied Errors

**Problem:** Permission errors accessing USB device

**Solutions:**

```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER

# Log out and back in, or:
newgrp dialout

# Check device permissions
ls -la /dev/ttyACM*
```

---

## 🚀 Deployment

### Quick Deployment Script

Use the automated deployment script:

```bash
./scripts/deployment/setup.sh
```

### Full Deployment with PM2 (Recommended)

PM2 keeps the application running automatically and restarts it if it crashes:

```bash
# Install PM2 globally
npm install -g pm2

# Run full deployment script
./scripts/deployment/FINAL_DEPLOYMENT.sh

# Or manually:
cd ~/smart-factory/pwa-dobot-plc
pm2 start deploy/ecosystem.config.js
pm2 save
pm2 startup  # Follow instructions to enable auto-start on boot
```

### Manual PM2 Setup

```bash
# Navigate to project
cd ~/smart-factory/pwa-dobot-plc

# Start with PM2
pm2 start deploy/ecosystem.config.js

# Save PM2 configuration
pm2 save

# Set PM2 to start on boot
pm2 startup
# Run the command it gives you (with sudo)

# Check status
pm2 status
pm2 logs pwa-dobot-plc
```

### Official API Setup (Optional)

If you want to use the official Dobot API instead of pydobot:

```bash
./scripts/deployment/setup_official_dobot_api.sh
```

---

## 📋 PLC Memory Map

### DB1 (Data Block)

- **DBD0-3**: Target X position (REAL)
- **DBD4-7**: Target Y position (REAL)
- **DBD8-11**: Target Z position (REAL)

### Merkers (M Memory)

- **M0.0**: Start movement
- **M0.1**: Stop
- **M0.2**: Home
- **M0.3**: Emergency stop
- **M0.4**: Suction cup control
- **M0.5**: Ready status (read-only)
- **M0.6**: Busy status (read-only)
- **M0.7**: Error status (read-only)

---

## 🎯 Key Solution

The main issue (Dobot not moving) was solved by adding **automatic alarm clearing** to the initialization sequence. When the robot starts up, it may have alarms from previous sessions. These alarms prevent movement commands from working. The fix clears all alarms automatically when connecting.

**See [Solution Summary](docs/solutions/SOLUTION_SUMMARY.md) for complete details.**

---

## 📞 Support

### Quick Help

- **Connection issues:** See [Troubleshooting](#-troubleshooting) section above
- **Code examples:** Check [Solution Summary](docs/solutions/SOLUTION_SUMMARY.md)
- **Deployment:** Use `./scripts/deployment/FINAL_DEPLOYMENT.sh`

### Documentation Resources

- **[Documentation Index](DOCUMENTATION_INDEX.md)** - Complete guide to all documentation
- **[Solution Summary](docs/solutions/SOLUTION_SUMMARY.md)** - Main fix documentation
- **[Quick Start Guide](docs/guides/QUICK_START_ON_PI.md)** - Setup instructions

### Common Questions

**Q: Why isn't my robot moving?**  
A: Make sure alarms are being cleared. Check that you're using the updated `dobot_client.py` with alarm clearing enabled.

**Q: How do I find my Dobot USB device?**  
A: Run `ls -la /dev/ttyACM*` and check which device appears when you plug/unplug the robot.

**Q: Can I use this without a PLC?**  
A: Yes! The web interface allows manual control without PLC integration.

**Q: How do I update the code?**  
A: Pull latest changes with `git pull origin main` and restart the application.

---

## 📊 Project Status

✅ **WORKING** - Dobot movement issue resolved with alarm clearing  
✅ **TESTED** - All core functionality verified  
✅ **DEPLOYED** - Production-ready on Raspberry Pi  
✅ **ORGANIZED** - Clean project structure for maintainability  
✅ **DOCUMENTED** - Comprehensive documentation available

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🙏 Credits

- **Flask & Flask-SocketIO** - Web framework and real-time communication
- **python-snap7** - PLC communication library
- **pydobot** - Dobot robot control library
- **OpenCV** - Camera and vision support

---

**Last Updated:** 2025-01-27  
**Version:** v4.1  
**Status:** Production Ready ✅
