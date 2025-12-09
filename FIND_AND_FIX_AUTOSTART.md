# Find and Fix Auto-Startup Script

Run these commands on the Pi (via SSH) to find and fix the auto-startup:

## Step 1: Find the Auto-Startup Script

```bash
# Check systemd services
sudo systemctl list-units --type=service | grep -E "dobot|plc|app"
systemctl --user list-units --type=service | grep -E "dobot|plc|app"

# Check cron
crontab -l | grep app.py

# Check rc.local
cat /etc/rc.local | grep app.py

# Check for common autostart locations
ls -la ~/.config/autostart/
cat ~/.bashrc | grep app.py
cat ~/.profile | grep app.py

# Search everywhere
sudo grep -r "app.py" /etc/systemd/ /etc/cron* ~/.config/ 2>/dev/null | grep -v ".pyc"
```

---

## Step 2: Once Found, Fix It

### If it's a **systemd service**:

```bash
# Find the service file
sudo systemctl status <service-name>

# Edit it
sudo nano /etc/systemd/system/<service-name>.service

# Add PORT=8080 to the Environment line:
Environment="PORT=8080"

# Or in the ExecStart line:
ExecStart=/usr/bin/env PORT=8080 python3 /path/to/app.py

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart <service-name>
```

### If it's a **cron job**:

```bash
# Edit crontab
crontab -e

# Change the line to include PORT=8080:
@reboot cd /home/pi/rpi-dobot/pwa-dobot-plc/backend && PORT=8080 python3 app.py

# Save and exit
```

### If it's **rc.local**:

```bash
# Edit rc.local
sudo nano /etc/rc.local

# Change the line to:
cd /home/pi/rpi-dobot/pwa-dobot-plc/backend && PORT=8080 python3 app.py &

# Save and exit
```

---

## Step 3: Quick Fix For Now

While you search for the auto-startup, just restart the backend manually:

```bash
# Stop current backend
pkill -f app.py

# Start on port 8080
cd ~/rpi-dobot/pwa-dobot-plc/backend
PORT=8080 python3 app.py
```

Then refresh your browser (Ctrl + Shift + R) and it should work!

---

## What to Report Back

Once you find it, let me know:
1. **Where is the auto-startup script?** (systemd/cron/rc.local/other)
2. **What does the line look like?**

I'll help you fix it permanently!
