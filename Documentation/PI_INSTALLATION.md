# Raspberry Pi Installation Guide

## Quick Install (Copy and paste these commands on your Pi)

```bash
# 1. Clone the repository
cd ~
git clone https://github.com/hadefuwa/rpi-dobot.git
cd rpi-dobot

# 2. Run the setup script
chmod +x scripts/deployment/setup_vision_system.sh
./scripts/deployment/setup_vision_system.sh
```

## Manual Installation Steps

If you prefer to install manually:

```bash
# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install dependencies
sudo apt-get install -y build-essential git python3 python3-pip python3-venv python3-dev \
    libatlas-base-dev libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    v4l-utils nodejs npm

# 3. Install PM2
sudo npm install -g pm2

# 4. Clone repository
cd ~
git clone https://github.com/hadefuwa/rpi-dobot.git
cd rpi-dobot/pwa-dobot-plc/backend

# 5. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 6. Install Python packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install ultralytics

# 7. Create directories
mkdir -p ~/logs
mkdir -p ~/counter_images

# 8. Start with PM2
cd ~/rpi-dobot/pwa-dobot-plc/deploy
pm2 start ecosystem.config.js
pm2 save
pm2 startup

# 9. Follow the PM2 startup command it outputs (usually requires sudo)
```

## After Installation

1. **Place your YOLO model**: Copy `counter_detector.pt` to `~/counter_detector.pt`
2. **Access the app**: Open `http://YOUR_PI_IP:8080/vision-system.html` in your browser
3. **Check status**: Run `pm2 status` to see if services are running
4. **View logs**: Run `pm2 logs` to see application logs

## Troubleshooting

- **Camera not working**: Check with `ls -la /dev/video*`
- **Services not starting**: Check logs with `pm2 logs`
- **Port already in use**: Check with `sudo netstat -tlnp | grep 8080`

