#!/bin/bash
# Quick Pi Setup Script - Run this on the Raspberry Pi

echo "=========================================="
echo "🤖 Starting Dobot PWA Installation"
echo "=========================================="

# Step 1: Update system
echo ""
echo "📦 Step 1: Updating system packages..."
sudo apt-get update

# Step 2: Install dependencies
echo ""
echo "📦 Step 2: Installing dependencies..."
sudo apt-get install -y build-essential git python3 python3-pip python3-venv python3-dev nodejs npm

# Step 3: Install PM2
echo ""
echo "📦 Step 3: Installing PM2..."
sudo npm install -g pm2

# Step 4: Clone or update repo
echo ""
echo "📦 Step 4: Setting up repository..."
if [ ! -d ~/rpi-dobot ]; then
    cd ~
    git clone https://github.com/hadefuwa/rpi-dobot.git
else
    cd ~/rpi-dobot
    git pull
fi

# Step 5: Create Python virtual environment
echo ""
echo "🐍 Step 5: Setting up Python environment..."
cd ~/rpi-dobot/pwa-dobot-plc/backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install ultralytics

# Step 6: Create directories
echo ""
echo "📁 Step 6: Creating directories..."
mkdir -p ~/logs
mkdir -p ~/counter_images

# Step 7: Setup PM2
echo ""
echo "⚙️ Step 7: Configuring PM2 auto-start..."
cd ~/rpi-dobot/pwa-dobot-plc/deploy
pm2 delete all 2>/dev/null || true
pm2 start ecosystem.config.js
pm2 save
pm2 startup systemd -u pi --hp /home/pi | grep "sudo" | bash || echo "Run: sudo env PATH=\$PATH:/usr/bin pm2 startup systemd -u pi --hp /home/pi"

# Step 8: Check status
echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
pm2 status
echo ""
echo "🌐 Access the app at: http://$(hostname -I | awk '{print $1}'):8080"
echo ""





