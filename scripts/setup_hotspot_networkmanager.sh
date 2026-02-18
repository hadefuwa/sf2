#!/bin/bash
# Configure Raspberry Pi as WiFi Access Point when NetworkManager is present.
# Run this if the standard setup left wlan0 managed by NetworkManager.
# Requires: hostapd, dnsmasq already installed (from setup_wifi_access_point.sh)

set -e

echo "=========================================="
echo "Hotspot Setup (NetworkManager fix)"
echo "=========================================="
echo ""

# Step 1: Tell NetworkManager to not manage wlan0
echo "Step 1: Unmanaging wlan0 from NetworkManager..."
sudo nmcli connection down "$(nmcli -t -f NAME,DEVICE connection show --active | grep wlan0 | cut -d: -f1)" 2>/dev/null || true
sudo nmcli device set wlan0 managed no
sleep 2

# Step 2: Stop hostapd/dnsmasq, bring down wlan0
echo "Step 2: Resetting wlan0..."
sudo systemctl stop hostapd dnsmasq 2>/dev/null || true
sudo ip link set wlan0 down
sleep 1

# Step 3: Assign static IP via ip (dhcpcd may not apply when NM had control)
echo "Step 3: Assigning 192.168.4.1 to wlan0..."
sudo ip addr flush dev wlan0
sudo ip addr add 192.168.4.1/24 dev wlan0
sudo ip link set wlan0 up
sleep 2

# Step 4: Start dnsmasq then hostapd
echo "Step 4: Starting dnsmasq and hostapd..."
sudo systemctl start dnsmasq
sleep 2
sudo systemctl start hostapd
sleep 3

# Step 5: Restart app services (they may have stopped when network changed)
echo "Step 5: Restarting Smart Factory app services..."
sudo systemctl restart smart-factory 2>/dev/null || true
sudo systemctl restart vision 2>/dev/null || true
sleep 3

# Step 6: Verify
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
ip addr show wlan0 | grep "inet "
echo ""
echo "Connect to WiFi 'SmartFactory' (password: matrix123)"
echo "Then open: http://192.168.4.1:8080"
echo ""
echo "If app still doesn't load, run: sudo systemctl restart smart-factory vision"
echo ""
