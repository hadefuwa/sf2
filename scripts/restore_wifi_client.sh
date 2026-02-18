#!/bin/bash
# Restore wlan0 to WiFi client mode (connect to your router).
# Run this if the Pi became unreachable after hotspot setup.
# You may need to run this over Ethernet or serial console.

echo "=========================================="
echo "Restore WiFi Client Mode"
echo "=========================================="
echo ""

echo "Step 1: Stopping hostapd and dnsmasq..."
sudo systemctl stop hostapd dnsmasq 2>/dev/null || true
sleep 1

echo "Step 2: Re-enabling NetworkManager control of wlan0..."
sudo nmcli device set wlan0 managed yes
sleep 2

echo "Step 3: Bringing wlan0 back up..."
sudo ip link set wlan0 down
sleep 1
sudo ip link set wlan0 up
sleep 3

echo ""
echo "Pi should reconnect to your WiFi. Try: ping rpi"
echo ""
