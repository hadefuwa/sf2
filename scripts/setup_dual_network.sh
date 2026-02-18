#!/bin/bash
# Configure Pi for BOTH:
#   - eth0: 192.168.7.5 (PLC network)
#   - wlan0: 192.168.4.1 (SmartFactory WiFi hotspot)
# Run once, or at boot via systemd.

set -e

echo "=========================================="
echo "Dual Network Setup: PLC + Hotspot"
echo "=========================================="
echo "  eth0  -> 192.168.7.5 (PLC)"
echo "  wlan0 -> 192.168.4.1 (SmartFactory AP)"
echo ""

# ========== PART 1: Ensure eth0 has 192.168.7.5 ==========
echo "Step 1: Configuring eth0 for PLC (192.168.7.5)..."
CURRENT_ETH0=$(ip -4 addr show eth0 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1)

if [ "$CURRENT_ETH0" = "192.168.7.5" ]; then
    echo "  eth0 already has 192.168.7.5"
else
    # Use NetworkManager if available
    if systemctl is-active --quiet NetworkManager 2>/dev/null; then
        CONN=$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | grep ":eth0" | cut -d: -f1)
        if [ -n "$CONN" ]; then
            sudo nmcli connection modify "$CONN" ipv4.addresses 192.168.7.5/24 ipv4.method manual ipv4.gateway "" ipv4.dns "" 2>/dev/null || true
            sudo nmcli connection down "$CONN" 2>/dev/null || true
            sleep 1
            sudo nmcli connection up "$CONN" 2>/dev/null || true
            sleep 2
        else
            sudo ip addr flush dev eth0 2>/dev/null || true
            sudo ip addr add 192.168.7.5/24 dev eth0
            sudo ip link set eth0 up
        fi
    else
        sudo ip addr flush dev eth0 2>/dev/null || true
        sudo ip addr add 192.168.7.5/24 dev eth0
        sudo ip link set eth0 up
    fi
    echo "  eth0 set to 192.168.7.5"
fi
sleep 1

# ========== PART 2: Configure wlan0 as hotspot ==========
echo ""
echo "Step 2: Configuring wlan0 as SmartFactory hotspot..."

# Unmanage wlan0 from NetworkManager so hostapd can use it
sudo nmcli connection down "$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | grep wlan0 | cut -d: -f1)" 2>/dev/null || true
sudo nmcli device set wlan0 managed no 2>/dev/null || true
sleep 1

# Stop, reset, assign IP
sudo systemctl stop hostapd dnsmasq 2>/dev/null || true
sudo ip link set wlan0 down 2>/dev/null || true
sleep 1
sudo ip addr flush dev wlan0 2>/dev/null || true
sudo ip addr add 192.168.4.1/24 dev wlan0
sudo ip link set wlan0 up
sleep 2

# Start hotspot services
echo "  Starting hostapd and dnsmasq..."
sudo systemctl start dnsmasq
sleep 2
sudo systemctl start hostapd
sleep 3

# ========== PART 3: Restart app ==========
echo ""
echo "Step 3: Restarting Smart Factory app..."
sudo systemctl restart smart-factory 2>/dev/null || true
sudo systemctl restart vision 2>/dev/null || true
sleep 3

# ========== Verify ==========
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "eth0:  $(ip -4 addr show eth0 2>/dev/null | grep 'inet ' | awk '{print $2}' || echo 'not configured')"
echo "wlan0: $(ip -4 addr show wlan0 2>/dev/null | grep 'inet ' | awk '{print $2}' || echo 'not configured')"
echo ""
echo "  PLC:       Connect via eth0 (192.168.7.5)"
echo "  Hotspot:   Connect phone to 'SmartFactory' (password: matrix123)"
echo "  App:       http://192.168.4.1:8080"
echo ""
