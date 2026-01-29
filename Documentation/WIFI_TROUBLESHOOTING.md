# WiFi Access Point Troubleshooting

## Quick Check Commands

### SSH to Pi and Run Diagnostic
```bash
ssh pi@rpi
cd ~/rpi-dobot/scripts
chmod +x check_wifi_ap.sh
./check_wifi_ap.sh
```

### Fix WiFi Access Point
```bash
cd ~/rpi-dobot/scripts
chmod +x fix_wifi_ap.sh
sudo ./fix_wifi_ap.sh
```

## Manual Commands to Check Status

### Check hostapd service
```bash
sudo systemctl status hostapd
```

### Check dnsmasq service
```bash
sudo systemctl status dnsmasq
```

### Check wlan0 interface
```bash
ip addr show wlan0
```

### Restart WiFi services
```bash
sudo systemctl restart hostapd
sudo systemctl restart dnsmasq
```

### View hostapd logs
```bash
sudo journalctl -u hostapd -n 50
```

### View dnsmasq logs
```bash
sudo journalctl -u dnsmasq -n 50
```

## Common Issues

### WiFi network not showing up
1. Check if hostapd is running: `sudo systemctl status hostapd`
2. Check logs: `sudo journalctl -u hostapd -n 50`
3. Restart: `sudo systemctl restart hostapd`

### Can't connect to WiFi
1. Check dnsmasq is running: `sudo systemctl status dnsmasq`
2. Check IP address: `ip addr show wlan0` (should be 192.168.4.1)
3. Restart: `sudo systemctl restart dnsmasq`

### Services won't start
1. Check config files exist:
   - `/etc/hostapd/hostapd.conf`
   - `/etc/dnsmasq.conf`
2. Run setup script again: `sudo ./setup_wifi_access_point.sh`
3. Reboot: `sudo reboot`

## Expected WiFi Details
- **SSID**: SmartFactory
- **Password**: matrix123
- **Pi IP**: 192.168.4.1
- **Web Server**: http://192.168.4.1:8080

