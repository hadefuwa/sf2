#!/bin/bash
# Install a systemd service to run dual network setup at boot.
# Run: sudo bash install_dual_network_service.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo tee /etc/systemd/system/smart-factory-dual-network.service > /dev/null <<EOF
[Unit]
Description=Configure eth0 (PLC) + wlan0 (hotspot) for Smart Factory
After=network-online.target NetworkManager.service
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=$SCRIPT_DIR/setup_dual_network.sh
RemainAfterExit=yes
User=root

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable smart-factory-dual-network.service
echo "Service installed and enabled. It will run at boot."
echo "To run now: sudo systemctl start smart-factory-dual-network"
