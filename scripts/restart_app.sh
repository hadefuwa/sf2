#!/bin/bash
# Restart Smart Factory app services. Run this if the app doesn't load on the hotspot.
echo "Restarting Smart Factory and Vision services..."
sudo systemctl restart smart-factory
sudo systemctl restart vision
sleep 2
echo "Done. Try http://192.168.4.1:8080"
