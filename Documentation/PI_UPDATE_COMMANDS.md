# Regular Update Commands
cd ~/rpi-dobot
git pull origin main

cd pwa-dobot-plc/backend
source venv/bin/activate

# Restart the app
python app.py

---

# Official Dobot API Migration (One-time)

## Full automated deployment from Windows:
bash deploy_official_api.sh

## Or manual steps on Pi:
cd ~/rpi-dobot
bash setup_official_dobot_api.sh
bash migrate_to_official_api.sh

## Test after migration:
python3 test_official_api_connection.py
python3 test_official_api_movement.py
python3 test_official_api_peripherals.py