# Git Deployment Commands


cd C:\Users\HamedA\Documents\rpi-dobot
git add .
git commit -m "updates"
git push

ssh pi@rpi



cd ~/rpi-dobot
git pull
pm2 restart pwa-dobot-plc

