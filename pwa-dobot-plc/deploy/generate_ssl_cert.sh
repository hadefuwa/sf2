#!/bin/bash
# Generate self-signed SSL certificate for HTTPS (required for WinCC Unified HMI)
# Run this on the Raspberry Pi: ./deploy/generate_ssl_cert.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(dirname "$SCRIPT_DIR")/backend"
SSL_DIR="$BACKEND_DIR/ssl"
CERT_FILE="$SSL_DIR/cert.pem"
KEY_FILE="$SSL_DIR/key.pem"

echo "=========================================="
echo "  Generate HTTPS Certificate for WinCC"
echo "=========================================="

# Create ssl directory
mkdir -p "$SSL_DIR"
cd "$SSL_DIR"

# Get IP or hostname for Common Name (helps with certificate validation)
CN="${1:-192.168.7.5}"
echo "Using Common Name: $CN (use your Pi's IP or hostname)"
echo ""

# Generate self-signed certificate (valid 10 years)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 3650 -nodes \
  -subj "/CN=$CN/O=Smart Factory/OU=Vision Camera" \
  -addext "subjectAltName=IP:192.168.7.5,IP:192.168.1.1,DNS:rpi,DNS:raspberrypi.local"

# Set permissions
chmod 600 key.pem
chmod 644 cert.pem

echo ""
echo "✅ Certificate generated successfully!"
echo ""
echo "  Certificate: $CERT_FILE"
echo "  Private key: $KEY_FILE"
echo ""
echo "  Restart the app to enable HTTPS:"
echo "  pm2 restart pwa-dobot-plc"
echo ""
echo "  HTTPS URLs:"
echo "  - https://192.168.7.5:8080/api/camera/stream"
echo "  - https://$CN:8080/"
echo ""
echo "  Note: WinCC may show a certificate warning (self-signed)."
echo "  Accept/trust the certificate in the HMI to enable the camera."
echo "=========================================="
