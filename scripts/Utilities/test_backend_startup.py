#!/usr/bin/env python3
"""
Quick test to check if backend can start without errors
"""

import sys
import os

# Add backend directory to path
backend_dir = os.path.join(os.path.dirname(__file__), 'pwa-dobot-plc', 'backend')
sys.path.insert(0, backend_dir)

print("Testing backend imports...")

try:
    print("1. Importing plc_client...")
    from plc_client import PLCClient
    print("   ✓ plc_client imported successfully")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

try:
    print("2. Importing dobot_client...")
    from dobot_client import DobotClient
    print("   ✓ dobot_client imported successfully")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

try:
    print("3. Importing camera_service...")
    from camera_service import CameraService
    print("   ✓ camera_service imported successfully")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

try:
    print("4. Testing PLCClient instantiation...")
    plc = PLCClient(ip='192.168.1.150', rack=0, slot=1)
    print("   ✓ PLCClient created successfully")
    print(f"   - snap7 available: {plc.client is not None}")
    print(f"   - initial state: {plc.get_status()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All backend components can be imported and instantiated!")
print("Backend should be able to start.")
