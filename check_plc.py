#!/usr/bin/env python3
"""Quick diagnostic script to test PLC connection"""
import sys
sys.path.insert(0, '/home/pi/rpi-dobot/pwa-dobot-plc/backend')

from plc_client import PLCClient
import json

print("="*50)
print("PLC Connection Diagnostic")
print("="*50)

# Create client
client = PLCClient("192.168.7.2", 0, 1)
print(f"Client created: {client}")

# Try to connect
print("\nAttempting to connect to PLC at 192.168.7.2...")
connected = client.connect()
print(f"Connection result: {connected}")

# Get status
status = client.get_status()
print(f"\nPLC Status:")
print(json.dumps(status, indent=2))

if connected:
    print("\n✓ PLC IS CONNECTED")
    
    # Try to read vision tags
    print("\nReading vision tags from DB123...")
    try:
        tags = client.read_vision_tags(123)
        print(f"Vision tags:")
        print(json.dumps(tags, indent=2))
        
        print(f"\n>>> Start bit (DB123.DBX40.0) = {tags['start']}")
        
    except Exception as e:
        print(f"✗ Error reading tags: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ PLC IS NOT CONNECTED")
    print(f"Error: {status.get('last_error', 'Unknown')}")

print("\n" + "="*50)


