"""
Diagnostic: Binary protocol on COM5.
CLOSE ZABER CONSOLE BEFORE RUNNING THIS SCRIPT.
"""
from zaber_motion import Units, Library
from zaber_motion.binary import Connection

# Enable local device database for unit conversions
Library.enable_device_db_store()

print("=" * 50)
print("Zaber Binary Protocol Diagnostic — COM5")
print("=" * 50)
print("\n** Make sure Zaber Console is CLOSED! **\n")

try:
    with Connection.open_serial_port("COM5") as connection:
        print("[OK] Serial port COM5 opened.\n")

        print("Detecting devices (binary protocol)...")
        device_list = connection.detect_devices(identify_devices=True)
        print(f"[OK] Found {len(device_list)} device(s).\n")

        for dev in device_list:
            print(f"--- Device Address: {dev.device_address} ---")
            print(f"  Device ID:  {dev.device_id}")
            print(f"  Name:       {dev.name}")
            try:
                pos = dev.get_position()
                print(f"  Position:   {pos} (native units)")
            except Exception as e:
                print(f"  Position:   ERROR - {e}")
            print()

except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    print("\nIf 'NoDeviceFoundException', make sure:")
    print("  1. Zaber Console is CLOSED")
    print("  2. Devices are powered on")
    print("  3. USB cable is connected")
