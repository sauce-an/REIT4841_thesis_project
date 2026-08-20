"""
Test: Home Device 6 only (X-axis — Left/Right).
Purpose: Verify device 6 is the X-axis (left/right movement).
CLOSE ZABER CONSOLE BEFORE RUNNING THIS SCRIPT.
"""
from zaber_motion import Units, Library
from zaber_motion.binary import Connection

Library.enable_device_db_store()

DEVICE_ADDRESS = 6  # Expected: X-axis (Left/Right)

print("=" * 50)
print(f"Homing DEVICE {DEVICE_ADDRESS} (X-AXIS)")
print("Expected movement: LEFT/RIGHT")
print("=" * 50)
print("\n** Make sure Zaber Console is CLOSED! **\n")

try:
    with Connection.open_serial_port("COM5") as connection:
        device_list = connection.detect_devices(identify_devices=True)
        print(f"[OK] Found {len(device_list)} device(s).\n")

        device = connection.get_device(DEVICE_ADDRESS)
        print(f"Device Address: {device.device_address}")
        print(f"Device ID:      {device.device_id}")
        print(f"Name:           {device.name}")

        pos_before = device.get_position()
        print(f"\nPosition before homing: {pos_before} (native units)")

        print("\n>>> HOMING NOW — watch for LEFT/RIGHT movement <<<")
        device.home()

        pos_after = device.get_position()
        print(f"Position after homing:  {pos_after} (native units)")

        print("\n[DONE] Did the stage move LEFT/RIGHT?")
        print("  YES → Device 6 is correctly assigned as X-axis.")
        print("  NO  → Device 6 is mapped to the wrong axis!")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
