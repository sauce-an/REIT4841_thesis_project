"""
Test: Home Device 4 only (Y-axis — Up/Down).
Purpose: Verify device 4 is the Y-axis (up/down movement).
CLOSE ZABER CONSOLE BEFORE RUNNING THIS SCRIPT.
"""
from zaber_motion import Units, Library
from zaber_motion.binary import Connection

Library.enable_device_db_store()

DEVICE_ADDRESS = 4  # Expected: Y-axis (Up/Down)

print("=" * 50)
print(f"Homing DEVICE {DEVICE_ADDRESS} (Y-AXIS)")
print("Expected movement: UP/DOWN")
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

        print("\n>>> HOMING NOW — watch for UP/DOWN movement <<<")
        device.home()

        pos_after = device.get_position()
        print(f"Position after homing:  {pos_after} (native units)")

        print("\n[DONE] Did the stage move UP/DOWN?")
        print("  YES → Device 4 is correctly assigned as Y-axis.")
        print("  NO  → Device 4 is mapped to the wrong axis!")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
