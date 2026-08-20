"""
Test: Home Device 5 only (Z-axis — Forward/Backward / Focus).
Purpose: Verify device 5 is the Z-axis (forward/backward movement).
CLOSE ZABER CONSOLE BEFORE RUNNING THIS SCRIPT.
"""
from zaber_motion import Units, Library
from zaber_motion.binary import Connection

Library.enable_device_db_store()

DEVICE_ADDRESS = 5  # Expected: Z-axis (Forward/Backward)

print("=" * 50)
print(f"Homing DEVICE {DEVICE_ADDRESS} (Z-AXIS)")
print("Expected movement: FORWARD/BACKWARD")
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

        print("\n>>> HOMING NOW — watch for FORWARD/BACKWARD movement <<<")
        device.home()

        pos_after = device.get_position()
        print(f"Position after homing:  {pos_after} (native units)")

        print("\n[DONE] Did the stage move FORWARD/BACKWARD?")
        print("  YES → Device 5 is correctly assigned as Z-axis.")
        print("  NO  → Device 5 is mapped to the wrong axis!")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
