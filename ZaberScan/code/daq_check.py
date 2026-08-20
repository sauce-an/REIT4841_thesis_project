"""
Diagnostic: NI-DAQmx Dual-Channel Verification (COM / Bench Setup).
Verifies connections for:
  - Channel 7  (Dev1/ai7)  : Laser Terminal Signal
  - Channel 15 (Dev1/ai15) : Optical Chopper Reference / Sync Signal
"""
import sys
import numpy as np

print("=" * 65)
print("NI-DAQmx Diagnostic — Laser & Optical Chopper Channel Check")
print("=" * 65)

# --- Configuration ---
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"
CHOPPER_CHANNEL = f"{DEVICE_NAME}/ai15"
SAMPLE_RATE = 10_000        # 10 kHz sampling
NUM_SAMPLES = 1_000         # 1000 samples = 0.10 seconds (covers ~30 cycles at 300 Hz)

try:
    import nidaqmx
    import nidaqmx.system
    from nidaqmx.errors import DaqError
    print("[OK] nidaqmx Python package loaded.\n")
except ImportError as e:
    print(f"[ERROR] Could not import nidaqmx: {e}")
    print("Run: pip install nidaqmx")
    sys.exit(1)

try:
    system = nidaqmx.system.System.local()
    driver_ver = f"{system.driver_version.major_version}.{system.driver_version.minor_version}.{system.driver_version.update_version}"
    print(f"NI-DAQmx Driver Version: {driver_ver}")
    
    devices = system.devices
    if len(devices) == 0:
        print(f"[ERROR] No NI DAQ devices detected on system!")
        sys.exit(1)
        
    print(f"[OK] Found device(s): {', '.join([d.name + ' (' + d.product_type + ')' for d in devices])}\n")

    print(f"Sampling configuration:")
    print(f"  Laser Channel   : {LASER_CHANNEL}  (Channel 7)")
    print(f"  Chopper Channel : {CHOPPER_CHANNEL} (Channel 15)")
    print(f"  Sample Rate     : {SAMPLE_RATE:,} Hz")
    print(f"  Acquisition Size: {NUM_SAMPLES} samples ({NUM_SAMPLES / SAMPLE_RATE * 1000:.1f} ms duration)")
    print("-" * 65)

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(LASER_CHANNEL, name_to_assign_to_channel="Laser_ai7")
        task.ai_channels.add_ai_voltage_chan(CHOPPER_CHANNEL, name_to_assign_to_channel="Chopper_ai15")
        task.timing.cfg_samp_clk_timing(rate=SAMPLE_RATE, samps_per_chan=NUM_SAMPLES)

        print("Acquiring simultaneous data on ai7 and ai15...")
        raw_data = np.array(task.read(number_of_samples_per_channel=NUM_SAMPLES))
        
        laser_data = raw_data[0]
        chopper_data = raw_data[1]

    print("[OK] Acquisition complete.\n")

    # =========================================================================
    # 1. OPTICAL CHOPPER ANALYSIS (Channel 15)
    # =========================================================================
    print("=" * 30 + " CHOPPER (ai15) " + "=" * 19)
    chop_min = np.min(chopper_data)
    chop_max = np.max(chopper_data)
    chop_vpp = chop_max - chop_min
    chop_mean = np.mean(chopper_data)
    
    print(f"  Voltage Range  : {chop_min:.4f} V to {chop_max:.4f} V")
    print(f"  Peak-to-Peak   : {chop_vpp:.4f} V")
    print(f"  Mean (DC)      : {chop_mean:.4f} V")

    # Frequency estimation via mid-level threshold crossing
    chop_mid = (chop_max + chop_min) / 2.0
    binary_chop = (chopper_data > chop_mid).astype(int)
    rising_edges = np.where(np.diff(binary_chop) == 1)[0]
    
    if len(rising_edges) >= 2 and chop_vpp > 0.5:
        periods = np.diff(rising_edges) / SAMPLE_RATE
        est_freq = 1.0 / np.mean(periods)
        duty_cycle = np.mean(binary_chop) * 100.0
        print(f"  Signal Type    : Active Square Wave / Pulse detected")
        print(f"  Estimated Freq : {est_freq:.1f} Hz (Expected: ~300 Hz)")
        print(f"  Duty Cycle     : {duty_cycle:.1f}%")
        print(f"  Status         : [OK] Chopper is spinning & sending sync pulses!")
    elif chop_vpp <= 0.5:
        print(f"  Signal Type    : Static DC / Low amplitude (Vpp <= 0.5 V)")
        print(f"  Status         : [CHECK] Chopper might be stationary, unpowered, or disconnected.")
    else:
        print(f"  Signal Type    : Irregular / Noise")
        print(f"  Status         : [CHECK] Verify optical chopper wiring and sync cable.")

    # =========================================================================
    # 2. LASER TERMINAL ANALYSIS (Channel 7)
    # =========================================================================
    print("\n" + "=" * 30 + " LASER (ai7) " + "=" * 22)
    laser_min = np.min(laser_data)
    laser_max = np.max(laser_data)
    laser_vpp = laser_max - laser_min
    laser_mean = np.mean(laser_data)
    laser_std = np.std(laser_data)

    print(f"  Voltage Range  : {laser_min:.4f} V to {laser_max:.4f} V")
    print(f"  Peak-to-Peak   : {laser_vpp:.4f} V ({laser_vpp * 1000:.2f} mV)")
    print(f"  Mean (DC)      : {laser_mean:.4f} V")
    print(f"  Std Dev (AC)   : {laser_std:.6f} V")

    # Median-split demodulation (Mowla et al. 2018)
    laser_threshold = np.median(laser_data)
    high_level = np.mean(laser_data[laser_data > laser_threshold])
    low_level = np.mean(laser_data[laser_data <= laser_threshold])
    confocal_signal = high_level - low_level

    print(f"  Demodulated LFI: {confocal_signal * 1000:.3f} mV (High level - Low level)")
    
    if abs(laser_mean) > 0.01 or laser_vpp > 0.005:
        print(f"  Status         : [OK] Laser signal detected on ai7!")
    else:
        print(f"  Status         : [CHECK] Very low signal. Verify laser bias current and amplifier power.")

    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY:")
    print(f"  [Channel 7  / ai7 ] Laser Terminal   : {'CONNECTED & ACTIVE' if (laser_vpp > 0.005 or abs(laser_mean) > 0.01) else 'LOW / NO SIGNAL'}")
    print(f"  [Channel 15 / ai15] Optical Chopper  : {'CONNECTED & MODULATING' if (chop_vpp > 0.5 and len(rising_edges) >= 2) else 'STATIC / CHECK SYNC'}")
    print("=" * 65)

except DaqError as e:
    print(f"\n[ERROR] NI-DAQmx Error: {e}")
    print("\nTroubleshooting tips:")
    print(f"  1. Verify device name '{DEVICE_NAME}' in NI MAX")
    print(f"  2. Check physical screw terminal connections for AI 7 and AI 15 (and AI GND / COM)")
    print("  3. Ensure no other application (NI MAX test panel, LabVIEW) is currently using the channels")
except Exception as e:
    print(f"\n[ERROR] Unexpected error ({type(e).__name__}): {e}")
