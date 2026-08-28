"""
Raster scan imaging script using Zaber Stages and NI-DAQmx.
Performs a 2D serpentine raster scan over user-defined coordinates
and generates a grayscale LFI reflectance image.

CLOSE ZABER CONSOLE AND NI MAX BEFORE RUNNING THIS SCRIPT.
"""
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from zaber_motion import Units, Library
from zaber_motion.binary import Connection
import nidaqmx

# Ensure unbuffered real-time stdout updates in PowerShell/terminal
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# Setup results directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Enable local device database for unit conversions
Library.enable_device_db_store()

# ==============================================================================
# 1. HARDWARE CONFIGURATION & CALIBRATED POSITIONS
# ==============================================================================
SERIAL_PORT = "COM5"

# Axis & Device Mapping
Y_DEVICE = 4  # Up / Down
Z_DEVICE = 5  # Forward / Backward (Focus)
X_DEVICE = 6  # Left / Right

# Optimized Scan Area (Native Microsteps - tightly cropped with ~0.7 mm safety margin)
# Microstep resolution = 0.047625 um/step (approx 20,997.4 steps/mm)
X_START_NATIVE = 890000    # Left boundary of coin window (~21.4 mm span)
X_END_NATIVE   = 440000    # Right boundary of coin window
Y_START_NATIVE = 475000    # Bottom boundary of coin window (~20.5 mm span)
Y_END_NATIVE   = 905000    # Top boundary of coin window
Z_FOCUS_NATIVE = 108498    # Current Z focus depth (~5.167 mm)

# Note: Physical setup is confirmed collision-safe across full travel range.

# Target Step Size (200 um)
TARGET_STEP_UM = 200.0
SCAN_LABEL = "scan_2"  # Identifier for multi-round scans (e.g. scan_1, scan_2, scan_3)
MICROSTEPS_PER_UM = 1.0 / 0.047625  # ~20.9974 steps per um
STEP_NATIVE = TARGET_STEP_UM * MICROSTEPS_PER_UM  # ~4199.47 steps

# Calculate Grid Points
NUM_X = int(round(abs(X_END_NATIVE - X_START_NATIVE) / STEP_NATIVE)) + 1
NUM_Y = int(round(abs(Y_END_NATIVE - Y_START_NATIVE) / STEP_NATIVE)) + 1

# Generate Coordinate Grid
x_coords = np.linspace(X_START_NATIVE, X_END_NATIVE, NUM_X, dtype=int)
y_coords = np.linspace(Y_START_NATIVE, Y_END_NATIVE, NUM_Y, dtype=int)

actual_x_step_um = abs(x_coords[1] - x_coords[0]) * 0.047625
actual_y_step_um = abs(y_coords[1] - y_coords[0]) * 0.047625
scan_width_mm = abs(X_END_NATIVE - X_START_NATIVE) * 0.047625 / 1000.0
scan_height_mm = abs(Y_END_NATIVE - Y_START_NATIVE) * 0.047625 / 1000.0

# ==============================================================================
# 2. DAQ ACQUISITION & DEMODULATION SETTINGS (High-Low Median Split)
# ==============================================================================
DAQ_CHANNEL = "Dev1/ai7"           # Laser terminal AC-coupled signal (Channel 7)
CHOPPER_FREQ_HZ = 1000             # Optical chopper frequency (1000 Hz)
PERIODS_PER_POINT = 16             # 16 chopper periods averaged per pixel (16.0 ms)
DAQ_RATE = 20_000                  # 20 kHz DAQ rate (20 samples per chopper cycle)
DAQ_SAMPLES_PER_POINT = int(DAQ_RATE * PERIODS_PER_POINT / CHOPPER_FREQ_HZ)  # 320 samples (16.0 ms)
SATURATION_THRESHOLD_V = 9.8       # Voltage rail threshold (samples above this are saturated)


def extract_confocal_signal(raw_samples: np.ndarray, saturation_v: float = 9.8) -> tuple[float, bool]:
    """
    Demodulate AC-coupled chopper-modulated waveform into a confocal LFI signal.
    Follows Mowla et al. 2018: signal = mean(high state) - mean(low state).
    Returns (signal, is_saturated).
    """
    data = np.asarray(raw_samples)
    is_saturated = bool(np.all(np.abs(data) > saturation_v) or np.ptp(data) < 1e-4)
    if is_saturated:
        return np.nan, True
    threshold = np.median(data)
    high_vals = data[data > threshold]
    low_vals = data[data <= threshold]
    if high_vals.size and low_vals.size:
        return float(high_vals.mean() - low_vals.mean()), False
    return np.nan, True


# ==============================================================================
# 3. MAIN SCANNING ROUTINE
# ==============================================================================
print("=" * 70)
print(f"ZABER RASTER SCANNING & LFI IMAGING — {int(TARGET_STEP_UM)} um RESOLUTION (HIGH-LOW DEMOD)")
print("=" * 70)
print(f"Scan Dimensions : {scan_width_mm:.2f} mm (X) x {scan_height_mm:.2f} mm (Y)")
print(f"Grid Resolution : {NUM_X} (X) x {NUM_Y} (Y) = {NUM_X * NUM_Y:,} total pixels")
print(f"Actual Step Size: {actual_x_step_um:.1f} um (X) x {actual_y_step_um:.1f} um (Y)")
print(f"Z Focus Position: {Z_FOCUS_NATIVE} native units")
print(f"DAQ Channel     : {DAQ_CHANNEL} ({DAQ_SAMPLES_PER_POINT} samples/pixel @ {DAQ_RATE:,} Hz | 1000 Hz Chopper, High-Low)")
print("=" * 70)

# Initialize 2D Image Matrix (Row = Y, Col = X)
image_data = np.zeros((NUM_Y, NUM_X), dtype=np.float64)

try:
    with Connection.open_serial_port(SERIAL_PORT) as connection:
        device_list = connection.detect_devices(identify_devices=True)
        print(f"[OK] Connected on {SERIAL_PORT}. Found {len(device_list)} Zaber devices.\n")

        y_axis = connection.get_device(Y_DEVICE)
        x_axis = connection.get_device(X_DEVICE)
        z_axis = connection.get_device(Z_DEVICE)

        # Move to Focus Position
        print(f"Setting Z focus to {Z_FOCUS_NATIVE} native units...")
        z_axis.move_absolute(Z_FOCUS_NATIVE)
        current_z = z_axis.get_position()
        print(f"[OK] Z positioned at {current_z:.0f} native units.")

        # Move to Starting Corner (Bottom-Left)
        print(f"Moving stages to Start Corner: X={X_START_NATIVE}, Y={Y_START_NATIVE}...")
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages aligned at start position.\n")

        # Initialize DAQ Task
        with nidaqmx.Task() as daq_task:
            daq_task.ai_channels.add_ai_voltage_chan(DAQ_CHANNEL)
            daq_task.timing.cfg_samp_clk_timing(rate=DAQ_RATE, samps_per_chan=DAQ_SAMPLES_PER_POINT)
            print(f"[OK] DAQ Task initialized on {DAQ_CHANNEL}.\n")

            print(">>> STARTING SERPENTINE RASTER SCAN <<<")
            start_time = time.time()
            saturated_points_count = 0

            for row_idx in range(NUM_Y):
                row_y_pos = int(y_coords[row_idx])
                y_axis.move_absolute(row_y_pos)

                # Serpentine direction
                is_even_row = (row_idx % 2 == 0)
                col_indices = range(NUM_X) if is_even_row else range(NUM_X - 1, -1, -1)

                for col_idx in col_indices:
                    col_x_pos = int(x_coords[col_idx])
                    x_axis.move_absolute(col_x_pos)

                    # Stage settling delay (optimized for 50 um microsteps)
                    time.sleep(0.008)

                    # Acquire DAQ Signal
                    raw_samples = daq_task.read(number_of_samples_per_channel=DAQ_SAMPLES_PER_POINT)
                    confocal_val, is_sat = extract_confocal_signal(raw_samples, saturation_v=SATURATION_THRESHOLD_V)

                    # Save to Image Array
                    image_data[row_idx, col_idx] = confocal_val
                    if is_sat:
                        saturated_points_count += 1

                # Periodic Checkpoint Saving every 10 rows
                if (row_idx + 1) % 10 == 0:
                    np.save(os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}_checkpoint.npy"), image_data)

                # Progress Report after each row
                elapsed = time.time() - start_time
                rows_done = row_idx + 1
                est_total = (elapsed / rows_done) * NUM_Y
                est_remaining = max(0.0, est_total - elapsed)
                pct = (rows_done / NUM_Y) * 100.0

                row_valid = image_data[row_idx][~np.isnan(image_data[row_idx])]
                row_range_str = f"[{row_valid.min():.3f}V, {row_valid.max():.3f}V]" if row_valid.size else "[ALL SATURATED/NAN]"

                print(f"  Row {rows_done:3d}/{NUM_Y} ({pct:5.1f}%) | "
                      f"Elapsed: {elapsed / 60:.1f}m | "
                      f"Remaining: {est_remaining / 60:.1f}m | "
                      f"Last Row Signal: {row_range_str} | Sat: {saturated_points_count}", flush=True)

            total_scan_time = time.time() - start_time
            print(f"\n[DONE] Scan finished in {total_scan_time / 60:.2f} minutes!")

        # Return safely to start corner
        print("Returning stages to start position...")
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages returned to start.")

except Exception as e:
    print(f"\n[ERROR] An exception occurred during scan: {e}")
    sys.exit(1)

# ==============================================================================
# 4. SAVE DATA & DISPLAY RESULT
# ==============================================================================
print("\n" + "=" * 70)
print("PROCESSING & SAVING IMAGE DATA")
print("=" * 70)

# Save Raw Numerical Data (for MATLAB / Python / ImageJ analysis)
npy_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.npy")
csv_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.csv")
png_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.png")

np.save(npy_path, image_data)
np.savetxt(csv_path, image_data, delimiter=",", fmt="%.6e")
print(f"  - Saved raw matrix to: {npy_path}")
print(f"  - Saved CSV data to  : {csv_path}")

# Compute Adaptive Robust Contrast Limits (1% - 99% percentile on valid non-NaN pixels)
# This eliminates outlier specular saturation spikes (10V) and dropouts (0V),
# stretching the full black-to-white dynamic range across real surface features.
valid_pixels = image_data[~np.isnan(image_data)]
if valid_pixels.size:
    vmin = float(np.percentile(valid_pixels, 1.0))
    vmax = float(np.percentile(valid_pixels, 99.0))
    if vmax - vmin < 1e-3:
        vmin, vmax = float(np.min(valid_pixels)), float(np.max(valid_pixels))
    raw_min_str = f"{np.min(valid_pixels):.3f} V"
    raw_max_str = f"{np.max(valid_pixels):.3f} V"
else:
    vmin, vmax = 0.0, 1.0
    raw_min_str, raw_max_str = "N/A", "N/A"

sat_pct = (saturated_points_count / (NUM_X * NUM_Y)) * 100.0
print(f"  - Saturated/Rail points : {saturated_points_count:,} / {NUM_X * NUM_Y:,} ({sat_pct:.2f}%)")
print(f"  - Valid raw signal range: [{raw_min_str}, {raw_max_str}]")
print(f"  - Adaptive display range: [{vmin:.3f} V, {vmax:.3f} V]")

# Plot & Save Grayscale Figure with Adaptive Contrast & Saturation Masking
plt.figure(figsize=(9, 8), dpi=150)
cmap_custom = plt.cm.gray.copy()
cmap_custom.set_bad(color="red")  # Highlights saturated rail pixels in red

im = plt.imshow(
    image_data,
    cmap=cmap_custom,
    origin="lower",  # Row 0 (Y_START) at bottom, Row N (Y_END) at top
    extent=[0, scan_width_mm, 0, scan_height_mm],
    aspect="equal",
    vmin=vmin,
    vmax=vmax,
)
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label(f"Demodulated LFI Reflectance (V) [Adaptive {vmin:.2f}V – {vmax:.2f}V]", fontsize=11, fontweight="bold")

plt.title(f"LFI 2D Raster Scan — Coin Target\n({scan_width_mm:.2f} x {scan_height_mm:.2f} mm, ~{int(TARGET_STEP_UM)} µm step | High-Low Demod)", fontsize=12, fontweight="bold")
plt.xlabel("X Position (mm)", fontsize=11, fontweight="bold")
plt.ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
plt.grid(False)
plt.tight_layout()

plt.savefig(png_path, dpi=300)
plt.close()
print(f"  - Saved image figure : {png_path}")
print("\n[SUCCESS] Image processing and saving complete.")