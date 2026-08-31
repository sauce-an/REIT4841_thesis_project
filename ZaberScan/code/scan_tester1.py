"""
Raster scan imaging script using Zaber Stages and NI-DAQmx with SYNCHRONOUS LIVE 3-SUBPLOT OSCILLOSCOPE.
Performs a 2D serpentine raster scan over user-defined coordinates while displaying the live 3-subplot
diagnostic oscilloscope on every pixel:
  - Subplot 1 (Top)    : Laser Terminal AC-Coupled Signal (Dev1/ai7 — 320 samples @ 20 kHz / 16.0 ms)
  - Subplot 2 (Middle) : Optical Chopper Reference Signal (Dev1/ai15 — 320 samples @ 20 kHz / 16.0 ms)
  - Subplot 3 (Bottom) : Per-Pixel Demodulation Breakdown (High/Low plateaus, median split, and ΔV)

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
from nidaqmx.constants import TerminalConfiguration

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

# Target Step Size (200 um)
TARGET_STEP_UM = 200.0
SCAN_LABEL = "scan_5"      # Identifier for multi-round scans (e.g. scan_1, scan_2, scan_3, scan_4, scan_5)
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
# 2. DAQ ACQUISITION & DEMODULATION SETTINGS
# ==============================================================================
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"       # Channel 7: Laser terminal AC-coupled signal
CHOPPER_CHANNEL = f"{DEVICE_NAME}/ai15"    # Channel 15: Optical chopper TTL reference
CHOPPER_FREQ_HZ = 1000                     # Optical chopper frequency (1000 Hz)
PERIODS_PER_POINT = 16                     # 16 chopper periods averaged per pixel (16.0 ms)
DAQ_RATE = 20_000                          # 20 kHz DAQ rate (20 samples per chopper cycle)
DAQ_SAMPLES_PER_POINT = int(DAQ_RATE * PERIODS_PER_POINT / CHOPPER_FREQ_HZ)  # 320 samples (16.0 ms)
SATURATION_THRESHOLD_V = 9.8               # Voltage rail threshold


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
# 3. SET UP LIVE 3-SUBPLOT GUI (POPS UP IMMEDIATELY)
# ==============================================================================
print("=" * 70)
print(f"ZABER RASTER SCANNING & LFI IMAGING — {int(TARGET_STEP_UM)} um RESOLUTION (LIVE MONITOR)")
print("=" * 70)
print(f"Scan Dimensions : {scan_width_mm:.2f} mm (X) x {scan_height_mm:.2f} mm (Y)")
print(f"Grid Resolution : {NUM_X} (X) x {NUM_Y} (Y) = {NUM_X * NUM_Y:,} total pixels")
print(f"Actual Step Size: {actual_x_step_um:.1f} um (X) x {actual_y_step_um:.1f} um (Y)")
print(f"Z Focus Position: {Z_FOCUS_NATIVE} native units")
print(f"DAQ Rate        : {DAQ_RATE:,} Hz | Channels: {LASER_CHANNEL}, {CHOPPER_CHANNEL}")
print("=" * 70, flush=True)

# Initialize 2D Image Matrix (Row = Y, Col = X)
image_data = np.full((NUM_Y, NUM_X), np.nan, dtype=np.float64)

# Create 3-Subplot GUI Window
plt.ion()
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax_laser, ax_chop, ax_pixel) = plt.subplots(3, 1, figsize=(12, 9.5))
try:
    fig.canvas.manager.set_window_title(f"NI-DAQ Live Monitor & Scanning — {int(TARGET_STEP_UM)} µm ({SCAN_LABEL})")
except Exception:
    pass

t_pixel_ms = np.arange(DAQ_SAMPLES_PER_POINT) / DAQ_RATE * 1000  # Time axis (16.0 ms)

# --- SUBPLOT 1: Laser Terminal AC Waveform (320 samples @ 20 kHz) ---
line_laser, = ax_laser.plot(t_pixel_ms, np.zeros(DAQ_SAMPLES_PER_POINT), color="#1f77b4", lw=1.6, label="Laser Terminal (ai7)")
line_scope_med = ax_laser.axhline(0.0, color="#d62728", linestyle="--", lw=1.2, alpha=0.8, label="Median Threshold")
ax_laser.axhline(0.0, color="#666666", linestyle=":", lw=1.0, alpha=0.6)  # 0V Baseline
ax_laser.set_ylabel("Laser (V)", fontsize=10.5, fontweight="bold")
ax_laser.set_ylim(-5.0, 5.0)
ax_laser.set_xlim(0, 16.0)
ax_laser.grid(True, linestyle="--", alpha=0.6)
ax_laser.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
ax_laser.set_title("(1) Laser Terminal AC Signal (ai7) | Initializing...", fontsize=10.5, fontweight="bold", color="#0b5394")

# --- SUBPLOT 2: Optical Chopper Reference Signal (ai15) ---
line_chop, = ax_chop.plot(t_pixel_ms, np.zeros(DAQ_SAMPLES_PER_POINT), color="#ff7f0e", lw=1.6, label="Chopper Reference (ai15)")
ax_chop.axhline(0.0, color="#666666", linestyle=":", lw=1.0, alpha=0.6)
ax_chop.set_ylabel("Chopper TTL (V)", fontsize=10.5, fontweight="bold")
ax_chop.set_ylim(-0.5, 5.5)
ax_chop.set_xlim(0, 16.0)
ax_chop.grid(True, linestyle="--", alpha=0.6)
ax_chop.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
ax_chop.set_title("(2) Optical Chopper Reference Signal (ai15) | Initializing...", fontsize=10.5, fontweight="bold", color="#b45f06")

# --- SUBPLOT 3: Per-Pixel Demodulation Breakdown (High/Low plateaus, ΔV) ---
line_pixel_high, = ax_pixel.plot([], [], 'o', color="#2ca02c", markersize=3.5, label="High State Samples (> Median)")
line_pixel_low,  = ax_pixel.plot([], [], 'o', color="#9467bd", markersize=3.5, label="Low State Samples (<= Median)")
line_pixel_wave, = ax_pixel.plot(t_pixel_ms, np.zeros(DAQ_SAMPLES_PER_POINT), color="#7f7f7f", lw=1.0, alpha=0.6)
line_pixel_hmean = ax_pixel.axhline(0.0, color="#2ca02c", linestyle="-", lw=1.5, label="High Plateau Mean")
line_pixel_lmean = ax_pixel.axhline(0.0, color="#9467bd", linestyle="-", lw=1.5, label="Low Plateau Mean")
line_pixel_med   = ax_pixel.axhline(0.0, color="#d62728", linestyle="--", lw=1.2, label="Median Threshold")

ax_pixel.set_xlabel("Pixel Acquisition Time (ms) — [16 Chopper Cycles / 320 Samples @ 20 kHz]", fontsize=11, fontweight="bold")
ax_pixel.set_ylabel("Sampled Voltage (V)", fontsize=10.5, fontweight="bold")
ax_pixel.set_ylim(-5.0, 5.0)
ax_pixel.set_xlim(0, 16.0)
ax_pixel.grid(True, linestyle="--", alpha=0.6)
ax_pixel.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=3)
ax_pixel.set_title("(3) Per-Pixel Demodulation Breakdown (320 Samples / 16.0 ms) | Waiting...", fontsize=10.5, fontweight="bold", color="#1b5e20")

plt.tight_layout()
plt.show(block=False)
fig.canvas.flush_events()
plt.pause(0.1)

# ==============================================================================
# 4. MAIN SCANNING ROUTINE
# ==============================================================================
try:
    with Connection.open_serial_port(SERIAL_PORT) as connection:
        device_list = connection.detect_devices(identify_devices=True)
        print(f"[OK] Connected on {SERIAL_PORT}. Found {len(device_list)} Zaber devices.\n", flush=True)

        y_axis = connection.get_device(Y_DEVICE)
        x_axis = connection.get_device(X_DEVICE)
        z_axis = connection.get_device(Z_DEVICE)

        # Move to Focus Position
        print(f"Setting Z focus to {Z_FOCUS_NATIVE} native units...", flush=True)
        z_axis.move_absolute(Z_FOCUS_NATIVE)
        current_z = z_axis.get_position()
        print(f"[OK] Z positioned at {current_z:.0f} native units.", flush=True)

        # Move to Starting Corner (Bottom-Left)
        print(f"Moving stages to Start Corner: X={X_START_NATIVE}, Y={Y_START_NATIVE}...", flush=True)
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages aligned at start position.\n", flush=True)

        # Initialize DAQ Task for both channels
        with nidaqmx.Task() as daq_task:
            daq_task.ai_channels.add_ai_voltage_chan(LASER_CHANNEL, terminal_config=TerminalConfiguration.RSE)
            daq_task.ai_channels.add_ai_voltage_chan(CHOPPER_CHANNEL, terminal_config=TerminalConfiguration.RSE)
            daq_task.timing.cfg_samp_clk_timing(rate=DAQ_RATE, samps_per_chan=DAQ_SAMPLES_PER_POINT)
            print(f"[OK] DAQ Task initialized on {LASER_CHANNEL} & {CHOPPER_CHANNEL}.\n", flush=True)

            print(">>> STARTING SERPENTINE RASTER SCAN <<<", flush=True)
            start_time = time.time()
            saturated_points_count = 0
            point_counter = 0

            for row_idx in range(NUM_Y):
                row_y_pos = int(y_coords[row_idx])
                y_axis.move_absolute(row_y_pos)

                is_even_row = (row_idx % 2 == 0)
                col_indices = list(range(NUM_X) if is_even_row else range(NUM_X - 1, -1, -1))

                for col_idx in col_indices:
                    col_x_pos = int(x_coords[col_idx])
                    x_axis.move_absolute(col_x_pos)
                    point_counter += 1

                    # Settling delay
                    time.sleep(0.008)

                    # Acquire both channels synchronously
                    samples_2ch = np.asarray(daq_task.read(number_of_samples_per_channel=DAQ_SAMPLES_PER_POINT))
                    raw_laser = samples_2ch[0]
                    raw_chop = samples_2ch[1]

                    confocal_val, is_sat = extract_confocal_signal(raw_laser, saturation_v=SATURATION_THRESHOLD_V)
                    image_data[row_idx, col_idx] = confocal_val
                    if is_sat:
                        saturated_points_count += 1

                    # --- LIVE 3-SUBPLOT WAVEFORM UPDATE ON EVERY PIXEL ---
                    line_laser.set_ydata(raw_laser)
                    line_chop.set_ydata(raw_chop)
                    line_pixel_wave.set_ydata(raw_laser)

                    med_val = float(np.median(raw_laser))
                    high_mask = raw_laser > med_val
                    low_mask = raw_laser <= med_val
                    high_pts = raw_laser[high_mask]
                    low_pts = raw_laser[low_mask]

                    line_pixel_high.set_data(t_pixel_ms[high_mask], high_pts)
                    line_pixel_low.set_data(t_pixel_ms[low_mask], low_pts)

                    h_mean = float(high_pts.mean()) if high_pts.size else 0.0
                    l_mean = float(low_pts.mean()) if low_pts.size else 0.0

                    line_scope_med.set_ydata([med_val, med_val])
                    line_pixel_hmean.set_ydata([h_mean, h_mean])
                    line_pixel_lmean.set_ydata([l_mean, l_mean])
                    line_pixel_med.set_ydata([med_val, med_val])

                    # Dynamic auto-scale Y axes
                    l_min, l_max = float(raw_laser.min()), float(raw_laser.max())
                    margin_l = max(0.4, (l_max - l_min) * 0.18)
                    ax_laser.set_ylim(l_min - margin_l, l_max + margin_l)
                    ax_pixel.set_ylim(l_min - margin_l, l_max + margin_l)

                    c_min, c_max = float(raw_chop.min()), float(raw_chop.max())
                    margin_c = max(0.5, (c_max - c_min) * 0.18)
                    ax_chop.set_ylim(min(-0.5, c_min - margin_c), max(5.5, c_max + margin_c))

                    # Update Titles
                    is_clipped = bool(np.any(np.abs(raw_laser) > 9.8))
                    clip_warn = " [RAIL CLIPPING!]" if is_clipped else ""
                    elapsed_now = time.time() - start_time
                    est_rem_now = max(0.0, (elapsed_now / max(1, point_counter)) * (NUM_X * NUM_Y) - elapsed_now)

                    ax_laser.set_title(
                        f"[Row {row_idx+1}/{NUM_Y} ({point_counter/(NUM_X*NUM_Y)*100:.1f}%) | Rem: {est_rem_now/60:.1f}m] Laser (ai7) | High-Low: {confocal_val:.3f} V | DC: {raw_laser.mean():+.2f} V{clip_warn}",
                        fontsize=10.0,
                        fontweight="bold",
                        color="#b30000" if is_clipped else "#0b5394",
                    )
                    ax_chop.set_title(
                        f"Optical Chopper (ai15) | TTL Span: {c_min:.1f}V – {c_max:.1f}V | Point {point_counter}/{NUM_X*NUM_Y}",
                        fontsize=10.0,
                        fontweight="bold",
                        color="#b45f06",
                    )
                    ax_pixel.set_title(
                        f"Per-Pixel Demodulation [Col {col_idx+1}/{NUM_X}] | ΔV = {confocal_val:.3f} V | High: {h_mean:.2f}V, Low: {l_mean:.2f}V",
                        fontsize=10.0,
                        fontweight="bold",
                        color="#1b5e20",
                    )

                    # Instant non-blocking GUI event flush
                    fig.canvas.flush_events()

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

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)

                print(f"  Row {rows_done:3d}/{NUM_Y} ({pct:5.1f}%) | "
                      f"Elapsed: {elapsed / 60:.1f}m | "
                      f"Remaining: {est_remaining / 60:.1f}m | "
                      f"Last Row Signal: {row_range_str} | Sat: {saturated_points_count}", flush=True)

            total_scan_time = time.time() - start_time
            print(f"\n[DONE] Scan finished in {total_scan_time / 60:.2f} minutes!", flush=True)

        # Return safely to start corner
        print("Returning stages to start position...", flush=True)
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages returned to start.", flush=True)

except Exception as e:
    print(f"\n[ERROR] An exception occurred during scan: {e}", flush=True)
    sys.exit(1)

# ==============================================================================
# 5. SAVE DATA & FINAL HIGH-RES FIGURE
# ==============================================================================
print("\n" + "=" * 70)
print("PROCESSING & SAVING IMAGE DATA")
print("=" * 70, flush=True)

npy_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.npy")
csv_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.csv")
png_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}.png")
chk_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}_checkpoint.npy")

np.save(npy_path, image_data)
np.savetxt(csv_path, image_data, delimiter=",", fmt="%.6e")
if os.path.exists(chk_path):
    try:
        os.remove(chk_path)
    except Exception:
        pass

print(f"  - Saved raw matrix to: {npy_path}")
print(f"  - Saved CSV data to  : {csv_path}")

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

# Save standalone publication figure
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig_save = Figure(figsize=(9, 8), dpi=150)
FigureCanvasAgg(fig_save)
ax_out = fig_save.add_subplot(111)

cmap_fig = plt.cm.gray.copy()
cmap_fig.set_bad(color="white")
im_out = ax_out.imshow(
    image_data,
    cmap=cmap_fig,
    origin="lower",
    extent=[0, scan_width_mm, 0, scan_height_mm],
    aspect="equal",
    vmin=vmin,
    vmax=vmax,
)
cbar_out = fig_save.colorbar(im_out, ax=ax_out, fraction=0.046, pad=0.04)
cbar_out.set_label(f"Demodulated LFI Reflectance (V) [Adaptive {vmin:.2f}V – {vmax:.2f}V]", fontsize=11, fontweight="bold")
ax_out.set_title(f"LFI 2D Raster Scan — Coin Target\n({scan_width_mm:.2f} x {scan_height_mm:.2f} mm, ~{int(TARGET_STEP_UM)} µm step | High-Low Demod)", fontsize=12, fontweight="bold")
ax_out.set_xlabel("X Position (mm)", fontsize=11, fontweight="bold")
ax_out.set_ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
fig_save.tight_layout()
fig_save.savefig(png_path, dpi=300)

print(f"  - Saved image figure : {png_path}")
print("\n[SUCCESS] Image processing and saving complete.")

# Update window title to done and wait for close
ax_laser.set_title(f"[SCAN COMPLETE] Saved to {png_path}", fontsize=10.5, fontweight="bold", color="#1b5e20")
fig.canvas.draw_idle()
print("\nClose the Live GUI Monitor window to exit.")
plt.ioff()
plt.show()