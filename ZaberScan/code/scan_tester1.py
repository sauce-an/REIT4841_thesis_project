"""
Raster scan imaging script using Zaber Stages and NI-DAQmx with LIVE 3-SUBPLOT OSCILLOSCOPE GUI.
Simultaneously performs the 2D serpentine raster scan in a background worker while displaying
the exact 3-subplot real-time diagnostic oscilloscope from daq_live.py:
  - Subplot 1 (Top)    : Laser Terminal AC-Coupled Confocal LFI Signal (Dev1/ai7 @ 100 kS/s)
  - Subplot 2 (Middle) : Optical Chopper Reference / Sync Signal (Dev1/ai15 @ 100 kS/s)
  - Subplot 3 (Bottom) : 320-Sample Per-Pixel Acquisition & Demodulation Emulator (20 kHz, 16 periods)

CLOSE ZABER CONSOLE AND NI MAX BEFORE RUNNING THIS SCRIPT.
"""
import sys
import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from zaber_motion import Units, Library
from zaber_motion.binary import Connection
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

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

# Optimized Scan Area (Native Microsteps)
X_START_NATIVE = 890000    # Left boundary of coin window (~21.4 mm span)
X_END_NATIVE   = 440000    # Right boundary of coin window
Y_START_NATIVE = 475000    # Bottom boundary of coin window (~20.5 mm span)
Y_END_NATIVE   = 905000    # Top boundary of coin window
Z_FOCUS_NATIVE = 108498    # Current Z focus depth (~5.167 mm)

# Target Step Size (200 um)
TARGET_STEP_UM = 200.0
SCAN_LABEL = "scan_4"      # Identifier for scan run
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
# 2. DAQ ACQUISITION & LIVE MONITOR SETTINGS
# ==============================================================================
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"       # Channel 7: Laser terminal AC-coupled signal
CHOPPER_CHANNEL = f"{DEVICE_NAME}/ai15"    # Channel 15: Optical chopper TTL reference
DAQ_RATE = 100_000                         # Live scope sampling rate (100 kS/s per channel)
EST_CHOPPER_FREQ = 1000                    # Nominal chopper frequency (1000 Hz)
DISPLAY_PERIODS = 10                       # Number of chopper cycles on top 2 subplots (10 ms)
AUTO_DETECT_CHOPPER_FREQ = True            # Auto-measure chopper frequency from TTL edges

# --- Subplot 3: Per-Pixel Acquisition Settings ---
SCAN_DAQ_RATE = 20_000                     # 20 kHz sampling rate for demodulation
SCAN_PERIODS_PER_POINT = 16                # 16 chopper periods per pixel
SCAN_SAMPLES_PER_POINT = int(SCAN_DAQ_RATE * SCAN_PERIODS_PER_POINT / EST_CHOPPER_FREQ)  # 320 samples (16.0 ms)
SCAN_DURATION_SEC = SCAN_PERIODS_PER_POINT / EST_CHOPPER_FREQ                             # 0.016 s (16.0 ms)
RAW_SAMPLES_IN_PIXEL = int(DAQ_RATE * SCAN_DURATION_SEC)                                  # 1600 samples @ 100 kHz
SATURATION_THRESHOLD_V = 9.8               # Voltage rail threshold

# Buffer & Window Setup
CHUNK_SIZE = int(DAQ_RATE * DISPLAY_PERIODS / EST_CHOPPER_FREQ)  # 1000 samples @ 100 kHz (10.0 ms)
BUFFER_SIZE = 200_000                      # 2-second circular hardware buffer
downsample_step = int(DAQ_RATE / SCAN_DAQ_RATE)  # 5

# Shared Data Matrix & Synchronization State
image_data = np.full((NUM_Y, NUM_X), np.nan, dtype=np.float64)
scan_status = {
    "running": False,
    "finished": False,
    "row": 0,
    "col": 0,
    "pct": 0.0,
    "elapsed_min": 0.0,
    "rem_min": 0.0,
    "sat_count": 0,
    "error": None,
}


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
# 3. DAQ HARDWARE INITIALIZATION
# ==============================================================================
task = nidaqmx.Task()
try:
    task.ai_channels.add_ai_voltage_chan(
        LASER_CHANNEL, 
        name_to_assign_to_channel="Laser_ai7",
        terminal_config=TerminalConfiguration.RSE
    )
    task.ai_channels.add_ai_voltage_chan(
        CHOPPER_CHANNEL, 
        name_to_assign_to_channel="Chopper_ai15",
        terminal_config=TerminalConfiguration.RSE
    )
    task.timing.cfg_samp_clk_timing(
        rate=DAQ_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=BUFFER_SIZE,
    )
    task.start()
    print("[OK] NI-DAQ Continuous Task running.\n", flush=True)
except Exception as e:
    print(f"\n[ERROR] Failed to initialize DAQ hardware: {e}", flush=True)
    print("Ensure NI MAX is closed and USB-6451 is connected.")
    try:
        task.close()
    except Exception:
        pass
    sys.exit(1)

# Rolling caches for continuous streaming
MAX_CACHE_LEN = max(CHUNK_SIZE, RAW_SAMPLES_IN_PIXEL)
cached_laser = np.zeros(MAX_CACHE_LEN)
cached_chop = np.zeros(MAX_CACHE_LEN)
cache_lock = threading.Lock()
frame_count = 0


# ==============================================================================
# 4. BACKGROUND STAGE SCANNING WORKER THREAD
# ==============================================================================
def scanning_worker():
    global image_data, scan_status
    print("=" * 70)
    print(f"ZABER RASTER SCANNING & LFI IMAGING — {int(TARGET_STEP_UM)} um RESOLUTION")
    print("=" * 70)
    print(f"Scan Dimensions : {scan_width_mm:.2f} mm (X) x {scan_height_mm:.2f} mm (Y)")
    print(f"Grid Resolution : {NUM_X} (X) x {NUM_Y} (Y) = {NUM_X * NUM_Y:,} total pixels")
    print(f"Actual Step Size: {actual_x_step_um:.1f} um (X) x {actual_y_step_um:.1f} um (Y)")
    print(f"Z Focus Position: {Z_FOCUS_NATIVE} native units")
    print(f"DAQ Rate        : {DAQ_RATE:,} Hz (Continuous 3-subplot scope) | Demod: 320 samples @ 20 kHz")
    print("=" * 70, flush=True)

    try:
        with Connection.open_serial_port(SERIAL_PORT) as connection:
            connection.detect_devices(identify_devices=True)
            y_axis = connection.get_device(Y_DEVICE)
            x_axis = connection.get_device(X_DEVICE)
            z_axis = connection.get_device(Z_DEVICE)

            print(f"Setting Z focus to {Z_FOCUS_NATIVE} native units...", flush=True)
            z_axis.move_absolute(Z_FOCUS_NATIVE)
            print(f"[OK] Z positioned at {z_axis.get_position():.0f} native units.", flush=True)

            print(f"Moving stages to Start Corner: X={X_START_NATIVE}, Y={Y_START_NATIVE}...", flush=True)
            x_axis.move_absolute(X_START_NATIVE)
            y_axis.move_absolute(Y_START_NATIVE)
            print("[OK] Stages aligned at start position.\n", flush=True)

            print(">>> STARTING SERPENTINE RASTER SCAN <<<", flush=True)
            scan_status["running"] = True
            start_time = time.time()
            sat_points = 0

            for row_idx in range(NUM_Y):
                row_y_pos = int(y_coords[row_idx])
                y_axis.move_absolute(row_y_pos)

                is_even_row = (row_idx % 2 == 0)
                col_indices = list(range(NUM_X) if is_even_row else range(NUM_X - 1, -1, -1))

                for col_idx in col_indices:
                    col_x_pos = int(x_coords[col_idx])
                    x_axis.move_absolute(col_x_pos)

                    # Settling delay
                    time.sleep(0.008)

                    # Wait for 16.0 ms of fresh laser samples
                    time.sleep(SCAN_DURATION_SEC)

                    with cache_lock:
                        raw_100k = cached_laser[-RAW_SAMPLES_IN_PIXEL:].copy()

                    samples_320 = raw_100k[::downsample_step][:SCAN_SAMPLES_PER_POINT]
                    confocal_val, is_sat = extract_confocal_signal(samples_320, saturation_v=SATURATION_THRESHOLD_V)

                    image_data[row_idx, col_idx] = confocal_val
                    if is_sat:
                        sat_points += 1

                    scan_status["row"] = row_idx + 1
                    scan_status["col"] = col_idx + 1
                    scan_status["sat_count"] = sat_points

                # Periodic Checkpoint Saving every 10 rows
                if (row_idx + 1) % 10 == 0:
                    np.save(os.path.join(RESULTS_DIR, f"coin_scan_{int(TARGET_STEP_UM)}um_{SCAN_LABEL}_checkpoint.npy"), image_data)

                elapsed = time.time() - start_time
                rows_done = row_idx + 1
                est_total = (elapsed / rows_done) * NUM_Y
                est_rem = max(0.0, est_total - elapsed)
                pct = (rows_done / NUM_Y) * 100.0

                scan_status["pct"] = pct
                scan_status["elapsed_min"] = elapsed / 60.0
                scan_status["rem_min"] = est_rem / 60.0

                row_valid = image_data[row_idx][~np.isnan(image_data[row_idx])]
                row_str = f"[{row_valid.min():.3f}V, {row_valid.max():.3f}V]" if row_valid.size else "[ALL SAT/NAN]"

                print(f"  Row {rows_done:3d}/{NUM_Y} ({pct:5.1f}%) | "
                      f"Elapsed: {elapsed / 60:.1f}m | "
                      f"Remaining: {est_rem / 60:.1f}m | "
                      f"Last Row Signal: {row_str} | Sat: {sat_points}", flush=True)

            total_scan_time = time.time() - start_time
            print(f"\n[DONE] Scan finished in {total_scan_time / 60:.2f} minutes!", flush=True)

            print("Returning stages to start position...", flush=True)
            x_axis.move_absolute(X_START_NATIVE)
            y_axis.move_absolute(Y_START_NATIVE)
            print("[OK] Stages returned to start.", flush=True)

        # Save output files
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

        valid_pixels = image_data[~np.isnan(image_data)]
        if valid_pixels.size:
            vmin = float(np.percentile(valid_pixels, 1.0))
            vmax = float(np.percentile(valid_pixels, 99.0))
            if vmax - vmin < 1e-3:
                vmin, vmax = float(np.min(valid_pixels)), float(np.max(valid_pixels))
        else:
            vmin, vmax = 0.0, 1.0

        plt.figure(figsize=(9, 8), dpi=150)
        cmap_fig = plt.cm.gray.copy()
        cmap_fig.set_bad(color="red")
        im_out = plt.imshow(
            image_data,
            cmap=cmap_fig,
            origin="lower",
            extent=[0, scan_width_mm, 0, scan_height_mm],
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        cbar_out = plt.colorbar(im_out, fraction=0.046, pad=0.04)
        cbar_out.set_label(f"Demodulated LFI Reflectance (V) [Adaptive {vmin:.2f}V – {vmax:.2f}V]", fontsize=11, fontweight="bold")
        plt.title(f"LFI 2D Raster Scan — Coin Target\n({scan_width_mm:.2f} x {scan_height_mm:.2f} mm, ~{int(TARGET_STEP_UM)} µm step | High-Low Demod)", fontsize=12, fontweight="bold")
        plt.xlabel("X Position (mm)", fontsize=11, fontweight="bold")
        plt.ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(png_path, dpi=300)
        plt.close()

        print(f"  - Saved raw matrix to: {npy_path}")
        print(f"  - Saved CSV data to  : {csv_path}")
        print(f"  - Saved image figure : {png_path}")
        print("\n[SUCCESS] Image processing and saving complete.", flush=True)
        scan_status["finished"] = True

    except Exception as e:
        print(f"\n[ERROR] Scanning error: {e}", flush=True)
        scan_status["error"] = str(e)


# Launch scanning thread
scan_thread = threading.Thread(target=scanning_worker, daemon=True)
scan_thread.start()

# ==============================================================================
# 5. SET UP EXACT 3-SUBPLOT GUI (MAIN THREAD)
# ==============================================================================
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax_laser, ax_chop, ax_pixel) = plt.subplots(3, 1, figsize=(12, 9.5))
try:
    fig.canvas.manager.set_window_title(f"NI-DAQ Live Monitor & Scanning — {int(TARGET_STEP_UM)} µm ({SCAN_LABEL})")
except Exception:
    pass

t_scope_ms = np.arange(CHUNK_SIZE) / DAQ_RATE * 1000  # Time axis for scope (10.0 ms)
t_pixel_ms = np.arange(SCAN_SAMPLES_PER_POINT) / SCAN_DAQ_RATE * 1000  # Time axis for pixel (16.0 ms)

# --- SUBPLOT 1: Live Analog Laser Waveform (Ch 7) ---
line_laser, = ax_laser.plot(t_scope_ms, np.zeros(CHUNK_SIZE), color="#1f77b4", lw=1.6, label="Laser Terminal (ai7)")
line_scope_med = ax_laser.axhline(0.0, color="#d62728", linestyle="--", lw=1.2, alpha=0.8, label="Median Threshold")
ax_laser.axhline(0.0, color="#666666", linestyle=":", lw=1.0, alpha=0.6)  # 0V Baseline
ax_laser.set_ylabel("Laser (V)", fontsize=10.5, fontweight="bold")
ax_laser.set_ylim(-5.0, 5.0)
ax_laser.grid(True, linestyle="--", alpha=0.6)
ax_laser.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
ax_laser.set_title("(1) Laser Terminal AC Signal (100 kS/s Scope View) | Initializing...", fontsize=11, fontweight="bold", color="#0b5394")

# --- SUBPLOT 2: Live Optical Chopper Reference (Ch 15) ---
line_chop, = ax_chop.plot(t_scope_ms, np.zeros(CHUNK_SIZE), color="#ff7f0e", lw=1.6, label="Optical Chopper TTL (ai15)")
ax_chop.axhline(0.0, color="#666666", linestyle=":", lw=1.0, alpha=0.6)
ax_chop.set_ylabel("Chopper TTL (V)", fontsize=10.5, fontweight="bold")
ax_chop.set_ylim(-0.5, 5.5)
ax_chop.grid(True, linestyle="--", alpha=0.6)
ax_chop.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
ax_chop.set_title("(2) Optical Chopper Reference Signal | Initializing...", fontsize=11, fontweight="bold", color="#b45f06")

# --- SUBPLOT 3: Single-Pixel Acquisition & Demodulation Emulator (320 samples @ 20 kHz) ---
line_pixel_high, = ax_pixel.plot([], [], 'o', color="#2ca02c", markersize=3.5, label="High State Samples (> Median)")
line_pixel_low,  = ax_pixel.plot([], [], 'o', color="#9467bd", markersize=3.5, label="Low State Samples (<= Median)")
line_pixel_wave, = ax_pixel.plot(t_pixel_ms, np.zeros(SCAN_SAMPLES_PER_POINT), color="#7f7f7f", lw=1.0, alpha=0.6)
line_pixel_hmean = ax_pixel.axhline(0.0, color="#2ca02c", linestyle="-", lw=1.5, label="High Plateau Mean")
line_pixel_lmean = ax_pixel.axhline(0.0, color="#9467bd", linestyle="-", lw=1.5, label="Low Plateau Mean")
line_pixel_med   = ax_pixel.axhline(0.0, color="#d62728", linestyle="--", lw=1.2, label="Median Threshold")

ax_pixel.set_xlabel("Pixel Acquisition Time (ms) — [16 Chopper Cycles / 320 Samples @ 20 kHz]", fontsize=11, fontweight="bold")
ax_pixel.set_ylabel("Sampled Voltage (V)", fontsize=10.5, fontweight="bold")
ax_pixel.set_ylim(-5.0, 5.0)
ax_pixel.grid(True, linestyle="--", alpha=0.6)
ax_pixel.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=3)
ax_pixel.set_title("(3) Per-Pixel Demodulation Emulator (320 Samples / 16.0 ms) | Initializing...", fontsize=11, fontweight="bold", color="#1b5e20")

plt.tight_layout()


# ==============================================================================
# 6. REAL-TIME ANIMATION UPDATE FUNCTION
# ==============================================================================
def update(frame):
    global cached_laser, cached_chop, frame_count
    frame_count += 1

    try:
        avail = task.in_stream.avail_samp_per_chan
        if avail > 0:
            raw = np.array(task.read(number_of_samples_per_channel=avail))
            shift = raw.shape[1]
            with cache_lock:
                if shift >= MAX_CACHE_LEN:
                    cached_laser = raw[0, -MAX_CACHE_LEN:]
                    cached_chop = raw[1, -MAX_CACHE_LEN:]
                else:
                    cached_laser = np.roll(cached_laser, -shift)
                    cached_chop = np.roll(cached_chop, -shift)
                    cached_laser[-shift:] = raw[0]
                    cached_chop[-shift:] = raw[1]
    except Exception as read_err:
        return line_laser, line_chop

    with cache_lock:
        scope_laser = cached_laser[-CHUNK_SIZE:].copy()
        scope_chop = cached_chop[-CHUNK_SIZE:].copy()
        pixel_raw_100k = cached_laser[-RAW_SAMPLES_IN_PIXEL:].copy()

    # Update Subplot 1 & 2 waveforms
    line_laser.set_ydata(scope_laser)
    line_chop.set_ydata(scope_chop)

    # --- Subplot 3: Extract & Downsample Exact 320 Samples (16.0 ms @ 20 kHz) ---
    samples_320 = pixel_raw_100k[::downsample_step][:SCAN_SAMPLES_PER_POINT]
    
    if len(samples_320) == SCAN_SAMPLES_PER_POINT:
        med_pixel = float(np.median(samples_320))
        high_mask = samples_320 > med_pixel
        low_mask = samples_320 <= med_pixel

        high_pts = samples_320[high_mask]
        low_pts = samples_320[low_mask]

        line_pixel_wave.set_ydata(samples_320)
        line_pixel_high.set_data(t_pixel_ms[high_mask], high_pts)
        line_pixel_low.set_data(t_pixel_ms[low_mask], low_pts)

        h_mean = float(high_pts.mean()) if high_pts.size else 0.0
        l_mean = float(low_pts.mean()) if low_pts.size else 0.0
        pixel_val = float(h_mean - l_mean)

        line_pixel_hmean.set_ydata([h_mean, h_mean])
        line_pixel_lmean.set_ydata([l_mean, l_mean])
        line_pixel_med.set_ydata([med_pixel, med_pixel])

    # Periodic metrics & auto-scale update (every 3 frames)
    if frame_count % 3 == 0:
        l_min, l_max = scope_laser.min(), scope_laser.max()
        c_min, c_max = scope_chop.min(), scope_chop.max()

        margin_l = max(0.3, (l_max - l_min) * 0.18)
        ax_laser.set_ylim(l_min - margin_l, l_max + margin_l)

        margin_c = max(0.5, (c_max - c_min) * 0.18)
        ax_chop.set_ylim(min(-0.5, c_min - margin_c), max(5.5, c_max + margin_c))

        if len(samples_320) == SCAN_SAMPLES_PER_POINT:
            p_min, p_max = samples_320.min(), samples_320.max()
            margin_p = max(0.3, (p_max - p_min) * 0.18)
            ax_pixel.set_ylim(p_min - margin_p, p_max + margin_p)

        # Subplot 1 Header: Laser Signal + Live Scanning Progress
        med_scope = float(np.median(scope_laser))
        line_scope_med.set_ydata([med_scope, med_scope])

        high_vals = scope_laser[scope_laser > med_scope]
        low_vals = scope_laser[scope_laser <= med_scope]

        if high_vals.size and low_vals.size:
            diff_v = float(high_vals.mean() - low_vals.mean())
            vpp = float(l_max - l_min)
            dc_mean = float(scope_laser.mean())
            is_clipped = bool(np.any(np.abs(scope_laser) > 9.8))
            clip_warn = " [RAIL CLIPPING!]" if is_clipped else ""

            # Display progress tag in title
            if scan_status["finished"]:
                prog_str = "[SCAN COMPLETE] "
            elif scan_status["running"]:
                prog_str = f"[SCAN Row {scan_status['row']}/{NUM_Y} ({scan_status['pct']:.1f}%) | Rem: {scan_status['rem_min']:.1f}m] "
            else:
                prog_str = "[INITIALIZING STAGES] "

            ax_laser.set_title(
                f"{prog_str}Laser (ai7) | High-Low: {diff_v:.3f} V | Vpp: {vpp:.2f} V | DC: {dc_mean:+.2f} V{clip_warn}",
                fontsize=10.0,
                fontweight="bold",
                color="#b30000" if is_clipped else "#0b5394",
            )

        # Subplot 2 Header: Chopper Frequency
        c_mid = (c_max + c_min) / 2.0
        binary_chop = (scope_chop > c_mid).astype(int)
        rising_edges = np.where(np.diff(binary_chop) == 1)[0]

        if len(rising_edges) >= 2 and (c_max - c_min) > 0.8:
            periods = np.diff(rising_edges) / DAQ_RATE
            freq_hz = float(1.0 / np.mean(periods))
            duty_pct = float(np.mean(binary_chop) * 100.0)
            ax_chop.set_title(
                f"Chopper (ai15) | Freq: {freq_hz:.1f} Hz (Period: {1000.0/freq_hz:.2f} ms) | Duty: {duty_pct:.1f}% | TTL: {c_min:.1f}V – {c_max:.1f}V",
                fontsize=10.0,
                fontweight="bold",
                color="#b45f06",
            )
        else:
            ax_chop.set_title(
                f"Chopper (ai15) | Syncing... | Span: {c_max - c_min:.2f} V",
                fontsize=10.0,
                fontweight="bold",
                color="#666666",
            )

        # Subplot 3 Header: Pixel Demodulation Result
        if len(samples_320) == SCAN_SAMPLES_PER_POINT:
            p_vpp = float(samples_320.max() - samples_320.min())
            ax_pixel.set_title(
                f"Per-Pixel Demodulation (320 samples @ 20 kHz) | Image Pixel Voltage ΔV = {pixel_val:.3f} V | High: {h_mean:.2f}V, Low: {l_mean:.2f}V | Vpp: {p_vpp:.2f}V",
                fontsize=10.0,
                fontweight="bold",
                color="#1b5e20",
            )

    return line_laser, line_chop, line_pixel_wave


ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)

try:
    plt.show()
finally:
    try:
        task.stop()
        task.close()
        print("\n[OK] DAQ Task stopped and closed cleanly.")
    except Exception:
        pass