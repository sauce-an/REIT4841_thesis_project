"""
Automated Batch Scanning Script: 100 um Scan followed immediately by 50 um Scan.
Automatically logs results to notes.txt and pushes results to GitHub upon completion.

Sequence:
  1. 100 um scan (scan_4) — 215 x 206 = 44,290 pixels
  2. 50 um scan (scan_6)  — 430 x 411 = 176,730 pixels
  3. Automatic Git commit and push

CLOSE ZABER CONSOLE AND NI MAX BEFORE RUNNING THIS SCRIPT.
"""
import sys
import os
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from zaber_motion import Units, Library
from zaber_motion.binary import Connection
import nidaqmx
from nidaqmx.constants import TerminalConfiguration

# Unbuffered terminal output
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
NOTES_PATH = os.path.join(SCRIPT_DIR, "notes.txt")
os.makedirs(RESULTS_DIR, exist_ok=True)

Library.enable_device_db_store()

SERIAL_PORT = "COM5"
Y_DEVICE = 4  # Up / Down
Z_DEVICE = 5  # Forward / Backward (Focus)
X_DEVICE = 6  # Left / Right

# Calibrated Scan Area
X_START_NATIVE = 890000    # Left boundary (~21.4 mm span)
X_END_NATIVE   = 440000    # Right boundary
Y_START_NATIVE = 475000    # Bottom boundary (~20.5 mm span)
Y_END_NATIVE   = 905000    # Top boundary
Z_FOCUS_NATIVE = 103116    # Calibrated Z focus depth (~4.911 mm)

MICROSTEPS_PER_UM = 1.0 / 0.047625  # ~20.9974 steps per um

# DAQ Acquisition Settings
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"
CHOPPER_FREQ_HZ = 1000
PERIODS_PER_POINT = 32             # 32 chopper periods (32.0 ms per pixel)
DAQ_RATE = 20_000                  # 20 kHz DAQ rate
DAQ_SAMPLES_PER_POINT = int(DAQ_RATE * PERIODS_PER_POINT / CHOPPER_FREQ_HZ)  # 640 samples
SATURATION_THRESHOLD_V = 9.8


def extract_confocal_signal(raw_samples: np.ndarray, saturation_v: float = 9.8) -> tuple[float, bool]:
    """Demodulate AC-coupled chopper-modulated waveform via High-Low Median Split with Software AC Coupling."""
    data = np.asarray(raw_samples)
    is_saturated = bool(np.all(np.abs(data) > saturation_v) or np.ptp(data) < 1e-4)
    if is_saturated:
        return np.nan, True
    
    # Software AC Coupling: remove DC offset to center waveform symmetrically at 0.0V
    ac_data = data - np.mean(data)
    
    threshold = np.median(ac_data)
    high_vals = ac_data[ac_data > threshold]
    low_vals = ac_data[ac_data <= threshold]
    if high_vals.size and low_vals.size:
        return float(high_vals.mean() - low_vals.mean()), False
    return np.nan, True


def run_single_scan(target_step_um: float, scan_label: str) -> dict:
    """Execute a single 2D raster scan at specified step size."""
    step_native = target_step_um * MICROSTEPS_PER_UM
    num_x = int(round(abs(X_END_NATIVE - X_START_NATIVE) / step_native)) + 1
    num_y = int(round(abs(Y_END_NATIVE - Y_START_NATIVE) / step_native)) + 1

    x_coords = np.linspace(X_START_NATIVE, X_END_NATIVE, num_x, dtype=int)
    y_coords = np.linspace(Y_START_NATIVE, Y_END_NATIVE, num_y, dtype=int)

    actual_x_step_um = abs(x_coords[1] - x_coords[0]) * 0.047625
    actual_y_step_um = abs(y_coords[1] - y_coords[0]) * 0.047625
    scan_width_mm = abs(X_END_NATIVE - X_START_NATIVE) * 0.047625 / 1000.0
    scan_height_mm = abs(Y_END_NATIVE - Y_START_NATIVE) * 0.047625 / 1000.0

    print("\n" + "=" * 75)
    print(f"STARTING SCAN: {int(target_step_um)} µm Resolution ({scan_label})")
    print("=" * 75)
    print(f"  Grid Dimensions : {scan_width_mm:.2f} mm (X) x {scan_height_mm:.2f} mm (Y)")
    print(f"  Grid Resolution : {num_x} (X) x {num_y} (Y) = {num_x * num_y:,} total pixels")
    print(f"  Actual Step Size: {actual_x_step_um:.1f} µm (X) x {actual_y_step_um:.1f} µm (Y)")
    print(f"  Z Focus Depth   : {Z_FOCUS_NATIVE} native units (~{Z_FOCUS_NATIVE*0.047625/1000.0:.3f} mm)")
    print(f"  Averaging Dwell : {PERIODS_PER_POINT} periods ({DAQ_SAMPLES_PER_POINT} samples @ {DAQ_RATE:,} Hz / 32.0 ms)")
    print("=" * 75, flush=True)

    image_data = np.full((num_y, num_x), np.nan, dtype=np.float64)
    start_time = time.time()
    saturated_points_count = 0

    with Connection.open_serial_port(SERIAL_PORT) as connection:
        connection.detect_devices(identify_devices=True)
        y_axis = connection.get_device(Y_DEVICE)
        x_axis = connection.get_device(X_DEVICE)
        z_axis = connection.get_device(Z_DEVICE)

        print(f"[OK] Connected to Zaber stages. Moving Z focus to {Z_FOCUS_NATIVE}...", flush=True)
        z_axis.move_absolute(Z_FOCUS_NATIVE)
        print(f"[OK] Z positioned at {z_axis.get_position():.0f} native units.", flush=True)

        print(f"Moving to Start Corner: X={X_START_NATIVE}, Y={Y_START_NATIVE}...", flush=True)
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages aligned at start position.\n", flush=True)

        with nidaqmx.Task() as daq_task:
            daq_task.ai_channels.add_ai_voltage_chan(LASER_CHANNEL, terminal_config=TerminalConfiguration.RSE)
            daq_task.timing.cfg_samp_clk_timing(rate=DAQ_RATE, samps_per_chan=DAQ_SAMPLES_PER_POINT)
            print(f"[OK] DAQ Task initialized on {LASER_CHANNEL}.\n", flush=True)

            print(f">>> SCANNING {int(target_step_um)} µm GRID ({num_y} ROWS) <<<", flush=True)

            for row_idx in range(num_y):
                row_y_pos = int(y_coords[row_idx])
                y_axis.move_absolute(row_y_pos)

                is_even_row = (row_idx % 2 == 0)
                col_indices = list(range(num_x) if is_even_row else range(num_x - 1, -1, -1))

                for col_idx in col_indices:
                    col_x_pos = int(x_coords[col_idx])
                    x_axis.move_absolute(col_x_pos)

                    time.sleep(0.008)  # Settling delay

                    raw_samples = daq_task.read(number_of_samples_per_channel=DAQ_SAMPLES_PER_POINT)
                    confocal_val, is_sat = extract_confocal_signal(raw_samples, saturation_v=SATURATION_THRESHOLD_V)

                    image_data[row_idx, col_idx] = confocal_val
                    if is_sat:
                        saturated_points_count += 1

                # Checkpoint saving every 10 rows
                if (row_idx + 1) % 10 == 0:
                    chk_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(target_step_um)}um_{scan_label}_checkpoint.npy")
                    np.save(chk_path, image_data)

                elapsed = time.time() - start_time
                rows_done = row_idx + 1
                est_total = (elapsed / rows_done) * num_y
                est_remaining = max(0.0, est_total - elapsed)
                pct = (rows_done / num_y) * 100.0

                row_valid = image_data[row_idx][~np.isnan(image_data[row_idx])]
                row_str = f"[{row_valid.min():.3f}V, {row_valid.max():.3f}V]" if row_valid.size else "[ALL SAT/NAN]"

                print(f"  Row {rows_done:3d}/{num_y} ({pct:5.1f}%) | "
                      f"Elapsed: {elapsed / 60:.1f}m | "
                      f"Remaining: {est_remaining / 60:.1f}m | "
                      f"Last Row: {row_str} | Sat: {saturated_points_count}", flush=True)

            total_scan_time = time.time() - start_time
            print(f"\n[DONE] {int(target_step_um)} µm scan finished in {total_scan_time / 60:.2f} minutes!", flush=True)

        print("Returning stages to start position...", flush=True)
        x_axis.move_absolute(X_START_NATIVE)
        y_axis.move_absolute(Y_START_NATIVE)
        print("[OK] Stages returned to start.", flush=True)

    # Save output files
    npy_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(target_step_um)}um_{scan_label}.npy")
    csv_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(target_step_um)}um_{scan_label}.csv")
    png_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(target_step_um)}um_{scan_label}.png")
    chk_path = os.path.join(RESULTS_DIR, f"coin_scan_{int(target_step_um)}um_{scan_label}_checkpoint.npy")

    np.save(npy_path, image_data)
    np.savetxt(csv_path, image_data, delimiter=",", fmt="%.6e")
    if os.path.exists(chk_path):
        try:
            os.remove(chk_path)
        except Exception:
            pass

    valid_pixels = image_data[~np.isnan(image_data)]
    if valid_pixels.size:
        vmin = float(np.percentile(valid_pixels, 0.5))
        vmax = float(np.percentile(valid_pixels, 95.5))
        if vmax - vmin < 1e-3:
            vmin, vmax = float(np.min(valid_pixels)), float(np.max(valid_pixels))
        raw_min_str = f"{np.min(valid_pixels):.3f} V"
        raw_max_str = f"{np.max(valid_pixels):.3f} V"
    else:
        vmin, vmax = 0.0, 1.0
        raw_min_str, raw_max_str = "N/A", "N/A"

    sat_pct = (saturated_points_count / (num_x * num_y)) * 100.0

    # Thread-safe headless figure export with white saturation mask
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
    ax_out.set_title(f"LFI 2D Raster Scan — Coin Target\n({scan_width_mm:.2f} x {scan_height_mm:.2f} mm, ~{int(target_step_um)} µm step | High-Low Demod)", fontsize=12, fontweight="bold")
    ax_out.set_xlabel("X Position (mm)", fontsize=11, fontweight="bold")
    ax_out.set_ylabel("Y Position (mm)", fontsize=11, fontweight="bold")
    fig_save.tight_layout()
    fig_save.savefig(png_path, dpi=300)

    print(f"  - Saved raw matrix to: {npy_path}")
    print(f"  - Saved CSV data to  : {csv_path}")
    print(f"  - Saved image figure : {png_path}")
    print(f"  - Saturated points   : {saturated_points_count:,} / {num_x * num_y:,} ({sat_pct:.2f}%)")
    print(f"  - Signal range       : [{raw_min_str}, {raw_max_str}]", flush=True)

    return {
        "step_um": target_step_um,
        "label": scan_label,
        "num_x": num_x,
        "num_y": num_y,
        "total_pixels": num_x * num_y,
        "duration_min": total_scan_time / 60.0,
        "sat_count": saturated_points_count,
        "sat_pct": sat_pct,
        "raw_min": raw_min_str,
        "raw_max": raw_max_str,
        "vmin": vmin,
        "vmax": vmax,
    }


def append_notes_and_push(results_50um: dict, results_100um: dict):
    """Log findings into notes.txt and execute git push."""
    print("\n" + "=" * 75)
    print("UPDATING NOTES.TXT & PUSHING TO GITHUB")
    print("=" * 75, flush=True)

    timestamp_str = time.strftime("%Y-%m-%d")
    notes_entry = f"""
16. 50 µm — {results_50um['label']} ({timestamp_str}):
    - Grid: {results_50um['num_x']} (X) x {results_50um['num_y']} (Y) = {results_50um['total_pixels']:,} points
    - Focus: Z = {Z_FOCUS_NATIVE} native units (~{Z_FOCUS_NATIVE*0.047625/1000.0:.3f} mm)
    - Configuration: High-Low Median-Split Demodulation with Software AC Coupling, 1000 Hz Chopper, 20 kHz DAQ, 32 periods avg (640 samples / 32.0 ms), 0.5%–95.5% Colormap Percentiles
    - Duration: {results_50um['duration_min']:.2f} minutes
    - Status: SUCCESSFUL
    - Saturated/Rail points: {results_50um['sat_count']:,} / {results_50um['total_pixels']:,} ({results_50um['sat_pct']:.2f}%)
    - Valid raw signal range: [{results_50um['raw_min']}, {results_50um['raw_max']}]
    - Adaptive display range: [{results_50um['vmin']:.3f} V, {results_50um['vmax']:.3f} V]
    - Files: coin_scan_50um_{results_50um['label']}.*

17. 100 µm — {results_100um['label']} ({timestamp_str}):
    - Grid: {results_100um['num_x']} (X) x {results_100um['num_y']} (Y) = {results_100um['total_pixels']:,} points
    - Focus: Z = {Z_FOCUS_NATIVE} native units (~{Z_FOCUS_NATIVE*0.047625/1000.0:.3f} mm)
    - Configuration: High-Low Median-Split Demodulation with Software AC Coupling, 1000 Hz Chopper, 20 kHz DAQ, 32 periods avg (640 samples / 32.0 ms), 0.5%–95.5% Colormap Percentiles
    - Duration: {results_100um['duration_min']:.2f} minutes
    - Status: SUCCESSFUL
    - Saturated/Rail points: {results_100um['sat_count']:,} / {results_100um['total_pixels']:,} ({results_100um['sat_pct']:.2f}%)
    - Valid raw signal range: [{results_100um['raw_min']}, {results_100um['raw_max']}]
    - Adaptive display range: [{results_100um['vmin']:.3f} V, {results_100um['vmax']:.3f} V]
    - Files: coin_scan_100um_{results_100um['label']}.*
"""
    try:
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            existing_notes = f.read()

        insert_marker = "================================================================================\nANALYZED BENCHMARK FIGURES"
        if insert_marker in existing_notes:
            parts = existing_notes.split(insert_marker)
            updated_notes = parts[0].rstrip() + "\n" + notes_entry + "\n" + insert_marker + parts[1]
        else:
            updated_notes = existing_notes + "\n" + notes_entry

        with open(NOTES_PATH, "w", encoding="utf-8") as f:
            f.write(updated_notes)
        print("[OK] notes.txt updated with scan 16 (50 µm) and scan 17 (100 µm) metrics.", flush=True)
    except Exception as e:
        print(f"[WARNING] Could not update notes.txt: {e}", flush=True)

    # Git commit and push
    try:
        env = os.environ.copy()
        git_add_cmd = [
            "git", "add",
            os.path.join(RESULTS_DIR, f"coin_scan_50um_{results_50um['label']}.*"),
            os.path.join(RESULTS_DIR, f"coin_scan_100um_{results_100um['label']}.*"),
            NOTES_PATH,
        ]
        subprocess.run(git_add_cmd, cwd=PROJECT_DIR, check=True, env=env)
        commit_msg = f"Add 50um ({results_50um['label']}) and 100um ({results_100um['label']}) scan results with Software AC Coupling and 0.5-95.5% scaling"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_DIR, check=True, env=env)
        subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_DIR, check=True, env=env)
        print("[SUCCESS] All results and notes successfully committed and pushed to GitHub!", flush=True)
    except Exception as push_err:
        print(f"[ERROR] Git push failed: {push_err}", flush=True)


if __name__ == "__main__":
    print("=" * 75)
    print("SEQUENTIAL BATCH SCAN RUNNER: 50 µm (scan_7) -> 100 µm (scan_5) -> Git Push")
    print(f"Focus Position: Z = {Z_FOCUS_NATIVE} native units (~4.911 mm)")
    print(f"Periods / Point: {PERIODS_PER_POINT} (640 samples @ 20 kHz / 32.0 ms)")
    print("=" * 75, flush=True)

    # Phase 1: 50 um scan (scan_7)
    res_50 = run_single_scan(target_step_um=50.0, scan_label="scan_7")

    # Phase 2: 100 um scan (scan_5)
    res_100 = run_single_scan(target_step_um=100.0, scan_label="scan_5")

    # Phase 3: Update notes & push
    append_notes_and_push(res_50, res_100)

    print("\n" + "=" * 75)
    print("ALL BATCH TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 75, flush=True)
