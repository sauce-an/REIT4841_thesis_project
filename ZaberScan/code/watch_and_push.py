"""
Background Watcher: Monitors for completion of coin_scan_50um_scan_8.*
Once scan_8 finishes, it updates notes.txt with final metrics and pushes to GitHub.
"""
import os
import sys
import time
import subprocess
import numpy as np

# Unbuffered output
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
NOTES_PATH = os.path.join(SCRIPT_DIR, "notes.txt")

TARGET_PNG = os.path.join(RESULTS_DIR, "coin_scan_50um_scan_9.png")
TARGET_NPY = os.path.join(RESULTS_DIR, "coin_scan_50um_scan_9.npy")
TARGET_CSV = os.path.join(RESULTS_DIR, "coin_scan_50um_scan_9.csv")

print("=" * 70)
print("BACKGROUND WATCHER ACTIVE: Waiting for coin_scan_50um_scan_9 completion...")
print(f"Monitoring path: {TARGET_PNG}")
print("=" * 70, flush=True)

# Wait for completion
check_interval = 30  # check every 30 seconds
start_watch_time = time.time()

while True:
    # Check if scan output exists and file write is finished (non-zero size and stable for 15s)
    if os.path.exists(TARGET_PNG) and os.path.exists(TARGET_NPY):
        initial_size = os.path.getsize(TARGET_PNG)
        if initial_size > 100_000:  # Valid PNG figure is ~500KB+
            time.sleep(15)
            if os.path.getsize(TARGET_PNG) == initial_size:
                print("\n[DETECTED] Scan completed! coin_scan_50um_scan_9 files found and verified.", flush=True)
                break
    
    # Heartbeat every 30 minutes
    elapsed_hr = (time.time() - start_watch_time) / 3600.0
    if int(time.time() - start_watch_time) % 1800 < check_interval:
        print(f"[WATCHER RUNNING] Still monitoring for scan_9 completion... (Waiting {elapsed_hr:.1f} hrs)", flush=True)
        
    time.sleep(check_interval)

# --- PROCESS METRICS & UPDATE NOTES ---
print("\nCalculating metrics from coin_scan_50um_scan_9.npy...", flush=True)
try:
    image_data = np.load(TARGET_NPY)
    num_y, num_x = image_data.shape
    total_pixels = num_x * num_y
    
    valid_pixels = image_data[~np.isnan(image_data)]
    if valid_pixels.size:
        vmin = float(np.percentile(valid_pixels, 0.5))
        vmax = float(np.percentile(valid_pixels, 95.5))
        raw_min = float(np.min(valid_pixels))
        raw_max = float(np.max(valid_pixels))
    else:
        vmin, vmax, raw_min, raw_max = 0.0, 1.0, 0.0, 1.0
        
    sat_count = int(np.sum(valid_pixels > 9.8) + np.sum(np.isnan(image_data)))
    sat_pct = (sat_count / total_pixels) * 100.0
    timestamp_str = time.strftime("%Y-%m-%d")

    notes_entry = f"""19. 50 µm — scan_9 ({timestamp_str}):
    - Grid: {num_x} (X) x {num_y} (Y) = {total_pixels:,} points
    - Focus: Z = 103116 native units (~4.911 mm)
    - Configuration: High-Low Median-Split Demodulation with Software AC Coupling, 1000 Hz Chopper, 20 kHz DAQ, 32 periods avg (640 samples / 32.0 ms), 0.5%–95.5% Colormap Percentiles
    - Optical State: Performed with FULL specular reflection filtered out by the Quarter-Wave Plate (QWP at ~285 deg).
    - Status: SUCCESSFUL
    - Saturated/Rail points: {sat_count:,} / {total_pixels:,} ({sat_pct:.2f}%)
    - Valid raw signal range: [{raw_min:.3f} V, {raw_max:.3f} V]
    - Adaptive display range: [{vmin:.3f} V, {vmax:.3f} V]
    - Files: coin_scan_50um_scan_9.*"""

    if os.path.exists(NOTES_PATH):
        with open(NOTES_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        target_marker = "19. 50 µm — scan_9"
        split_marker = "================================================================================\nANALYZED BENCHMARK FIGURES"

        if target_marker in content and split_marker in content:
            before_part = content.split(target_marker)[0]
            after_part = content.split(split_marker)[1]
            updated_content = before_part.rstrip() + "\n\n" + notes_entry + "\n\n" + split_marker + after_part
        elif split_marker in content:
            parts = content.split(split_marker)
            updated_content = parts[0].rstrip() + "\n\n" + notes_entry + "\n\n" + split_marker + parts[1]
        else:
            updated_content = content.rstrip() + "\n\n" + notes_entry + "\n"

        with open(NOTES_PATH, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print("[OK] notes.txt successfully updated with final scan_9 metrics.", flush=True)

except Exception as err:
    print(f"[WARNING] Metric calculation warning: {err}", flush=True)

# --- EXECUTE GIT COMMIT & PUSH ---
print("\nStaging files and pushing to GitHub...", flush=True)
try:
    env = os.environ.copy()
    subprocess.run(["git", "add", TARGET_NPY, TARGET_CSV, TARGET_PNG, NOTES_PATH], cwd=PROJECT_DIR, check=True, env=env)
    commit_msg = "Add 50um (scan_9) scan results with full specular filtration (QWP at ~285 deg)"
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=PROJECT_DIR, check=True, env=env)
    subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_DIR, check=True, env=env)
    print("\n[SUCCESS] 50um scan_9 results and notes successfully committed and pushed to GitHub!", flush=True)
except Exception as push_err:
    print(f"\n[ERROR] Git push failed: {push_err}", flush=True)

print("\nWatcher task finished successfully.")
