"""
================================================================================
NI-DAQmx LIVE 3-SUBPLOT OSCILLOSCOPE & PER-PIXEL SIGNAL EMULATOR
================================================================================
Purpose:
  Real-time 3-channel diagnostic monitor for optical bench alignment and
  demodulation inspection:
    - Subplot 1 (Top)    : Laser Terminal AC-Coupled Confocal LFI Signal (Dev1/ai7)
    - Subplot 2 (Middle) : Optical Chopper Reference / Sync Signal (Dev1/ai15)
    - Subplot 3 (Bottom) : 320-Sample Per-Pixel Acquisition & Demodulation Emulator
                           (Exact 20 kHz, 16-period median-split view as in scan_tester1.py)

Features:
  - Real-time continuous non-stopping waveform display (replaces NI MAX)
  - Configurable on-screen period counts and nominal chopper frequency
  - Automatic Chopper Frequency & Duty Cycle detection from Channel 15 TTL edges
  - Zero-baseline reference lines (0V) to monitor DC offset / centering in real time
  - Live Demodulation breakdown on Subplot 3: shows discrete 320 samples,
    High Plateau mean, Low Plateau mean, and resulting pixel intensity ΔV
  - Buffer-drain architecture: zero UI freezing, runs indefinitely without lag

Usage:
  Run in terminal: python ZaberScan/code/daq_live.py
  (Ensure NI MAX is closed before launching)
================================================================================
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration

# ==============================================================================
# USER CONFIGURATION PARAMETERS (ADJUST AS NEEDED)
# ==============================================================================
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"       # Channel 7: Laser terminal AC-coupled signal
CHOPPER_CHANNEL = f"{DEVICE_NAME}/ai15"    # Channel 15: Optical chopper TTL reference
DAQ_RATE = 100_000                         # Live scope sampling rate (100 kS/s per channel)
EST_CHOPPER_FREQ = 1000                    # Nominal chopper frequency (1000 Hz)
DISPLAY_PERIODS = 10                       # Number of chopper cycles on top 2 subplots (10 ms)
AUTO_DETECT_CHOPPER_FREQ = True            # Auto-measure chopper frequency from TTL edges

# --- Subplot 3: Per-Pixel Acquisition Settings (Matching scan_tester1.py) ---
SCAN_DAQ_RATE = 20_000                     # 20 kHz sampling rate in scanning script
SCAN_PERIODS_PER_POINT = 16                # 32 chopper periods per pixel (32.0 ms)
SCAN_SAMPLES_PER_POINT = int(SCAN_DAQ_RATE * SCAN_PERIODS_PER_POINT / EST_CHOPPER_FREQ)  # 640 samples (32.0 ms)
SCAN_DURATION_SEC = SCAN_PERIODS_PER_POINT / EST_CHOPPER_FREQ                             # 0.032 s (32.0 ms)
RAW_SAMPLES_IN_PIXEL = int(DAQ_RATE * SCAN_DURATION_SEC)                                  # 3200 samples @ 100 kHz

# Buffer & Window Setup
CHUNK_SIZE = int(DAQ_RATE * DISPLAY_PERIODS / EST_CHOPPER_FREQ)  # 1000 samples @ 100 kHz (10.0 ms)
BUFFER_SIZE = 200_000                      # 2-second circular hardware buffer

print("=" * 75)
print("STARTING NI-DAQmx LIVE 3-SUBPLOT OSCILLOSCOPE & SIGNAL EMULATOR")
print("=" * 75)
print(f"  Subplot 1 (Top)    : {LASER_CHANNEL}  (Laser AC LFI @ 100 kS/s)")
print(f"  Subplot 2 (Middle) : {CHOPPER_CHANNEL} (Optical Chopper TTL @ 100 kS/s)")
print(f"  Subplot 3 (Bottom) : Per-Pixel Emulator ({SCAN_SAMPLES_PER_POINT} samples @ {SCAN_DAQ_RATE:,} Hz / {SCAN_DURATION_SEC*1000:.1f} ms)")
print(f"  Chopper Frequency  : {EST_CHOPPER_FREQ:,} Hz nominal (Auto-detect: {AUTO_DETECT_CHOPPER_FREQ})")
print("=" * 75)

# ==============================================================================
# DAQ HARDWARE INITIALIZATION
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
    print("[OK] NI-DAQ Continuous Task running.\n")
except Exception as e:
    print(f"\n[ERROR] Failed to initialize DAQ hardware: {e}")
    print("Ensure NI MAX is closed and USB-6451 is connected.")
    try:
        task.close()
    except Exception:
        pass
    sys.exit(1)

# ==============================================================================
# SET UP 3-SUBPLOT GUI
# ==============================================================================
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax_laser, ax_chop, ax_pixel) = plt.subplots(3, 1, figsize=(12, 9.5))
fig.canvas.manager.set_window_title("NI-DAQ Live Monitor — Laser (ai7), Chopper (ai15), & Pixel Demodulation")

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

# --- SUBPLOT 3: Single-Period Phase-Averaged Waveform (Averaged across all 32 cycles) ---
SAMPLES_PER_CYCLE = int(SCAN_DAQ_RATE / EST_CHOPPER_FREQ)  # 20 samples per 1 ms cycle
t_cycle_ms = np.arange(SAMPLES_PER_CYCLE) / SCAN_DAQ_RATE * 1000  # 0.0 to 0.95 ms

line_pixel_avg,   = ax_pixel.plot(t_cycle_ms, np.zeros(SAMPLES_PER_CYCLE), '-o', color="#1b5e20", lw=2.4, markersize=5.5, label=f"Averaged Cycle ({SCAN_PERIODS_PER_POINT}× Ensemble Avg)")
line_pixel_hmean  = ax_pixel.axhline(0.0, color="#2ca02c", linestyle="-", lw=1.6, label="High Plateau Mean")
line_pixel_lmean  = ax_pixel.axhline(0.0, color="#9467bd", linestyle="-", lw=1.6, label="Low Plateau Mean")
line_pixel_med    = ax_pixel.axhline(0.0, color="#d62728", linestyle="--", lw=1.2, label="Median Threshold")

ax_pixel.set_xlabel("Chopper Cycle Phase Time (ms) — [1.0 ms Single Period]", fontsize=11, fontweight="bold")
ax_pixel.set_ylabel("Laser Voltage (V)", fontsize=10.5, fontweight="bold")
ax_pixel.set_ylim(-5.0, 5.0)
ax_pixel.set_xlim(0, 1.0)
ax_pixel.grid(True, linestyle="--", alpha=0.6)
ax_pixel.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=4)
ax_pixel.set_title(f"(3) Averaged Waveform ({SCAN_PERIODS_PER_POINT} Periods Folded) | Initializing...", fontsize=11, fontweight="bold", color="#1b5e20")

plt.tight_layout()

# Rolling caches for continuous streaming
MAX_CACHE_LEN = max(CHUNK_SIZE, RAW_SAMPLES_IN_PIXEL)
cached_laser = np.zeros(MAX_CACHE_LEN)
cached_chop = np.zeros(MAX_CACHE_LEN)
frame_count = 0
downsample_step = int(DAQ_RATE / SCAN_DAQ_RATE)  # 100 kHz / 20 kHz = 5


# ==============================================================================
# REAL-TIME ANIMATION UPDATE FUNCTION
# ==============================================================================
def update(frame):
    global cached_laser, cached_chop, frame_count
    frame_count += 1

    try:
        # Check available samples in hardware buffer
        avail = task.in_stream.avail_samp_per_chan
        if avail > 0:
            # Drain buffer fully to avoid buffer overflows and latency lag
            raw = np.array(task.read(number_of_samples_per_channel=avail))
            shift = raw.shape[1]
            
            if shift >= MAX_CACHE_LEN:
                cached_laser = raw[0, -MAX_CACHE_LEN:]
                cached_chop = raw[1, -MAX_CACHE_LEN:]
            else:
                cached_laser = np.roll(cached_laser, -shift)
                cached_chop = np.roll(cached_chop, -shift)
                cached_laser[-shift:] = raw[0]
                cached_chop[-shift:] = raw[1]
    except Exception as read_err:
        if frame_count % 60 == 0:
            print(f"[Notice] Read warning: {read_err}")
        return line_laser, line_chop

    # Extract scope window (10.0 ms) & Software AC-Couple
    scope_laser_raw = cached_laser[-CHUNK_SIZE:]
    scope_dc = float(np.mean(scope_laser_raw))
    scope_laser_ac = scope_laser_raw - scope_dc  # Software AC-Coupled (0V centered)
    scope_chop = cached_chop[-CHUNK_SIZE:]

    # Update Subplot 1 & 2 waveforms
    line_laser.set_ydata(scope_laser_ac)
    line_chop.set_ydata(scope_chop)

    # --- Subplot 3: Extract, Downsample, Software AC-Couple & Ensemble-Average Across 32 Cycles ---
    pixel_raw_100k = cached_laser[-RAW_SAMPLES_IN_PIXEL:]
    samples_raw = pixel_raw_100k[::downsample_step][:SCAN_SAMPLES_PER_POINT]
    
    if len(samples_raw) == SCAN_SAMPLES_PER_POINT:
        # Software AC Coupling for pixel
        pixel_dc = float(np.mean(samples_raw))
        samples_ac = samples_raw - pixel_dc

        # Fold into 32 periods x 20 samples matrix
        laser_matrix = samples_ac[:SCAN_PERIODS_PER_POINT * SAMPLES_PER_CYCLE].reshape((SCAN_PERIODS_PER_POINT, SAMPLES_PER_CYCLE))
        avg_cycle = np.mean(laser_matrix, axis=0)

        med_pixel = float(np.median(samples_ac))
        high_mask = samples_ac > med_pixel
        low_mask = samples_ac <= med_pixel

        high_pts = samples_ac[high_mask]
        low_pts = samples_ac[low_mask]

        line_pixel_avg.set_ydata(avg_cycle)

        h_mean = float(high_pts.mean()) if high_pts.size else 0.0
        l_mean = float(low_pts.mean()) if low_pts.size else 0.0
        pixel_val = float(h_mean - l_mean)

        line_pixel_hmean.set_ydata([h_mean, h_mean])
        line_pixel_lmean.set_ydata([l_mean, l_mean])
        line_pixel_med.set_ydata([med_pixel, med_pixel])

    # Periodic metrics & auto-scale update (every 3 frames for high responsiveness)
    if frame_count % 3 == 0:
        l_min, l_max = scope_laser_ac.min(), scope_laser_ac.max()
        c_min, c_max = scope_chop.min(), scope_chop.max()

        # Dynamic vertical auto-scaling for Subplots 1 & 2
        margin_l = max(0.3, (l_max - l_min) * 0.18)
        ax_laser.set_ylim(l_min - margin_l, l_max + margin_l)

        margin_c = max(0.5, (c_max - c_min) * 0.18)
        ax_chop.set_ylim(min(-0.5, c_min - margin_c), max(5.5, c_max + margin_c))

        # Auto-scale Subplot 3
        if len(samples_raw) == SCAN_SAMPLES_PER_POINT:
            p_min, p_max = avg_cycle.min(), avg_cycle.max()
            margin_p = max(0.3, (p_max - p_min) * 0.18)
            ax_pixel.set_ylim(p_min - margin_p, p_max + margin_p)

        # --- 1. Subplot 1 Metrics: Continuous Laser Signal Analysis ---
        med_scope = float(np.median(scope_laser_ac))
        line_scope_med.set_ydata([med_scope, med_scope])

        high_vals = scope_laser_ac[scope_laser_ac > med_scope]
        low_vals = scope_laser_ac[scope_laser_ac <= med_scope]

        if high_vals.size and low_vals.size:
            diff_v = float(high_vals.mean() - low_vals.mean())
            vpp = float(l_max - l_min)
            is_clipped = bool(np.any(np.abs(scope_laser_raw) > 9.8))
            clip_warn = " [RAIL CLIPPING!]" if is_clipped else ""

            ax_laser.set_title(
                f"Laser (ai7) [Software AC-Coupled] | High-Low: {diff_v:.3f} V ({diff_v * 1000:.1f} mV) | Vpp: {vpp:.2f} V | Raw DC: {scope_dc:+.2f} V{clip_warn}",
                fontsize=10.5,
                fontweight="bold",
                color="#b30000" if is_clipped else "#0b5394",
            )

        # --- 2. Subplot 2 Metrics: Chopper Frequency & Duty Cycle ---
        c_mid = (c_max + c_min) / 2.0
        binary_chop = (scope_chop > c_mid).astype(int)
        rising_edges = np.where(np.diff(binary_chop) == 1)[0]

        if len(rising_edges) >= 2 and (c_max - c_min) > 0.8:
            periods = np.diff(rising_edges) / DAQ_RATE
            freq_hz = float(1.0 / np.mean(periods))
            duty_pct = float(np.mean(binary_chop) * 100.0)
            
            ax_chop.set_title(
                f"Chopper (ai15) | Freq: {freq_hz:.1f} Hz (Period: {1000.0/freq_hz:.2f} ms) | Duty: {duty_pct:.1f}% | TTL: {c_min:.1f}V – {c_max:.1f}V",
                fontsize=10.5,
                fontweight="bold",
                color="#b45f06",
            )
        else:
            ax_chop.set_title(
                f"Chopper (ai15) | Static / Searching Pulse Sync... | Span: {c_max - c_min:.2f} V",
                fontsize=10.5,
                fontweight="bold",
                color="#666666",
            )

        # --- 3. Subplot 3 Metrics: Pixel Demodulation Result ---
        if len(samples_raw) == SCAN_SAMPLES_PER_POINT:
            p_vpp = float(avg_cycle.max() - avg_cycle.min())
            ax_pixel.set_title(
                f"Averaged Single Cycle ({SCAN_PERIODS_PER_POINT} Periods Folded) | Image Pixel Voltage ΔV = {pixel_val:.3f} V | High: {h_mean:.2f}V, Low: {l_mean:.2f}V | Vpp: {p_vpp:.2f}V",
                fontsize=10.5,
                fontweight="bold",
                color="#1b5e20",
            )

    return line_laser, line_chop, line_pixel_avg


# Launch fast 30 FPS animation
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