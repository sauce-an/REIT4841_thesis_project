"""
Live dual-channel oscilloscope-style view of Laser Terminal & Optical Chopper.

Features:
  - High-speed 100 kS/s sampling rate
  - Channel 7  (Dev1/ai7)  : VCSEL Laser Terminal AC-coupled LFI Signal
  - Channel 15 (Dev1/ai15) : Optical Chopper TTL Reference / Sync Signal
  - Buffer-drain architecture: runs indefinitely with zero freeze or backlog
  - Real-time frequency estimation and confocal demodulation metrics

CLOSE NI MAX OR ANY OTHER DAQ PROCESS BEFORE RUNNING THIS SCRIPT.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nidaqmx
from nidaqmx.constants import AcquisitionType

# --- CONFIGURATION ---
DEVICE_NAME = "Dev1"
LASER_CHANNEL = f"{DEVICE_NAME}/ai7"      # Channel 7: Laser terminal voltage
CHOPPER_CHANNEL = f"{DEVICE_NAME}/ai15"   # Channel 15: Optical chopper sync
DAQ_RATE = 100_000                        # 100 kHz (100,000 samples/sec per channel)
EST_CHOPPER_FREQ = 800                    # Approximate chopper frequency for display scale
DISPLAY_PERIODS = 12                      # Number of chopper cycles to show in window
CHUNK_SIZE = int(DAQ_RATE * DISPLAY_PERIODS / EST_CHOPPER_FREQ)  # ~1500 samples (15 ms)
BUFFER_SIZE = 200_000                     # 2-second circular hardware buffer

print("=" * 65)
print("Starting NI-DAQmx Live Dual-Channel Oscilloscope")
print(f"  Sample Rate   : {DAQ_RATE:,} Hz (100 kS/s)")
print(f"  Laser Channel : {LASER_CHANNEL} (Ch 7)")
print(f"  Chopper Sync  : {CHOPPER_CHANNEL} (Ch 15)")
print(f"  Window Size   : {CHUNK_SIZE} samples (~{CHUNK_SIZE / DAQ_RATE * 1000:.1f} ms)")
print("=" * 65)

# --- Initialize DAQ Continuous Task ---
task = nidaqmx.Task()
try:
    task.ai_channels.add_ai_voltage_chan(LASER_CHANNEL, name_to_assign_to_channel="Laser_ai7")
    task.ai_channels.add_ai_voltage_chan(CHOPPER_CHANNEL, name_to_assign_to_channel="Chopper_ai15")
    task.timing.cfg_samp_clk_timing(
        rate=DAQ_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=BUFFER_SIZE,
    )
    task.start()
except Exception as e:
    print(f"[ERROR] Failed to initialize DAQ: {e}")
    try:
        task.close()
    except Exception:
        pass
    sys.exit(1)

# --- Set Up Plot Window ---
fig, (ax_laser, ax_chop) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.canvas.manager.set_window_title("NI DAQ Live Monitor — Laser & Chopper Signals")

t_ms = np.arange(CHUNK_SIZE) / DAQ_RATE * 1000  # Time axis in milliseconds

# Top plot: Laser Signal
line_laser, = ax_laser.plot(t_ms, np.zeros(CHUNK_SIZE), color="#1f77b4", lw=1.5, label="Laser ai7")
ax_laser.set_ylabel("Laser Signal (V)", fontsize=11, fontweight="bold")
ax_laser.set_ylim(-6.0, 6.0)
ax_laser.grid(True, linestyle="--", alpha=0.6)
ax_laser.legend(loc="upper right", framealpha=0.8)

# Bottom plot: Chopper Sync Signal
line_chop, = ax_chop.plot(t_ms, np.zeros(CHUNK_SIZE), color="#ff7f0e", lw=1.5, label="Chopper ai15")
ax_chop.set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
ax_chop.set_ylabel("Chopper Sync (V)", fontsize=11, fontweight="bold")
ax_chop.set_ylim(-1.0, 6.0)
ax_chop.grid(True, linestyle="--", alpha=0.6)
ax_chop.legend(loc="upper right", framealpha=0.8)

plt.tight_layout()

# Rolling cache for smooth display
cached_laser = np.zeros(CHUNK_SIZE)
cached_chop = np.zeros(CHUNK_SIZE)
frame_count = 0


def update(frame):
    global cached_laser, cached_chop, frame_count
    frame_count += 1

    try:
        # Check how many samples are available in the DAQ hardware buffer
        avail = task.in_stream.avail_samp_per_chan
        if avail > 0:
            # Read all available samples to completely drain the buffer (prevents overflow)
            raw = np.array(task.read(number_of_samples_per_channel=avail))
            
            if raw.shape[1] >= CHUNK_SIZE:
                cached_laser = raw[0, -CHUNK_SIZE:]
                cached_chop = raw[1, -CHUNK_SIZE:]
            else:
                # Append to cached window if fewer than CHUNK_SIZE arrived
                shift = raw.shape[1]
                cached_laser = np.roll(cached_laser, -shift)
                cached_chop = np.roll(cached_chop, -shift)
                cached_laser[-shift:] = raw[0]
                cached_chop[-shift:] = raw[1]
    except Exception as read_err:
        if frame_count % 50 == 0:
            print(f"[Notice] Read exception: {read_err}")
        return line_laser, line_chop

    # Update plot line data
    line_laser.set_ydata(cached_laser)
    line_chop.set_ydata(cached_chop)

    # Periodic UI metrics & auto-scale update (every 4 frames for smooth rendering)
    if frame_count % 4 == 0:
        l_min, l_max = cached_laser.min(), cached_laser.max()
        margin_l = max(0.2, (l_max - l_min) * 0.15)
        ax_laser.set_ylim(l_min - margin_l, l_max + margin_l)

        c_min, c_max = cached_chop.min(), cached_chop.max()
        margin_c = max(0.5, (c_max - c_min) * 0.15)
        ax_chop.set_ylim(c_min - margin_c, c_max + margin_c)

        # --- Real-Time Signal Analysis ---
        # 1. Demodulated Laser Reflectance (Median split: high level - low level)
        med = np.median(cached_laser)
        high_vals = cached_laser[cached_laser > med]
        low_vals = cached_laser[cached_laser <= med]
        if high_vals.size and low_vals.size:
            diff_v = high_vals.mean() - low_vals.mean()
            vpp = l_max - l_min
            ax_laser.set_title(
                f"Laser (ai7) | LFI Signal: {diff_v * 1000:.1f} mV (High-Low) | Vpp: {vpp:.2f} V | Mean: {cached_laser.mean():.2f} V",
                fontsize=10,
                color="#0b5394",
            )

        # 2. Chopper Frequency & Duty Cycle Calculation
        c_mid = (c_max + c_min) / 2.0
        binary_chop = (cached_chop > c_mid).astype(int)
        rising_edges = np.where(np.diff(binary_chop) == 1)[0]

        if len(rising_edges) >= 2 and (c_max - c_min) > 0.5:
            periods = np.diff(rising_edges) / DAQ_RATE
            freq_hz = 1.0 / np.mean(periods)
            duty_pct = np.mean(binary_chop) * 100.0
            ax_chop.set_title(
                f"Chopper (ai15) | Measured Freq: {freq_hz:.1f} Hz | Duty Cycle: {duty_pct:.1f}% | Vpp: {c_max - c_min:.2f} V",
                fontsize=10,
                color="#b45f06",
            )
        else:
            ax_chop.set_title(
                f"Chopper (ai15) | Static / No Square Wave | Vpp: {c_max - c_min:.2f} V",
                fontsize=10,
                color="#990000",
            )

    return line_laser, line_chop


ani = animation.FuncAnimation(fig, update, interval=25, blit=False, cache_frame_data=False)

try:
    plt.show()
finally:
    try:
        task.stop()
        task.close()
        print("\n[OK] DAQ Task closed cleanly.")
    except Exception:
        pass