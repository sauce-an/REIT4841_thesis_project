"""
================================================================================
QUANTITATIVE DEMODULATION BENCHMARK SCRIPT
================================================================================
Purpose:
  Direct quantitative comparison between two distinct signal demodulation
  methods performed on the same 50 µm raster scan grid of the NT$ coin target:

Scans Compared:
  1. High-Low Method (Mowla et al.):
     - Source File : ZaberScan/results/coin_scan_50um_scan_2.npy
     - Resolution  : 50 µm step size (411 rows x 430 cols = 176,730 points)
     - Chopper     : 800 Hz optical chopper
     - DAQ Config  : 10 chopper periods (125 samples @ 10 kHz per point)
     - Demodulation: Median-split High-Low amplitude (mean(high) - mean(low))
     - Status      : Baseline high-contrast reconstruction

  2. AC RMS Method (Standard Deviation):
     - Source File : ZaberScan/results/coin_scan_50um_scan_3.npy
     - Resolution  : 50 µm step size (411 rows x 430 cols = 176,730 points)
     - Chopper     : 1000 Hz optical chopper
     - DAQ Config  : 16 chopper periods (320 samples @ 20 kHz per point)
     - Demodulation: AC RMS standard deviation (V_pp = 2 * std(raw_samples))
     - Status      : Severe dynamic range compression (~69% contrast loss)

Output:
  Generates a 4-panel publication-ready comparison figure saved to:
  ZaberScan/results/analyzed/demod_comparison_highlow_vs_acrms.png
================================================================================
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Directory Setup
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
ANALYZED_DIR = os.path.join(RESULTS_DIR, "analyzed")
os.makedirs(ANALYZED_DIR, exist_ok=True)

# Input Datasets
file_highlow = os.path.join(RESULTS_DIR, "coin_scan_50um_scan_2.npy")  # 50um scan_2: Mowla High-Low
file_acrms = os.path.join(RESULTS_DIR, "coin_scan_50um_scan_3.npy")    # 50um scan_3: AC RMS
output_png = os.path.join(ANALYZED_DIR, "demod_comparison_highlow_vs_acrms.png")

# Load datasets
data_hl = np.load(file_highlow)   # High-Low dataset (50um scan_2)
data_rms = np.load(file_acrms)    # AC RMS dataset (50um scan_3)

# Physical dimensions
scan_width_mm = 21.43
scan_height_mm = 20.48
num_y, num_x = data_hl.shape
x_mm = np.linspace(0, scan_width_mm, num_x)
y_mm = np.linspace(0, scan_height_mm, num_y)

# Extract statistics
valid_hl = data_hl[~np.isnan(data_hl)]
valid_rms = data_rms[~np.isnan(data_rms)]

p5_hl, p95_hl = np.percentile(valid_hl, 5), np.percentile(valid_hl, 95)
p5_rms, p95_rms = np.percentile(valid_rms, 5), np.percentile(valid_rms, 95)

dyn_range_hl = p95_hl - p5_hl
dyn_range_rms = p95_rms - p5_rms

# Choose cross-section row at Y = 5.0 mm
slice_y_target_mm = 5.0
slice_row_idx = int(np.argmin(np.abs(y_mm - slice_y_target_mm)))
slice_y_mm = y_mm[slice_row_idx]
slice_hl = data_hl[slice_row_idx, :]
slice_rms = data_rms[slice_row_idx, :]

# ==============================================================================
# CREATE 4-PANEL FIGURE
# ==============================================================================
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig = plt.figure(figsize=(16, 13), dpi=150)
fig.suptitle("Quantitative Demodulation Benchmark: High-Low vs. AC RMS Method", 
             fontsize=16, fontweight="bold", y=0.98)

# Custom Colormaps with Red for NaNs/Saturation
cmap_hl = plt.cm.gray.copy()
cmap_hl.set_bad(color="red")
cmap_rms = plt.cm.gray.copy()
cmap_rms.set_bad(color="red")

# --- PANEL A: 2D Image (High-Low Method) ---
ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(
    data_hl,
    cmap=cmap_hl,
    origin="lower",
    extent=[0, scan_width_mm, 0, scan_height_mm],
    aspect="equal",
    vmin=p5_hl,
    vmax=p95_hl,
)
ax1.axhline(slice_y_mm, color="#00ffcc", linestyle="--", linewidth=1.5, label=f"Profile Line (Y = {slice_y_mm:.1f} mm)")
ax1.set_title("(A) High-Low Method", fontsize=12, fontweight="bold", pad=8)
ax1.set_xlabel("X Position (mm)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Y Position (mm)", fontsize=10, fontweight="bold")
ax1.legend(loc="upper right", framealpha=0.85, fontsize=9)
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label("Demodulated Voltage (V)", fontsize=10, fontweight="bold")

# --- PANEL B: 2D Image (AC RMS Method) ---
ax2 = fig.add_subplot(2, 2, 2)
im2 = ax2.imshow(
    data_rms,
    cmap=cmap_rms,
    origin="lower",
    extent=[0, scan_width_mm, 0, scan_height_mm],
    aspect="equal",
    vmin=p5_rms,
    vmax=p95_rms,
)
ax2.axhline(slice_y_mm, color="#00ffcc", linestyle="--", linewidth=1.5, label=f"Profile Line (Y = {slice_y_mm:.1f} mm)")
ax2.set_title("(B) AC RMS Method", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("X Position (mm)", fontsize=10, fontweight="bold")
ax2.set_ylabel("Y Position (mm)", fontsize=10, fontweight="bold")
ax2.legend(loc="upper right", framealpha=0.85, fontsize=9)
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Demodulated Voltage (V)", fontsize=10, fontweight="bold")

# --- PANEL C: Histogram & Distribution Comparison ---
ax3 = fig.add_subplot(2, 2, 3)
bins = np.linspace(1.5, 6.0, 90)

ax3.hist(valid_hl, bins=bins, density=True, alpha=0.55, color="#1f77b4", edgecolor="#104e8b", 
         label=f"High-Low: $\\mu={np.mean(valid_hl):.2f}$ V, $\\sigma={np.std(valid_hl):.2f}$ V, Range={dyn_range_hl:.2f} V")
ax3.hist(valid_rms, bins=bins, density=True, alpha=0.55, color="#d95f02", edgecolor="#8b3a00", 
         label=f"AC RMS: $\\mu={np.mean(valid_rms):.2f}$ V, $\\sigma={np.std(valid_rms):.2f}$ V, Range={dyn_range_rms:.2f} V")

# Percentile markers
ax3.axvline(p5_hl, color="#1f77b4", linestyle=":", lw=1.8, label=f"High-Low 5th & 95th Percentile")
ax3.axvline(p95_hl, color="#1f77b4", linestyle=":", lw=1.8)
ax3.axvline(p5_rms, color="#d95f02", linestyle="--", lw=1.8, label=f"AC RMS 5th & 95th Percentile")
ax3.axvline(p95_rms, color="#d95f02", linestyle="--", lw=1.8)

ax3.set_title("(C) Global Pixel Voltage Probability Density (Histogram)", fontsize=12, fontweight="bold", pad=8)
ax3.set_xlabel("Demodulated LFI Voltage (V)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
ax3.set_xlim(1.5, 6.0)
ax3.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.6)

# Text annotation on collapse
ax3.annotate("Severe Dynamic Range\nCompression in AC RMS", 
             xy=(4.9, 1.8), xytext=(2.2, 1.5),
             arrowprops=dict(facecolor="#d95f02", shrink=0.08, width=1.5, headwidth=7),
             fontsize=10, fontweight="bold", color="#8b3a00",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff2e6", ec="#d95f02", lw=1))

# --- PANEL D: 1D Horizontal Cross-Sectional Line Profile ---
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x_mm, slice_hl, color="#1f77b4", lw=1.5, label="High-Low Demodulation Profile")
ax4.plot(x_mm, slice_rms, color="#d95f02", lw=1.5, linestyle="-.", label="AC RMS Demodulation Profile")

ax4.set_title(f"(D) Cross-Section Profile Across Coin Features (Y = {slice_y_mm:.1f} mm)", fontsize=12, fontweight="bold", pad=8)
ax4.set_xlabel("X Position across Coin (mm)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Demodulated Voltage (V)", fontsize=11, fontweight="bold")
ax4.set_xlim(0, scan_width_mm)
ax4.set_ylim(1.5, 6.0)
ax4.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
ax4.grid(True, linestyle="--", alpha=0.6)

# Highlight feature contrast
valid_slice_range = slice_hl[int(num_x * 0.25):int(num_x * 0.75)]
if valid_slice_range.size and not np.all(np.isnan(valid_slice_range)):
    dip_rel_idx = int(np.nanargmin(valid_slice_range))
    dip_idx = dip_rel_idx + int(num_x * 0.25)
    dip_x = x_mm[dip_idx]
    dip_v = slice_hl[dip_idx]
    
    ax4.annotate("High Feature Contrast Dip",
                 xy=(dip_x, dip_v), xytext=(dip_x + 2.0, dip_v - 1.0 if dip_v > 3.0 else dip_v + 1.2),
                 arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=7),
                 fontsize=9.5, fontweight="bold", color="#104e8b",
                 bbox=dict(boxstyle="round,pad=0.4", fc="#e6f2ff", ec="#1f77b4", lw=1))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(output_png, dpi=300)
print(f"[OK] High-resolution benchmark figure saved to: {output_png}")
