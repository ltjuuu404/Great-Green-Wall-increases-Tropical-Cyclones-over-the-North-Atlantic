#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compact, stable two-panel plot (left: CTL, right: GGW-CTL).
This script ONLY reads the previously saved NetCDF (SAVED_DATA_FILE)
and plots the results. All raw-data collection / smoothing / processing
functions have been removed — see comments where those blocks were removed.
"""

# Package portability setup: paths and batch export only; source plot parameters retained.
from pathlib import Path as _PackagePath
import os as _package_os
import matplotlib as _package_mpl
_package_mpl.use("Agg")
_package_here = _PackagePath(__file__).resolve().parent
_package_os.chdir(_package_here)
_package_output = _package_here / "outputs"
_package_output.mkdir(exist_ok=True)
import cartopy as _package_cartopy
_package_map_dir = str(_package_here.parent / "map_data")
_package_cartopy.config["data_dir"] = _package_map_dir
_package_cartopy.config["pre_existing_data_dir"] = _package_map_dir


import os
import math
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import warnings
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
warnings.filterwarnings("ignore")

# ========== Global adjustable parameters ==========
# NOTE: This script expects the processed NetCDF file (SAVED_DATA_FILE)
# produced by the previous pipeline to exist in the current working directory.
SAVED_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JAS_smoothed_data.nc')

PERIODS = {'JAS': ['07', '08', '09'],}
PERIOD_TITLES = {'JAS': 'JAS'}

LON_RANGE = (-70, 17.5)
LAT_RANGE = (-20, 50)
PLEV_TARGET = (10000, 100000)  # Pa

SMOOTH_WIN = 5
LAT_BUFFER = 5

# Figure size and layout
FIGURE_WIDTH = 9.5
FIGURE_HEIGHT = 5.8

GS_LEFT = 0.08
GS_RIGHT = 0.97
GS_BOTTOM = 0.12
GS_TOP = 0.92
GS_WSPACE = 0.18

EXP_DISPLAY = {'control': 'CTL', 'surface': 'GGW', 'diff': 'GGW-CTL'}

plt.rcParams['font.family'] = 'Times New Roman'
TITLE_FS = 20
OTHER_FS = 16
DIFF_TICKLABEL_FS = 14

# quiver parameters
PLOT_VECTORS = True
OMEGA_SCALE = -200
QUIVER_SKIP = {'lat': 2, 'plev': 1}
QUIVER_WIDTH = 0.003
QUIVER_SCALE = 60
QUIVER_SCALE_DIFF = 12
QUIVER_V_REF = 0.05
QUIVER_V_REF_DIFF = 0.005
QUIVERKEY_POS_X = 0.70
QUIVERKEY_POS_Y = 1.02
QUIVERKEY_FONTSIZE = 12

# ---------- axis/frame/tick thickness ----------
AXES_FRAME_LINEWIDTH = 2.0     # spine linewidth
TICK_MAJOR_LENGTH = 8.0
TICK_MAJOR_WIDTH = 2.0
TICK_MINOR_LENGTH = 5.0
TICK_MINOR_WIDTH = 0.8
# -----------------------------------------------

VERTICAL_LINE_COLOR = 'red'

levels_main = np.arange(-10, 31, 2)
levels_diff = np.arange(-2.0, 2.01, 0.5)

def make_centered_cmap(levels, base_cmap_name='Spectral_r', center_val=0.0):
    # Create a colormap whose center maps to center_val (useful for asymmetric ranges)
    vmin = float(levels[0]); vmax = float(levels[-1])
    centers = 0.5 * (levels[:-1] + levels[1:])
    t = (centers - vmin) / (vmax - vmin)
    t0 = (center_val - vmin) / (vmax - vmin)
    tprime = np.empty_like(t)
    if (t0 <= 0.0) or (t0 >= 1.0) or np.isnan(t0):
        tprime = t
    else:
        left_mask = t <= t0; right_mask = ~left_mask
        if t0 > 0:
            tprime[left_mask] = 0.5 * (t[left_mask] / t0)
        else:
            tprime[left_mask] = 0.0
        if (1.0 - t0) > 0:
            tprime[right_mask] = 0.5 + 0.5 * ((t[right_mask] - t0) / (1.0 - t0))
        else:
            tprime[right_mask] = 1.0
    tprime = np.clip(tprime, 0.0, 1.0)
    base = plt.get_cmap(base_cmap_name)
    colors = base(tprime)
    cmap = ListedColormap(colors, name=f'{base_cmap_name}_center_{center_val}')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    return cmap, norm

cmap_main, norm_main = make_centered_cmap(levels_main, base_cmap_name='Spectral_r', center_val=0.0)
cmap_diff = plt.get_cmap('RdBu_r', len(levels_diff)-1)
norm_diff = BoundaryNorm(levels_diff, ncolors=cmap_diff.N, clip=True)

# ------------------ Data processing REMOVED ------------------
# The following functions and code that collected raw NetCDF files,
# interpolated/smoothed them, stacked samples, and wrote the SAVED_DATA_FILE
# were intentionally removed from this script. This script assumes that
# 'JAS_smoothed_data.nc' already exists and contains:
#   control_JAS_U_mean, control_JAS_OMEGA_mean,
#   surface_JAS_U_mean, surface_JAS_OMEGA_mean,
#   control_JAS_U_samples, surface_JAS_U_samples (for p-value calculation).
#
# Removed functions include:
#   - collect_files()
#   - process_and_smooth()
#   - generate_data()
# The logic for generating the netCDF file should be run separately if needed.
# ----------------------------------------------------------------

def load_data():
    """
    Open the precomputed NetCDF file produced by the preprocessing step.
    If the file is not found, raise an informative error.
    """
    if os.path.exists(SAVED_DATA_FILE):
        print(f"Detected {SAVED_DATA_FILE}, opening dataset...")
        return xr.open_dataset(SAVED_DATA_FILE)
    else:
        # Inform the user that preprocessing is required to generate the NetCDF.
        raise FileNotFoundError(
            f"{SAVED_DATA_FILE} not found. This script only plots from the "
            "precomputed file. Run the preprocessing pipeline to create it, "
            "or place the file in the current working directory."
        )

# Helper: latitude label formatting
def lat_label_deg(x):
    try:
        xv = float(x)
    except:
        return ''
    iv = int(round(xv))
    if iv == 0:
        return '0°'
    if iv < 0:
        return f"{abs(iv)}°S"
    return f"{iv}°N"

# ========== Plotting function (reads from loaded dataset) ==========
def plot_from_nc(ds):
    period = list(PERIODS.keys())[0]

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    gs = fig.add_gridspec(
        1, 2,
        left=GS_LEFT,
        right=GS_RIGHT,
        bottom=GS_BOTTOM,
        top=GS_TOP,
        wspace=GS_WSPACE
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    lats_all = ds['lat'].values
    plevs = ds['plev'].values
    ple = plevs / 100.0

    # compute major lat ticks from available lats
    all_major_lats = np.arange(-90, 91, 10)
    lat_min = float(np.min(lats_all)); lat_max = float(np.max(lats_all))
    major_lats = all_major_lats[(all_major_lats >= lat_min - 1e-8) & (all_major_lats <= lat_max + 1e-8)]
    if major_lats.size == 0:
        major_lats = all_major_lats[(all_major_lats >= lat_min) & (all_major_lats <= lat_max)]
    # minor lat ticks: midpoints between majors (one minor between each pair)
    minor_lats = (major_lats[:-1] + major_lats[1:]) / 2.0 if major_lats.size >= 2 else np.array([])

    # y ticks: majors and minors (midpoints)
    y_major = np.arange(200, 1001, 200)
    y_minor = (y_major[:-1] + y_major[1:]) / 2.0  # 300,500,700,900

    # ----------------------------
    # Left: CTL
    # ----------------------------
    u_ctl = ds[f"control_{period}_U_mean"].values
    om_ctl = ds[f"control_{period}_OMEGA_mean"].values
    v_ctl = ds[f"control_{period}_V_mean"].values

    cf1 = ax_left.contourf(lats_all, ple, u_ctl, levels=levels_main, cmap=cmap_main, norm=norm_main, extend='both')
    cs1 = ax_left.contour(lats_all, ple, u_ctl, levels=levels_main, colors='k', linewidths=0.5)
    try:
        ax_left.clabel(cs1, levels=cs1.levels[::2], fmt='%.1f', fontsize=9, inline=True, inline_spacing=3)
    except:
        pass

    ax_left.set_title("(a) " + EXP_DISPLAY['control'], fontsize=TITLE_FS, loc='left', fontweight='bold')
    ax_left.set_ylim(1000, 100)
    ax_left.set_ylabel('Pressure (hPa)', fontsize=OTHER_FS, labelpad=0)

    ax_left.set_xticks(major_lats)
    if minor_lats.size:
        ax_left.set_xticks(minor_lats, minor=True)
    ax_left.set_yticks(y_major)
    ax_left.set_yticks(y_minor, minor=True)

    ax_left.set_xticklabels([lat_label_deg(x) for x in major_lats], fontsize=DIFF_TICKLABEL_FS)

    # --- apply thick spines & major/minor tick styling ---
    for spine in ax_left.spines.values():
        spine.set_linewidth(AXES_FRAME_LINEWIDTH)
        spine.set_edgecolor('black')

    ax_left.tick_params(axis='both', which='major', direction='out',
                        length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                        labelsize=DIFF_TICKLABEL_FS)
    ax_left.tick_params(axis='both', which='minor', direction='out',
                        length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH)

    ax_left.axvline(x=10, color=VERTICAL_LINE_COLOR, linewidth=1.2)
    ax_left.axvline(x=20, color=VERTICAL_LINE_COLOR, linewidth=1.2)

    # quiver on left (omega vectors scaled vertically)
    lat_sl = slice(None, None, QUIVER_SKIP['lat'])
    plev_sl = slice(None, None, QUIVER_SKIP['plev'])
    try:
        omega_plot = om_ctl[plev_sl, lat_sl] * OMEGA_SCALE
        v_plot = v_ctl[plev_sl, lat_sl]
        q1 = ax_left.quiver(lats_all[lat_sl], ple[plev_sl], v_plot, omega_plot,
                            color='black', pivot='middle', width=QUIVER_WIDTH, 
                            scale=QUIVER_SCALE, scale_units='width',
                            headwidth=12, headlength=8.5, headaxislength=2)
        ax_left.quiverkey(q1, X=QUIVERKEY_POS_X, Y=QUIVERKEY_POS_Y, U=QUIVER_V_REF * abs(OMEGA_SCALE),
                          label=f'{(QUIVER_V_REF * abs(OMEGA_SCALE)):.0f} / {QUIVER_V_REF:.2f} (v/ω)', labelpos='E', coordinates='axes', fontproperties={'size': QUIVERKEY_FONTSIZE})
    except Exception:
        q1 = None

    # colorbar: slightly wider fraction/pad
    cb1 = fig.colorbar(cf1, ax=ax_left, orientation='horizontal', pad=0.08, fraction=0.08, shrink=0.95)
    cb1.set_label('m s$^{{-1}}$', fontsize=OTHER_FS-1)
    cb1.ax.tick_params(labelsize=OTHER_FS-2)

    # ----------------------------
    # Right: DIFF (surface - control)
    # ----------------------------
    u_surf = ds[f"surface_{period}_U_mean"].values
    om_surf = ds[f"surface_{period}_OMEGA_mean"].values
    v_surf = ds[f"surface_{period}_V_mean"].values
    diff_u = u_surf - u_ctl
    diff_om = om_surf - om_ctl
    diff_v = v_surf - v_ctl

    cf2 = ax_right.contourf(lats_all, ple, diff_u, levels=levels_diff, cmap=cmap_diff, norm=norm_diff, extend='both')
    cs2 = ax_right.contour(lats_all, ple, diff_u, levels=levels_diff, colors='k', linewidths=0.5)
    try:
        ax_right.clabel(cs2, levels=cs2.levels[::2], fmt='%.2f', fontsize=9, inline=True, inline_spacing=3)
    except:
        pass

    ax_right.set_title("(b) " + EXP_DISPLAY['diff'], fontsize=TITLE_FS, loc='left', fontweight='bold')
    ax_right.set_ylim(1000, 100)

    ax_right.set_xticks(major_lats)
    if minor_lats.size:
        ax_right.set_xticks(minor_lats, minor=True)
    ax_right.set_yticks(y_major)
    ax_right.set_yticks(y_minor, minor=True)

    ax_right.set_xticklabels([lat_label_deg(x) for x in major_lats], fontsize=DIFF_TICKLABEL_FS)
    ax_right.set_ylabel('')  # keep spacing consistent without repeating label

    # --- apply thick spines & tick styling to right ---
    for spine in ax_right.spines.values():
        spine.set_linewidth(AXES_FRAME_LINEWIDTH)
        spine.set_edgecolor('black')

    ax_right.tick_params(axis='both', which='major', direction='out',
                         length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                         labelsize=DIFF_TICKLABEL_FS)
    ax_right.tick_params(axis='both', which='minor', direction='out',
                         length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH)

    ax_right.axvline(x=10, color=VERTICAL_LINE_COLOR, linewidth=1.2)
    ax_right.axvline(x=20, color=VERTICAL_LINE_COLOR, linewidth=1.2)

    # quiver diff
    try:
        omega_plot2 = diff_om[plev_sl, lat_sl] * OMEGA_SCALE
        v_plot2 = diff_v[plev_sl, lat_sl]
        q2 = ax_right.quiver(lats_all[lat_sl], ple[plev_sl], v_plot2, omega_plot2,
                             color='black', pivot='middle', width=QUIVER_WIDTH, 
                             scale=QUIVER_SCALE_DIFF, scale_units='width',
                             headwidth=12, headlength=8.5, headaxislength=2)
        ax_right.quiverkey(q2, X=QUIVERKEY_POS_X, Y=QUIVERKEY_POS_Y, U=QUIVER_V_REF_DIFF * abs(OMEGA_SCALE),
                           label=f'{(QUIVER_V_REF_DIFF * abs(OMEGA_SCALE)):.1f} / {QUIVER_V_REF_DIFF:.3f} (v/ω)', labelpos='E', coordinates='axes', fontproperties={'size': QUIVERKEY_FONTSIZE})
    except Exception:
        q2 = None



    cb2 = fig.colorbar(cf2, ax=ax_right, orientation='horizontal', pad=0.08, fraction=0.08, shrink=0.95)
    cb2.set_label('m s$^{{-1}}$', fontsize=OTHER_FS-1)
    cb2.ax.tick_params(labelsize=OTHER_FS-2)

    plt.savefig(_package_output / "figureS6.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # Load precomputed dataset and plot
    ds = load_data()
    plot_from_nc(ds)
