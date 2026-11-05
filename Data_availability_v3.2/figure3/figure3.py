
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm
import warnings
import matplotlib
warnings.filterwarnings("ignore")

# ========== Configurable parameters ==========
PERIODS = ['JAS']             # only the aggregated JAS variable exists in the file
SAVED_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'JAS_MSE_smoothed_data.nc')

FIGURE_WIDTH = 5
FIGURE_HEIGHT = 10

EXP_DISPLAY = {'control': 'CTL', 'surface': 'GGW', 'diff': 'GGW-CTL'}
SHOW_PANEL_LETTER = True
PANEL_LETTER_FORMAT = '({letter}) '
CUSTOM_TITLES = {}

plt.rcParams['font.family'] = 'Times New Roman'
TITLE_FS = 22
OTHER_FS = 22
DIFF_TICKLABEL_FS = 19
matplotlib.rcParams['savefig.dpi'] = 600
plt.rcParams['axes.titlesize'] = TITLE_FS
plt.rcParams['axes.labelsize'] = OTHER_FS - 6
plt.rcParams['xtick.labelsize'] = OTHER_FS - 6
plt.rcParams['ytick.labelsize'] = OTHER_FS - 6
plt.rcParams['legend.fontsize'] = OTHER_FS - 6
plt.rcParams['font.size'] = OTHER_FS

# Colorbar & layout
CB_VERTICAL_GAP = 0.12
CB1_SHIFT_UP = 0.09
CB2_SHIFT_DOWN = -0.11
BOTTOM_PADDING = 0.15
CB_HEIGHT = 0.015
CB_WIDTH_FACTOR = 0.98
SPACER_HEIGHT = 0.28
AXES_FRAME_LINEWIDTH = 2.0
VERTICAL_LINE_COLOR = 'red'

# Tick sizes
TICK_MAJOR_LENGTH = 8.0
TICK_MAJOR_WIDTH = 2
TICK_MINOR_LENGTH = 4.0
TICK_MINOR_WIDTH = 1.0

# MSE contour levels (kJ/kg)
levels_main = np.arange(320, 351, 2)
levels_diff = np.arange(-2.0, 2.01, 0.25)


def make_centered_cmap(levels, base_cmap_name='Spectral_r', center_val=0.0):
    """Create a colormap remapped so color spacing is centered on center_val."""
    vmin = float(levels[0])
    vmax = float(levels[-1])
    if vmax <= vmin:
        raise ValueError("levels must be increasing")
    centers = 0.5 * (levels[:-1] + levels[1:])
    t = (centers - vmin) / (vmax - vmin)
    t0 = (center_val - vmin) / (vmax - vmin)
    tprime = np.empty_like(t)
    if (t0 <= 0.0) or (t0 >= 1.0) or np.isnan(t0):
        tprime = t
    else:
        left_mask = t <= t0
        right_mask = ~left_mask
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
cmap_diff = plt.get_cmap('RdBu_r', len(levels_diff) - 1)
norm_diff = BoundaryNorm(levels_diff, ncolors=cmap_diff.N, clip=True)


def load_data():
    """Load the pre-saved NetCDF dataset."""
    if os.path.exists(SAVED_DATA_FILE):
        print(f"Loading {SAVED_DATA_FILE} ...")
        return xr.open_dataset(SAVED_DATA_FILE)
    raise FileNotFoundError(f"File not found: {SAVED_DATA_FILE}")


def format_title(exp, period, letter):
    """Generate subplot title text."""
    panel = PANEL_LETTER_FORMAT.format(letter=letter) if SHOW_PANEL_LETTER else ''
    exp_display = EXP_DISPLAY.get(exp, exp.capitalize() if exp is not None else '')
    period_title = period  # simple, since we only have 'JAS' key here
    key = (exp, period)
    if key in CUSTOM_TITLES:
        return CUSTOM_TITLES[key]
    if exp == 'control' or exp == 'surface' or exp == 'diff':
        return f"{panel}{exp_display}"
    return f"{panel}{exp_display}"


def lat_label_deg(x):
    """Format latitude labels as degrees N/S."""
    try:
        xv = float(x)
    except Exception:
        return ''
    iv = int(round(xv))
    if iv == 0:
        return '0°'
    if iv < 0:
        return f"{abs(iv)}°S"
    return f"{iv}°N"


def plot_from_nc(ds):
    """Main plotting routine using data from the dataset `ds`."""
    exps = ['control', 'surface']
    periods = PERIODS  # ['JAS']
    ncols = len(periods)
    nrows_gs = 4
    height_ratios = [1.0, 1.0, SPACER_HEIGHT, 1.0]

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    gs = fig.add_gridspec(nrows_gs, ncols, height_ratios=height_ratios, hspace=0.45, wspace=0.12)
    axes = np.empty((nrows_gs, ncols), dtype=object)
    for i in range(nrows_gs):
        for j in range(ncols):
            axes[i, j] = None if i == 2 else fig.add_subplot(gs[i, j])

    # Check required variables exist
    required = [f"control_{periods[0]}_MSE_mean", f"surface_{periods[0]}_MSE_mean"]
    for v in required:
        if v not in ds:
            raise KeyError(f"Required variable not found in dataset: {v}")

    lats_all = ds['lat'].values
    plevs = ds['plev'].values
    ple = plevs / 100.0  # convert to hPa if plev is Pa

    # X ticks (20S..90N every 20 deg, clamped to data range)
    all_major_lats = np.arange(-20, 91, 20)
    lat_min = float(np.min(lats_all))
    lat_max = float(np.max(lats_all))
    major_lats = all_major_lats[(all_major_lats >= lat_min - 2.0) & (all_major_lats <= lat_max + 1e-8)]
    if major_lats.size == 0:
        major_lats = all_major_lats[(all_major_lats >= lat_min) & (all_major_lats <= lat_max)]
    minor_lats = (major_lats[:-1] + major_lats[1:]) / 2.0 if major_lats.size >= 2 else np.array([])

    # Y ticks (1000 -> 200 hPa)
    y_major = np.arange(200, 1001, 200)
    y_minor = (y_major[:-1] + y_major[1:]) / 2.0

    # Plot control and sensitivity
    for i, exp in enumerate(exps):
        for j, period in enumerate(periods):
            ax = axes[i, j]
            mse_data = ds[f"{exp}_{period}_MSE_mean"].values  # shape: (plev, lat)

            ax.contourf(lats_all, ple, mse_data, levels=levels_main, extend='both', cmap=cmap_main, norm=norm_main)
            cs = ax.contour(lats_all, ple, mse_data, levels=levels_main, linewidths=0.5, colors='k')
            try:
                ax.clabel(cs, levels=cs.levels[::2], fmt='%.1f', fontsize=9, inline=True, inline_spacing=3)
            except Exception:
                pass

            panel_index = i * ncols + j
            letter = chr(ord('a') + panel_index)
            ax.set_title(format_title(exp, period, letter), fontsize=TITLE_FS, fontweight='bold', loc='left')

            # Axes setup
            ax.set_yticks(y_major)
            ax.set_yticks(y_minor, minor=True)
            if j == 0:
                ax.set_yticklabels([f"{int(v)}" for v in y_major], fontsize=OTHER_FS)
                ax.set_ylabel('hPa', fontsize=OTHER_FS - 2, labelpad=0)
            else:
                ax.set_yticklabels([''] * len(y_major))

            ax.set_xticks(major_lats)
            if minor_lats.size:
                ax.set_xticks(minor_lats, minor=True)
            ax.set_xticklabels([lat_label_deg(x) for x in major_lats], fontsize=OTHER_FS - 4)

            ax.tick_params(axis='both', which='major', direction='out',
                           length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                           labelsize=OTHER_FS - 4)
            ax.tick_params(axis='both', which='minor', direction='out',
                           length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH)

            ax.set_xlim(major_lats[0], lat_max)
            ax.set_ylim(1000, 100)
            for spine in ax.spines.values():
                spine.set_linewidth(AXES_FRAME_LINEWIDTH)
                spine.set_edgecolor('black')
            ax.axvline(x=10, color=VERTICAL_LINE_COLOR, linestyle='-', linewidth=1.5, zorder=10)
            ax.axvline(x=20, color=VERTICAL_LINE_COLOR, linestyle='-', linewidth=1.5, zorder=10)

    # Difference row
    for j, period in enumerate(periods):
        ax = axes[3, j]
        s_mse = ds[f"surface_{period}_MSE_mean"].values
        c_mse = ds[f"control_{period}_MSE_mean"].values
        diff_mse = s_mse - c_mse

        ax.contourf(lats_all, ple, diff_mse, levels=levels_diff, extend='both', cmap=cmap_diff, norm=norm_diff)
        cs = ax.contour(lats_all, ple, diff_mse, levels=levels_diff, linewidths=0.5, colors='k')
        try:
            ax.clabel(cs, levels=cs.levels[::2], fmt='%.2f', fontsize=9, inline=True, inline_spacing=3)
        except Exception:
            pass

        panel_index = 2 * ncols + j
        letter = chr(ord('a') + panel_index)
        ax.set_title(format_title('diff', period, letter), fontsize=TITLE_FS, fontweight='bold', loc='left')

        ax.set_yticks(y_major)
        ax.set_yticks(y_minor, minor=True)
        if j == 0:
            ax.set_yticklabels([f"{int(v)}" for v in y_major], fontsize=DIFF_TICKLABEL_FS)
            ax.set_ylabel('hPa', fontsize=OTHER_FS - 2, labelpad=0)
        else:
            ax.set_yticklabels([''] * len(y_major))

        ax.set_xticks(major_lats)
        if minor_lats.size:
            ax.set_xticks(minor_lats, minor=True)
        ax.set_xticklabels([lat_label_deg(x) for x in major_lats], fontsize=OTHER_FS - 4)

        ax.tick_params(axis='both', which='major', direction='out',
                       length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                       labelsize=OTHER_FS - 4)
        ax.tick_params(axis='both', which='minor', direction='out',
                       length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH)

        ax.set_xlabel('', fontsize=OTHER_FS)
        ax.set_xlim(major_lats[0], lat_max)
        ax.set_ylim(1000, 100)
        for spine in ax.spines.values():
            spine.set_linewidth(AXES_FRAME_LINEWIDTH)
            spine.set_edgecolor('black')
        ax.axvline(x=10, color=VERTICAL_LINE_COLOR, linestyle='-', linewidth=1.5, zorder=10)
        ax.axvline(x=20, color=VERTICAL_LINE_COLOR, linestyle='-', linewidth=1.5, zorder=10)

    # Colorbars placement
    try:
        fig.subplots_adjust(top=0.96, bottom=BOTTOM_PADDING, left=0.28)
    except Exception:
        pass

    # Draw to ensure positions are computed before reading axes positions
    fig.canvas.draw()
    valid_axes = [axes[i, j] for i in [0, 1, 3] for j in range(ncols) if axes[i, j] is not None]
    real_positions = [ax.get_position() for ax in valid_axes]
    fig_left = min(p.x0 for p in real_positions)
    fig_right = max(p.x0 + p.width for p in real_positions)
    full_width = fig_right - fig_left
    width = full_width * CB_WIDTH_FACTOR
    left = fig_left + (full_width - width) / 2.0

    main_cbar_default_bottom = 0.49
    diff_cbar_default_bottom = 0.10
    main_cbar_bottom = main_cbar_default_bottom - float(CB_VERTICAL_GAP) + float(CB1_SHIFT_UP)
    diff_cbar_bottom = diff_cbar_default_bottom - float(CB_VERTICAL_GAP) - float(CB2_SHIFT_DOWN)
    main_cbar_bottom = max(0.01, min(main_cbar_bottom, 0.98 - CB_HEIGHT))
    diff_cbar_bottom = max(0.01, min(diff_cbar_bottom, main_cbar_bottom - 0.02 - CB_HEIGHT))

    cbar_main_rect = [left, main_cbar_bottom, width, CB_HEIGHT]
    cax_main = fig.add_axes(cbar_main_rect)
    cbar_diff_rect = [left, diff_cbar_bottom, width, CB_HEIGHT]
    cax_diff = fig.add_axes(cbar_diff_rect)

    sm_main = cm.ScalarMappable(cmap=cmap_main, norm=norm_main)
    sm_main.set_array(np.linspace(levels_main[0], levels_main[-1], 256))
    cbar_main = fig.colorbar(sm_main, cax=cax_main, orientation='horizontal', extend='both',
                             boundaries=levels_main, ticks=np.arange(levels_main[0], levels_main[-1] + 1, 6), spacing='uniform')
    cbar_main.set_label('kJ kg$^{-1}$', fontsize=OTHER_FS - 4, labelpad=3)
    cbar_main.ax.tick_params(labelsize=OTHER_FS - 4)

    sm_diff = cm.ScalarMappable(cmap=cmap_diff, norm=norm_diff)
    sm_diff.set_array(np.linspace(levels_diff[0], levels_diff[-1], 256))
    cbar_diff = fig.colorbar(sm_diff, cax=cax_diff, orientation='horizontal', extend='both',
                             boundaries=levels_diff, ticks=np.arange(-2.0, 2.01, 1.0), spacing='uniform')
    cbar_diff.set_label('kJ kg$^{-1}$', fontsize=OTHER_FS - 4, labelpad=3)
    cbar_diff.ax.tick_params(labelsize=OTHER_FS - 4)

    plt.savefig(_package_output / "figure3.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    ds = load_data()
    plot_from_nc(ds)
