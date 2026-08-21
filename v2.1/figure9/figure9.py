#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from pathlib import Path

# Configuration
# Point to the pre-processed file with ocean mask applied
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "figure9_data.nc"
OUT_FILE = HERE / "figure9.svg"

# Plotting Region
PLOT_LON_MIN, PLOT_LON_MAX = -90, -10
PLOT_LAT_MIN, PLOT_LAT_MAX = 0, 35

# Box for region (for drawing on map)
BOX_LON_MIN, BOX_LON_MAX = -70.0, -17.5
BOX_LAT_MIN, BOX_LAT_MAX = 10.0, 20.0

# Fonts
FONT_NAME = "Times New Roman"
FONT_SIZE = 14
plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['savefig.dpi'] = 600

# Styling
AXES_FRAME_LINEWIDTH = 2
TICK_LENGTH_MAJOR = 8
TICK_LENGTH_MINOR = 5
BAR_WIDTH = 0.4
CBAR_PAD = 0.15 

# Colorbars (Synced for both DGPI and ENGPI)
CBAR_CTRL = {"vmin": 0.0, "vmax": 5.0, "nlevels": 11, "cmap": "YlOrRd"}
CBAR_DIFF = {"vmin": -1.0, "vmax": 1.0, "nlevels": 11, "cmap": "RdBu_r"}

def region_indices(lon_vals, lat_vals, lon0, lon1, lat0, lat1):
    lon = np.array(lon_vals)
    lat = np.array(lat_vals)
    def norm(l): return (((l + 180) % 360) - 180)
    lon0n, lon1n = norm(lon0), norm(lon1)
    if lon0n <= lon1n:
        lon_mask = (lon >= lon0n) & (lon <= lon1n)
    else:
        lon_mask = (lon >= lon0n) | (lon <= lon1n)
    lat_mask = (lat >= lat0) & (lat <= lat1)
    return lon_mask, lat_mask

def format_lon_deg(x, pos):
    xv = float(x)
    if xv > 180: xv = ((xv + 180) % 360) - 180
    if abs(xv - round(xv)) < 1e-6:
        deg = int(round(abs(xv))); s = f"{deg}°"
    else:
        s = f"{abs(xv):.1f}°"
    if xv < 0: return s + "W"
    if xv > 0: return s + "E"
    return s

def format_lat_deg(x, pos):
    yv = float(x)
    if abs(yv - round(yv)) < 1e-6:
        deg = int(round(abs(yv))); s = f"{deg}°"
    else:
        s = f"{abs(yv):.1f}°"
    if yv < 0: return s + "S"
    if yv > 0: return s + "N"
    return s

def compute_major_ticks(minv, maxv, step=10.0):
    start = np.ceil(minv / step) * step
    end = np.floor(maxv / step) * step
    if start > end: return np.array([minv, maxv])
    ticks = np.arange(start, end + 0.01, step)
    return ticks

def midpoints(arr):
    if arr.size < 2: return np.array([])
    return (arr[:-1] + arr[1:]) / 2.0

def plot_map_on_ax(ax, da, cbar_params, title, title_prefix=""):
    # Extract lat/lon from DA
    lat = da['lat_dgpi'].values if 'lat_dgpi' in da.coords else da['lat_engpi'].values
    lon = da['lon_dgpi'].values if 'lon_dgpi' in da.coords else da['lon_engpi'].values
    
    data = da.values
    levels = np.linspace(cbar_params['vmin'], cbar_params['vmax'], cbar_params['nlevels'])
    
    cf = ax.contourf(lon, lat, data, levels=levels, cmap=cbar_params['cmap'],
                     transform=ccrs.PlateCarree(), extend='both')
    
    # Draw box
    width = BOX_LON_MAX - BOX_LON_MIN
    height = BOX_LAT_MAX - BOX_LAT_MIN
    rect = mpatches.Rectangle((BOX_LON_MIN, BOX_LAT_MIN), width, height,
                              linewidth=2, edgecolor='k', facecolor='none', 
                              transform=ccrs.PlateCarree(), zorder=5)
    ax.add_patch(rect)
    
    ax.coastlines('50m', linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4)
    ax.set_extent([PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX], crs=ccrs.PlateCarree())
    
    full_title = f"{title_prefix} {title}"
    ax.set_title(full_title, loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    
    # --- Custom Ticks ---
    lon_major = np.arange(PLOT_LON_MIN, PLOT_LON_MAX + 0.01, 20.0)
    lat_major = compute_major_ticks(PLOT_LAT_MIN, PLOT_LAT_MAX, step=10.0)
    lon_minor = midpoints(lon_major)
    lat_minor = midpoints(lat_major)
    
    lon_formatter = mticker.FuncFormatter(format_lon_deg)
    lat_formatter = mticker.FuncFormatter(format_lat_deg)
    
    try: ax.set_xticks(lon_major, crs=ccrs.PlateCarree())
    except: ax.set_xticks(lon_major)
    ax.xaxis.set_major_formatter(lon_formatter)
    
    try: ax.set_yticks(lat_major, crs=ccrs.PlateCarree())
    except: ax.set_yticks(lat_major)
    ax.yaxis.set_major_formatter(lat_formatter)
    
    if lon_minor.size > 0:
        try: ax.set_xticks(lon_minor, minor=True, crs=ccrs.PlateCarree())
        except: ax.set_xticks(lon_minor, minor=True)
    if lat_minor.size > 0:
        try: ax.set_yticks(lat_minor, minor=True, crs=ccrs.PlateCarree())
        except: ax.set_yticks(lat_minor, minor=True)
        
    ax.tick_params(axis='x', which='major', labelbottom=True, direction='out', 
                   length=TICK_LENGTH_MAJOR, width=AXES_FRAME_LINEWIDTH, labelsize=FONT_SIZE-1)
    ax.tick_params(axis='x', which='minor', labelbottom=False, direction='out', 
                   length=TICK_LENGTH_MINOR, width=AXES_FRAME_LINEWIDTH*0.6)
    ax.tick_params(axis='y', which='major', labelleft=True, direction='out', 
                   length=TICK_LENGTH_MAJOR, width=AXES_FRAME_LINEWIDTH, labelsize=FONT_SIZE-1)
    ax.tick_params(axis='y', which='minor', labelleft=False, direction='out', 
                   length=TICK_LENGTH_MINOR, width=AXES_FRAME_LINEWIDTH*0.6)

    # Thick Spines
    try:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(AXES_FRAME_LINEWIDTH)
            spine.set_edgecolor('black')
    except:
        try:
            outline = getattr(ax, 'outline_patch', None)
            if outline is not None:
                outline.set_linewidth(AXES_FRAME_LINEWIDTH)
                outline.set_edgecolor('black')
        except: pass
        
    return cf

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return
        
    ds = xr.open_dataset(DATA_FILE)
    
    # DGPI Data
    dgpi_ctrl = ds['dgpi_ctrl']
    dgpi_diff = ds['dgpi_diff']
    dgpi_contribs = ds['dgpi_contribs'].values
    dgpi_factors = ds['factor_dgpi'].values
    
    # ENGPI Data
    engpi_ctrl = ds['engpi_ctrl']
    engpi_diff = ds['engpi_diff']
    engpi_contribs = ds['engpi_contribs'].values
    engpi_factors = ds['factor_engpi'].values
    
    # Determine ENGPI colorbar ranges
    engpi_max = np.nanpercentile(engpi_ctrl.values, 95)
    if engpi_max < 1e-3: engpi_max = 1.0
    engpi_max = np.ceil(engpi_max)
    CBAR_ENGPI_CTRL = {"vmin": 0.0, "vmax": engpi_max, "nlevels": 11, "cmap": "YlOrRd"}
    
    engpi_diff_max = np.nanpercentile(np.abs(engpi_diff.values), 98)
    if engpi_diff_max < 1e-3: engpi_diff_max = 0.5
    engpi_diff_max = float(f"{engpi_diff_max:.1g}")
    CBAR_ENGPI_DIFF = {"vmin": -engpi_diff_max, "vmax": engpi_diff_max, "nlevels": 11, "cmap": "RdBu_r"}
    
    # Setup Figure
    fig = plt.figure(figsize=(12, 16)) 
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.25)
    
    # (a) DGPI Climatology
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    cf0 = plot_map_on_ax(ax0, dgpi_ctrl, CBAR_CTRL, "DGPI Climatology", "(a)")
    cb0 = plt.colorbar(cf0, ax=ax0, orientation='horizontal', pad=CBAR_PAD, aspect=30)
    
    # (b) ENGPI Climatology (Uses same CBAR_CTRL)
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    cf1 = plot_map_on_ax(ax1, engpi_ctrl, CBAR_CTRL, "ENGPI Climatology", "(b)")
    cb1 = plt.colorbar(cf1, ax=ax1, orientation='horizontal', pad=CBAR_PAD, aspect=30)
    
    # (c) DGPI Change
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    cf2 = plot_map_on_ax(ax2, dgpi_diff, CBAR_DIFF, "DGPI Difference (GGW-CTL)", "(c)")
    cb2 = plt.colorbar(cf2, ax=ax2, orientation='horizontal', pad=CBAR_PAD, aspect=30)
    
    # (d) ENGPI Change (Uses same CBAR_DIFF)
    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    cf3 = plot_map_on_ax(ax3, engpi_diff, CBAR_DIFF, "ENGPI Difference (GGW-CTL)", "(d)")
    cb3 = plt.colorbar(cf3, ax=ax3, orientation='horizontal', pad=CBAR_PAD, aspect=30)
    
    # --- Bar Charts ---
    # Calculate totals from maps
    lon_dgpi, lat_dgpi = ds['lon_dgpi'].values, ds['lat_dgpi'].values
    lon_engpi, lat_engpi = ds['lon_engpi'].values, ds['lat_engpi'].values
    
    mask_lon_dgpi, mask_lat_dgpi = region_indices(lon_dgpi, lat_dgpi, BOX_LON_MIN, BOX_LON_MAX, BOX_LAT_MIN, BOX_LAT_MAX)
    mask_lon_engpi, mask_lat_engpi = region_indices(lon_engpi, lat_engpi, BOX_LON_MIN, BOX_LON_MAX, BOX_LAT_MIN, BOX_LAT_MAX)

    dgpi_total = np.nanmean(dgpi_diff.values[np.ix_(mask_lat_dgpi, mask_lon_dgpi)])
    engpi_total = np.nanmean(engpi_diff.values[np.ix_(mask_lat_engpi, mask_lon_engpi)])
    
    # Use total from file if available (consistent with Shapley sum)
    if 'engpi_total_diff' in ds:
        engpi_total = float(ds['engpi_total_diff'].values)
    
    # --- Prepare Data with Reordering ---
    # Desired Orders
    # DGPI: VWS, VORT, MZW, Omega
    # ENGPI: VWS, VORT, PI, RH  (corresponding to Vshear, eta, Vpot, H_600)
    
    # --- DGPI ---
    target_dgpi = ['VWS', 'VORT', 'MZW', 'Omega']
    dgpi_dict = dict(zip(dgpi_factors, dgpi_contribs))
    
    if all(k in dgpi_dict for k in target_dgpi):
        labels_dgpi = ['Total'] + target_dgpi
        vals_dgpi = [dgpi_total] + [dgpi_dict[k] for k in target_dgpi]
    else:
        print("Warning: DGPI factor names mismatch. Using original.")
        labels_dgpi = ['Total'] + list(dgpi_factors)
        vals_dgpi = [dgpi_total] + list(dgpi_contribs)

    # --- ENGPI ---
    engpi_dict = dict(zip(engpi_factors, engpi_contribs))
    
    # Determine which RH key exists
    rh_key = 'H_600'
    if 'H_700' in engpi_dict:
        rh_key = 'H_700'
    
    # User requested order: Total, VWS, VORT, PI, RH
    target_engpi_keys = ['Vshear', 'eta', 'Vpot', rh_key]
    target_engpi_labels = ['VWS', 'VORT', 'PI', 'RH']
    
    if all(k in engpi_dict for k in target_engpi_keys):
        labels_engpi = ['Total'] + target_engpi_labels
        vals_engpi = [engpi_total] + [engpi_dict[k] for k in target_engpi_keys]
    else:
        print("Warning: ENGPI factor names mismatch. Using original.")
        rename_map = {'Vpot': 'PI', 'Vshear': 'VWS', 'eta': 'VORT', 'H_600': 'RH', 'H_700': 'RH'}
        labels_engpi = ['Total'] + [rename_map.get(f, f) for f in engpi_factors]
        vals_engpi = [engpi_total] + list(engpi_contribs)
    
    # Determine Y-Axis Limits (Common)
    all_vals = vals_dgpi + vals_engpi
    ymin, ymax = min(all_vals), max(all_vals)
    range_y = ymax - ymin
    if range_y == 0: range_y = 1.0
    ymin -= range_y * 0.1
    ymax += range_y * 0.1
    
    # (e) DGPI Bar Chart
    ax4 = fig.add_subplot(gs[2, 0])
    x_dgpi = np.arange(len(labels_dgpi))
    colors_dgpi = ['gray'] + ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(labels_dgpi)-1]
    ax4.bar(x_dgpi, vals_dgpi, color=colors_dgpi, edgecolor='k', width=BAR_WIDTH)
    ax4.axhline(0, color='k', linewidth=0.8)
    ax4.set_xticks(x_dgpi)
    ax4.set_xticklabels(labels_dgpi, rotation=45, ha='right')
    ax4.set_title("(e) DGPI Decomposition", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    ax4.set_ylim(ymin, ymax)
    
    # (f) ENGPI Bar Chart
    ax5 = fig.add_subplot(gs[2, 1])
    x_engpi = np.arange(len(labels_engpi))
    colors_engpi = ['gray'] + ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'][:len(labels_engpi)-1]
    ax5.bar(x_engpi, vals_engpi, color=colors_engpi, edgecolor='k', width=BAR_WIDTH)
    ax5.axhline(0, color='k', linewidth=0.8)
    ax5.set_xticks(x_engpi)
    ax5.set_xticklabels(labels_engpi, rotation=45, ha='right')
    ax5.set_title("(f) ENGPI Decomposition", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    ax5.set_ylim(-0.01, 0.2)
    
    # Style Bar Axes
    for ax in [ax4, ax5]:
        for spine in ax.spines.values():
            spine.set_linewidth(AXES_FRAME_LINEWIDTH)
            spine.set_edgecolor('black')
        ax.tick_params(axis='both', direction='out', length=TICK_LENGTH_MAJOR, width=AXES_FRAME_LINEWIDTH)
    
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=600)
    print(f"Saved figure to {OUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()
