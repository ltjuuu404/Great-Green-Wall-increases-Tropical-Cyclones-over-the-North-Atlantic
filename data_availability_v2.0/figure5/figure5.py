
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib as mpl
import os
from pathlib import Path

# Set global font
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 16
mpl.rcParams['savefig.dpi'] = 600

# Paths
DATA_DIR = Path(__file__).resolve().parent
# OUTPUT_DIR = 
# OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'figure9_replot.png')

# Plotting Extent
# NOTE: The data in the NetCDF files is pre-sliced to lat=[0, 60] and lon=[-30, 60].
# You can reduce this range (zoom in) but expanding it will show blank areas.
EXTENT = [-30, 50, 0, 50] 

# Significance
SIGNIFICANCE_LEVEL = 0.05

# Colorbar Padding (controls distance between subplot and colorbar)
CBAR_PAD = 0.15

# Levels and Colormaps
ALBEDO_LEVELS = np.linspace(0, 0.6, 21)
ALBEDO_CMAP = 'Blues'

ALBEDO_DIFF_LEVELS = np.linspace(-0.15, 0.15, 11)
ALBEDO_DIFF_CMAP = 'bwr'

LH_DIFF_LEVELS = np.linspace(-20, 20, 11)
LH_DIFF_CMAP = 'bwr'

FSNS_DIFF_LEVELS = np.linspace(-20, 20, 21)
FSNS_DIFF_CMAP = 'RdBu_r'

SHFLX_DIFF_LEVELS = np.linspace(-5, 5, 21)
SHFLX_DIFF_CMAP = 'RdBu_r'

# Figure Layout
FIGSIZE = (12, 18)
GRID_HSPACE = 0.3
GRID_WSPACE = 0.1

def format_lon_deg(x, pos=None):
    try:
        xv = float(x)
    except Exception:
        return ''
    if np.isclose(xv, 0.0):
        return "0°"
    direction = 'E' if xv > 0 else 'W'
    return f"{int(abs(round(xv)))}°{direction}"

def format_lat_deg(y, pos=None):
    try:
        yv = float(y)
    except Exception:
        return ''
    if np.isclose(yv, 0.0):
        return "0°"
    direction = 'N' if yv > 0 else 'S'
    return f"{int(abs(round(yv)))}°{direction}"

def make_axes_bold(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_edgecolor('black')
        spine.set_zorder(100)

def plot_subplot(ax, lons, lats, data, levels, cmap, pval=None, title="", unit=""):
    # Apply Extent
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    cf = ax.contourf(lons, lats, data, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
    
    if pval is not None:
        sig_mask = (pval < SIGNIFICANCE_LEVEL)
        if sig_mask.any():
            cs = ax.contourf(lons, lats, sig_mask.astype(int), levels=[0.5, 1.5],
                        colors='none', hatches=['///'], transform=ccrs.PlateCarree(), zorder=10)
            for collection in cs.collections:
                collection.set_edgecolor('white')
                collection.set_linewidth(0.5)
                collection.set_facecolor('none')
    
    # Ticks Logic
    lon_min, lon_max, lat_min, lat_max = EXTENT
    
    # Determine Major Ticks (every 30 deg Lon, 20 deg Lat)
    major_xticks = np.arange(np.floor(lon_min/10)*10, np.ceil(lon_max/10)*10 + 1, 10)
    major_xticks = major_xticks[major_xticks % 30 == 0]
    major_yticks = np.arange(np.floor(lat_min/10)*10, np.ceil(lat_max/10)*10 + 1, 10)
    major_yticks = major_yticks[major_yticks % 20 == 0]
    
    # Filter Major Ticks to Extent
    major_xticks = major_xticks[(major_xticks >= lon_min - 0.1) & (major_xticks <= lon_max + 0.1)]
    major_yticks = major_yticks[(major_yticks >= lat_min - 0.1) & (major_yticks <= lat_max + 0.1)]

    # Determine Minor Ticks (Interlaced between major ticks)
    # Strategy: Place minor ticks at midpoints of major intervals, or fixed offset
    # If major spacing is 30, midpoints are at +15.
    # If major spacing is 20, midpoints are at +10.
    
    # Generate potential minor ticks
    all_xticks_15 = np.arange(np.floor(lon_min/15)*15, np.ceil(lon_max/15)*15 + 1, 15)
    minor_xticks = np.setdiff1d(all_xticks_15, major_xticks) # Exclude major ticks
    
    all_yticks_10 = np.arange(np.floor(lat_min/10)*10, np.ceil(lat_max/10)*10 + 1, 10)
    minor_yticks = np.setdiff1d(all_yticks_10, major_yticks)

    # Filter Minor Ticks
    minor_xticks = minor_xticks[(minor_xticks >= lon_min - 0.1) & (minor_xticks <= lon_max + 0.1)]
    minor_yticks = minor_yticks[(minor_yticks >= lat_min - 0.1) & (minor_yticks <= lat_max + 0.1)]

    # Set Ticks
    ax.set_xticks(major_xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(major_yticks, crs=ccrs.PlateCarree())
    ax.set_xticks(minor_xticks, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(minor_yticks, minor=True, crs=ccrs.PlateCarree())

    # Formatting
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon_deg))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat_deg))
    
    # Tick Styling: Bold Major, Normal Minor
    ax.tick_params(axis='both', which='major', labelsize=14, width=2.5, length=7)
    ax.tick_params(axis='both', which='minor', width=1.0, length=4)
    
    make_axes_bold(ax)
    
    ax.set_title(title, loc='left', fontsize=16, fontweight='bold')
    
    return cf

def load_var(filename, var_name):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None
    
    ds = xr.open_dataset(path)
    data = ds[var_name]
    pval = ds['p_value'] if 'p_value' in ds else None
    
    return data, pval

def get_cbar_ticks(vmin, vmax, n_ticks=5):
    return np.linspace(vmin, vmax, n_ticks)

def fmt_albedo(x, pos):
    if np.isclose(x, 0):
        return "0"
    return f"{x:.2f}"

def fmt_int(x, pos):
    return f"{int(round(x))}"

def main():
    # Plotting
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(3, 2, hspace=GRID_HSPACE, wspace=GRID_WSPACE)
    
    axs = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    ]
    
    # A: Albedo CTL
    alb_ctl, _ = load_var("albedo_ctl.nc", "albedo_ctl")
    if alb_ctl is not None:
        lons = alb_ctl.lon
        lats = alb_ctl.lat
        cf_a = plot_subplot(axs[0], lons, lats, alb_ctl, ALBEDO_LEVELS, ALBEDO_CMAP, None, "(a) Albedo CTL", "")
        cb_a = plt.colorbar(cf_a, ax=axs[0], orientation='horizontal', shrink=0.8, pad=CBAR_PAD, ticks=get_cbar_ticks(0, 0.6))
        cb_a.ax.tick_params(labelsize=12)
    
    # B: Albedo Diff
    alb_diff, alb_pval = load_var("albedo_diff.nc", "albedo_diff")
    if alb_diff is not None:
        lons = alb_diff.lon
        lats = alb_diff.lat
        cf_b = plot_subplot(axs[1], lons, lats, alb_diff, ALBEDO_DIFF_LEVELS, ALBEDO_DIFF_CMAP, alb_pval, "(b) Albedo Diff", "")
        cb_b = plt.colorbar(cf_b, ax=axs[1], orientation='horizontal', shrink=0.8, pad=CBAR_PAD, ticks=get_cbar_ticks(-0.15, 0.15))
        cb_b.ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_albedo))
        cb_b.ax.tick_params(labelsize=12)

    # C: FSNS Diff
    fsns_diff, fsns_pval = load_var("fsns_diff.nc", "fsns_diff")
    if fsns_diff is not None:
        lons = fsns_diff.lon
        lats = fsns_diff.lat
        cf_c = plot_subplot(axs[2], lons, lats, fsns_diff, FSNS_DIFF_LEVELS, FSNS_DIFF_CMAP, fsns_pval, "(c) Net Shortwave Diff", r'W m$^{-2}$')
        cb_c = plt.colorbar(cf_c, ax=axs[2], orientation='horizontal', shrink=0.8, pad=CBAR_PAD, label=r'W m$^{-2}$', ticks=get_cbar_ticks(-20, 20))
        cb_c.ax.tick_params(labelsize=12)

    # D: LHFLX Diff
    lh_diff, lh_pval = load_var("latent_heat_diff.nc", "latent_heat_diff")
    if lh_diff is not None:
        lons = lh_diff.lon
        lats = lh_diff.lat
        cf_d = plot_subplot(axs[3], lons, lats, lh_diff, LH_DIFF_LEVELS, LH_DIFF_CMAP, lh_pval, "(d) Latent Heat Diff", r'W m$^{-2}$')
        cb_d = plt.colorbar(cf_d, ax=axs[3], orientation='horizontal', shrink=0.8, pad=CBAR_PAD, label=r'W m$^{-2}$', ticks=get_cbar_ticks(-20, 20))
        cb_d.ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_int))
        cb_d.ax.tick_params(labelsize=12)

    # E: SHFLX Diff
    shflx_diff, shflx_pval = load_var("sensible_heat_diff.nc", "sensible_heat_diff")
    if shflx_diff is not None:
        lons = shflx_diff.lon
        lats = shflx_diff.lat
        cf_e = plot_subplot(axs[4], lons, lats, shflx_diff, SHFLX_DIFF_LEVELS, SHFLX_DIFF_CMAP, shflx_pval, "(e) Sensible Heat Diff", r'W m$^{-2}$')
        cb_e = plt.colorbar(cf_e, ax=axs[4], orientation='horizontal', shrink=0.8, pad=CBAR_PAD, label=r'W m$^{-2}$', ticks=get_cbar_ticks(-5, 5))
        cb_e.ax.tick_params(labelsize=12)

    # Hide unused subplot (2, 1)
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')

    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
        
    # plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    # print(f"Figure saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    main()
