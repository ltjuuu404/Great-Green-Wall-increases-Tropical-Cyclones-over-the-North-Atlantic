
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
CBAR_WIDTH_RATIO = 0.8
CBAR_HEIGHT_RATIO = 0.064  # Fixed physical height: 6.4% of each map panel.

# Levels and Colormaps
ALBEDO_LEVELS = np.linspace(0, 0.6, 21)
ALBEDO_CMAP = 'Blues'

ALBEDO_DIFF_LEVELS = np.linspace(-0.15, 0.15, 11)
ALBEDO_DIFF_CMAP = 'bwr'

LH_DIFF_LEVELS = np.linspace(-20, 20, 11)
# LHFLX/SHFLX are upward-positive in the unchanged input files.
# Reverse their mapping so increased upward loss is blue.
LH_DIFF_CMAP = 'RdBu'

FSNS_DIFF_LEVELS = np.linspace(-20, 20, 21)
FSNS_DIFF_CMAP = 'RdBu_r'

SHFLX_DIFF_LEVELS = np.linspace(-5, 5, 21)
SHFLX_DIFF_CMAP = 'RdBu'

# Figure Layout
FIGSIZE = (12, 18)
GRID_HSPACE = 0.2
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

def add_panel_colorbar(cf, ax, **kwargs):
    # Reserve exactly the original colorbar space; then control cax explicitly.
    cax, _ = mpl.colorbar.make_axes_gridspec(
        ax, orientation='horizontal', shrink=CBAR_WIDTH_RATIO, pad=CBAR_PAD)
    ax.apply_aspect()
    cax.apply_aspect()
    panel_box = ax.get_position()
    bar_box = cax.get_position()
    cax.set_box_aspect(None)
    width = panel_box.width * CBAR_WIDTH_RATIO
    height = panel_box.height * CBAR_HEIGHT_RATIO
    cax.set_position([panel_box.x0 + (panel_box.width - width) / 2,
                      bar_box.y0, width, height])
    return plt.colorbar(cf, cax=cax, orientation='horizontal', **kwargs)


def label_surface_flux_colorbar(cbar, upward_positive=False):
    """Display downward-positive anomalies without modifying source values.

    Upward-positive fields use RdBu and a reversed colorbar axis. Negating
    only the displayed ticks makes every flux bar blue/negative at the left
    and red/positive at the right, with an exact match to contour colors.
    """
    if upward_positive:
        cbar.ax.invert_xaxis()
    sign = -1.0 if upward_positive else 1.0
    def display_tick(raw_value, _position):
        value = sign * raw_value
        return '0' if np.isclose(value, 0) else f'{value:g}'
    cbar.ax.xaxis.set_major_formatter(mticker.FuncFormatter(display_tick))
    cbar.set_label(r'W m$^{-2}$', fontsize=14, labelpad=-2)
    cbar.ax.tick_params(labelsize=12)


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
    alb_ctl, _ = load_var("figure2_albedo_ctl.nc", "albedo_ctl")
    if alb_ctl is not None:
        lons = alb_ctl.lon
        lats = alb_ctl.lat
        cf_a = plot_subplot(axs[0], lons, lats, alb_ctl, ALBEDO_LEVELS, ALBEDO_CMAP, None, "(a) Albedo CTL", "")
        cb_a = add_panel_colorbar(cf_a, axs[0], ticks=get_cbar_ticks(0, 0.6))
        cb_a.ax.tick_params(labelsize=12)
    
    # B: Albedo Diff
    alb_diff, alb_pval = load_var("figure2_albedo_diff.nc", "albedo_diff")
    if alb_diff is not None:
        lons = alb_diff.lon
        lats = alb_diff.lat
        cf_b = plot_subplot(axs[1], lons, lats, alb_diff, ALBEDO_DIFF_LEVELS, ALBEDO_DIFF_CMAP, alb_pval, "(b) Albedo Diff", "")
        cb_b = add_panel_colorbar(cf_b, axs[1], ticks=get_cbar_ticks(-0.15, 0.15))
        cb_b.ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_albedo))
        cb_b.ax.tick_params(labelsize=12)

    # C: FSNS Diff
    fsns_diff, fsns_pval = load_var("figure2_fsns_diff.nc", "fsns_diff")
    if fsns_diff is not None:
        lons = fsns_diff.lon
        lats = fsns_diff.lat
        cf_c = plot_subplot(axs[2], lons, lats, fsns_diff, FSNS_DIFF_LEVELS, FSNS_DIFF_CMAP, fsns_pval, "(c) Net Shortwave Diff", r'W m$^{-2}$')
        cb_c = add_panel_colorbar(cf_c, axs[2], label=r'W m$^{-2}$', ticks=get_cbar_ticks(-20, 20))
        label_surface_flux_colorbar(cb_c, upward_positive=False)

    # D: LHFLX Diff
    lh_diff, lh_pval = load_var("figure2_latent_heat_diff.nc", "latent_heat_diff")
    if lh_diff is not None:
        lons = lh_diff.lon
        lats = lh_diff.lat
        cf_d = plot_subplot(axs[3], lons, lats, lh_diff, LH_DIFF_LEVELS, LH_DIFF_CMAP, lh_pval, "(d) Latent Heat Diff", r'W m$^{-2}$')
        cb_d = add_panel_colorbar(cf_d, axs[3], label=r'W m$^{-2}$', ticks=get_cbar_ticks(-20, 20))
        label_surface_flux_colorbar(cb_d, upward_positive=True)

    # E: SHFLX Diff
    shflx_diff, shflx_pval = load_var("figure2_sensible_heat_diff.nc", "sensible_heat_diff")
    if shflx_diff is not None:
        lons = shflx_diff.lon
        lats = shflx_diff.lat
        cf_e = plot_subplot(axs[4], lons, lats, shflx_diff, SHFLX_DIFF_LEVELS, SHFLX_DIFF_CMAP, shflx_pval, "(e) Sensible Heat Diff", r'W m$^{-2}$')
        cb_e = add_panel_colorbar(cf_e, axs[4], label=r'W m$^{-2}$', ticks=get_cbar_ticks(-5, 5))
        label_surface_flux_colorbar(cb_e, upward_positive=True)

    # Hide unused subplot (2, 1)
    ax_empty = fig.add_subplot(gs[2, 1])
    ax_empty.axis('off')

    # if not os.path.exists(OUTPUT_DIR):
    #     os.makedirs(OUTPUT_DIR)
        
    # plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    # print(f"Figure saved to {OUTPUT_FILE}")
    # Export the same figure beside the copied data and script.
    for suffix in ('png', 'pdf', 'svg'):
        target = DATA_DIR / f'figure2_surface_energy_sign_corrected.{suffix}'
        fig.savefig(target, bbox_inches='tight', facecolor='white')
        print(f'Saved: {target}')
    plt.show()

if __name__ == "__main__":
    main()
