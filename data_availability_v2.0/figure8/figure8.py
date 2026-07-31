#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
box_lon_min, box_lon_max = -70.0, -17.5
box_lat_min, box_lat_max = 10.0, 20.0

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("cartopy not available; plotting without coastlines/land-mask.")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Plotting Parameters
LON_MIN, LON_MAX = -90.0, 30.0
LAT_MIN, LAT_MAX = 0.0, 40.0
CENTRAL_LON = (LON_MIN + LON_MAX) / 2.0
GRID_RES = 2.5

CB_N_BINS = 11
CB_N_BINS_TOP = 15
FIG_DPI = 180
TOP_CBAR_VMIN = 0
TOP_CBAR_VMAX = 0.15
DIFF_CBAR_VMIN = -0.03
DIFF_CBAR_VMAX = 0.03
CB_WIDTH = 0.01
CB_GAP = 0.01
CB1_Y_SHIFT = 0.05
CB2_Y_SHIFT = 0.015
DIFF_CBAR_SCALE = 0.92

out_fig = "figure8.svg"
PALETTE = ['white', '#fff7bc', '#fee391', '#fdae61', '#f46d43', '#d73027', '#a50026']
CMAP_DIFF_NAME = 'RdBu_r'

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
TITLE_FONTSIZE = 28
OTHER_FONTSIZE = 26
ANNOT_FONTNAME = "Times New Roman"
ANNOT_FONTSIZE = 24
FIG_WIDTH = 12
FIG_HEIGHT = 11
BOTTOM_MARGIN = 0.12
axes_frame_linewidth = 1.5

# ---------------- Helper Functions ----------------

def make_global_grid(resolution=5.0, central_lon=0.0):
    lon_start = central_lon - 180.0
    lon_end = central_lon + 180.0
    lon_edges = np.arange(lon_start, lon_end + resolution, resolution)
    lat_edges = np.arange(-90.0, 90.0 + resolution, resolution)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    return lon_centers, lat_centers, lon_edges, lat_edges

def normalize_lon(lon, central_lon):
    lon_arr = np.asarray(lon, dtype=float)
    out = lon_arr.copy()
    out = (out - central_lon + 180.0) % 360.0 - 180.0 + central_lon
    return out

def normalize_lon_to_180(lon):
    lon_arr = np.asarray(lon, dtype=float)
    return (lon_arr + 180.0) % 360.0 - 180.0

def normalize_lon_to_center(lon, central_lon):
    return normalize_lon(lon, central_lon)

def format_lat_label(lat):
    if np.isclose(lat, 0.0):
        return "0°"
    elif lat > 0:
        return f"{int(np.round(lat))}°N"
    else:
        return f"{int(np.round(abs(lat)))}°S"

def format_lon_label(lon):
    lon_norm = normalize_lon_to_180(lon)
    if np.isclose(lon_norm, 0.0):
        return "0°"
    elif lon_norm > 0:
        return f"{int(np.round(lon_norm))}°E"
    else:
        return f"{int(np.round(abs(lon_norm)))}°W"

# ---------------- Plotting Function ----------------

def plot_jas_control_sensitive_diff(mean_ctrl_full, mean_sens_full,
                                   lon_min, lon_max, lat_min, lat_max,
                                   lon_edges_full, lat_edges_full,
                                   out_png, central_lon=0.0, grid_res=5.0,
                                   obs_genesis_points=None):
    
    grid_lon_start = central_lon - 180.0
    grid_lat_start = -90.0
    lon_idx0 = int(round((lon_min - grid_lon_start) / grid_res))
    lon_idx1 = int(round((lon_max - grid_lon_start) / grid_res))
    lat_idx0 = int(round((lat_min - grid_lat_start) / grid_res))
    lat_idx1 = int(round((lat_max - grid_lat_start) / grid_res))
    lon_idx0 = max(0, lon_idx0)
    lat_idx0 = max(0, lat_idx0)
    lon_idx1 = min(len(lon_edges_full) - 1, lon_idx1)
    lat_idx1 = min(len(lat_edges_full) - 1, lat_idx1)
    
    data_ctrl = mean_ctrl_full[lat_idx0:lat_idx1, lon_idx0:lon_idx1]
    data_sens = mean_sens_full[lat_idx0:lat_idx1, lon_idx0:lon_idx1]
    nx = data_ctrl.shape[1]
    ny = data_ctrl.shape[0]
    
    Lon_e = np.linspace(lon_min, lon_max, nx + 1)
    Lat_e = np.linspace(lat_min, lat_max, ny + 1)
    Lon_c = (Lon_e[:-1] + Lon_e[1:]) / 2.0
    Lat_c = (Lat_e[:-1] + Lat_e[1:]) / 2.0
    Lon_C, Lat_C = np.meshgrid(Lon_c, Lat_c)
    
    vals_top = np.concatenate([data_ctrl.ravel(), data_sens.ravel()])
    finite_top = vals_top[np.isfinite(vals_top)]
    if finite_top.size > 0:
        if TOP_CBAR_VMIN is None:
            if np.nanmin(finite_top) >= 0: vmin_top = 0.0
            else: vmin_top = float(np.nanmin(finite_top))
        else: vmin_top = float(TOP_CBAR_VMIN)
        if TOP_CBAR_VMAX is None: vmax_top = float(np.nanmax(finite_top))
        else: vmax_top = float(TOP_CBAR_VMAX)
    else:
        vmin_top = 0.0 if TOP_CBAR_VMIN is None else float(TOP_CBAR_VMIN)
        vmax_top = 1.0 if TOP_CBAR_VMAX is None else float(TOP_CBAR_VMAX)
    if vmin_top > vmax_top: vmin_top, vmax_top = vmax_top, vmin_top
    if np.isclose(vmin_top, vmax_top): vmin_top -= 1e-6; vmax_top += 1e-6
    
    boundaries_top = np.linspace(vmin_top, vmax_top, CB_N_BINS_TOP + 1)
    try:
        cmap_top_lin = LinearSegmentedColormap.from_list('white_warm_interp', PALETTE, N=CB_N_BINS_TOP)
        colors = cmap_top_lin(np.linspace(0, 1, CB_N_BINS_TOP))
        colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
        cmap_top = ListedColormap(colors)
        cmap_top.set_bad('white')
    except Exception:
        base_colors = PALETTE[:]
        if base_colors[0].lower() != 'white': base_colors = ['white'] + base_colors
        cmap_top = ListedColormap(base_colors)
        cmap_top.set_bad('white')
    norm_top = BoundaryNorm(boundaries_top, ncolors=cmap_top.N, clip=False)
    
    diff = data_sens - data_ctrl
    finite_diff = diff.ravel()[np.isfinite(diff.ravel())]
    if finite_diff.size > 0: auto_maxabs = float(np.nanmax(np.abs(finite_diff)))
    else: auto_maxabs = max(abs(vmin_top), abs(vmax_top), 1.0)
    
    if DIFF_CBAR_VMIN is None and DIFF_CBAR_VMAX is None:
        maxabs = auto_maxabs
        vmin_diff = -maxabs; vmax_diff = maxabs
    elif DIFF_CBAR_VMIN is None and DIFF_CBAR_VMAX is not None:
        vmax_diff = abs(float(DIFF_CBAR_VMAX)); vmin_diff = -vmax_diff
    elif DIFF_CBAR_VMIN is not None and DIFF_CBAR_VMAX is None:
        vmin_diff = float(DIFF_CBAR_VMIN); vmax_diff = abs(vmin_diff); vmin_diff = -abs(vmin_diff)
    else:
        vmin_diff = float(DIFF_CBAR_VMIN); vmax_diff = float(DIFF_CBAR_VMAX)
    if vmin_diff > vmax_diff: vmin_diff, vmax_diff = vmax_diff, vmin_diff
    if np.isclose(vmin_diff, vmax_diff): vmin_diff -= 1e-6; vmax_diff += 1e-6
    
    boundaries_diff = np.linspace(vmin_diff, vmax_diff, CB_N_BINS + 1)
    try:
        cmap_diff_full = plt.get_cmap(CMAP_DIFF_NAME)
        diff_colors = cmap_diff_full(np.linspace(0, 1, CB_N_BINS))
        cmap_diff = ListedColormap(diff_colors)
        cmap_diff.set_bad('white')
    except Exception:
        cmap_diff = plt.get_cmap(CMAP_DIFF_NAME)
        try: cmap_diff.set_bad('white')
        except: pass
    norm_diff = BoundaryNorm(boundaries_diff, ncolors=CB_N_BINS, clip=False)
    
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    HSPACE = 0.45
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.0, 1.0], hspace=HSPACE)
    if HAS_CARTOPY:
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
        ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
        ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
    else:
        ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[2, 0])
        
    # plot control
    if HAS_CARTOPY:
        cf1 = ax1.contourf(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax1.contour(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try: ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except: pass
        ax1.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf1 = ax1.contourf(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', zorder=1)
        ax1.contour(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax1.set_xlim(lon_min, lon_max); ax1.set_ylim(lat_min, lat_max)
    ax1.set_title("(a) CTL", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')

    if obs_genesis_points is not None and len(obs_genesis_points) > 0:
        obs_lons = np.array([p['lon'] for p in obs_genesis_points])
        obs_lats = np.array([p['lat'] for p in obs_genesis_points])
        obs_lons_plot = normalize_lon(obs_lons, central_lon)
        try:
            ax1.scatter(obs_lons_plot, obs_lats, s=25, edgecolors=(0, 0, 0, 0.8), facecolors='black',
                        linewidths=0, alpha=0.6, transform=ccrs.PlateCarree(), zorder=6)
        except Exception:
            ax1.scatter(obs_lons_plot, obs_lats, s=60, edgecolors='k', facecolors='black', linewidths=1.2, zorder=6)

    # plot sensitive
    if HAS_CARTOPY:
        cf2 = ax2.contourf(Lon_C, Lat_C, data_sens, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax2.contour(Lon_C, Lat_C, data_sens, levels=boundaries_top, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try: ax2.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except: pass
        ax2.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf2 = ax2.contourf(Lon_C, Lat_C, data_sens, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', zorder=1)
        ax2.contour(Lon_C, Lat_C, data_sens, levels=boundaries_top, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax2.set_xlim(lon_min, lon_max); ax2.set_ylim(lat_min, lat_max)
    ax2.set_title("(b) GGW", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')

    # plot diff
    if HAS_CARTOPY:
        cf3 = ax3.contourf(Lon_C, Lat_C, diff, levels=boundaries_diff, cmap=cmap_diff, norm=norm_diff,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax3.contour(Lon_C, Lat_C, diff, levels=boundaries_diff, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try: ax3.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except: pass
        ax3.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax3.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf3 = ax3.contourf(Lon_C, Lat_C, diff, levels=boundaries_diff, cmap=cmap_diff, norm=norm_diff,
                           extend='both', zorder=1)
        ax3.contour(Lon_C, Lat_C, diff, levels=boundaries_diff, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax3.set_xlim(lon_min, lon_max); ax3.set_ylim(lat_min, lat_max)
    ax3.set_title("(c) GGW-CTL", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')

    box_w = box_lon_max - box_lon_min
    box_h = box_lat_max - box_lat_min
    if HAS_CARTOPY:
        rect_box = mpl.patches.Rectangle(
            (box_lon_min, box_lat_min), box_w, box_h,
            linewidth=2.5, edgecolor='black', facecolor='none',
            transform=ccrs.PlateCarree(), zorder=20
        )
        ax3.add_patch(rect_box)
    else:
        rect_box = mpl.patches.Rectangle(
            (box_lon_min, box_lat_min), box_w, box_h,
            linewidth=2.5, edgecolor='black', facecolor='none', zorder=20
        )
        ax3.add_patch(rect_box)

    for ax in (ax1, ax2, ax3):
        try:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(axes_frame_linewidth)
                spine.set_edgecolor('black')
        except: pass

    step = 30.0
    xticks_raw = np.arange(int(np.ceil(lon_min / step)) * int(step),
                           int(np.floor(lon_max / step)) * int(step) + 1, int(step)).astype(float)
    if xticks_raw.size == 0: xticks_raw = np.array([lon_min, lon_max], dtype=float)
    xticks_plot = normalize_lon_to_center(xticks_raw, central_lon)
    order = np.argsort(xticks_plot)
    xticks_plot_sorted = xticks_plot[order]
    xticks_raw_sorted = xticks_raw[order]

    y_major0 = int(np.ceil(lat_min / 10.0) * 10)
    y_major1 = int(np.floor(lat_max / 10.0) * 10)
    if y_major0 > y_major1: y_major = np.array([lat_min, lat_max])
    else: y_major = np.arange(y_major0, y_major1 + 1, 10)
    y_minor0 = int(np.ceil(lat_min / 5.0) * 5)
    y_minor1 = int(np.floor(lat_max / 5.0) * 5)
    if y_minor0 > y_minor1: y_minor = np.array([])
    else: y_minor = np.arange(y_minor0, y_minor1 + 1, 5)
    if (lat_min <= 0.0 <= lat_max) and (0 not in y_major):
        y_major = np.unique(np.sort(np.concatenate((y_major, [0]))))
        y_major = np.sort(y_major)

    xlabels_display = []
    for i, x_raw in enumerate(xticks_raw_sorted):
        if i % 2 == 0: xlabels_display.append(format_lon_label(x_raw))
        else: xlabels_display.append('')

    for ax in (ax1, ax2, ax3):
        try:
            ax.set_xticks(xticks_plot_sorted, crs=ccrs.PlateCarree())
            ax.set_yticks(y_major, crs=ccrs.PlateCarree())
            ax.set_yticks(y_minor, minor=True, crs=ccrs.PlateCarree())
        except Exception:
            ax.set_xticks(xticks_plot_sorted)
            ax.set_yticks(y_major)
            ax.set_yticks(y_minor, minor=True)

        ylabels_major = [format_lat_label(y) for y in y_major]
        ax.set_yticklabels(ylabels_major, fontsize=OTHER_FONTSIZE - 6, family='Times New Roman')
        ax.tick_params(axis='y', which='major', labelleft=True, labelsize=OTHER_FONTSIZE - 6,
                       direction='out', length=10, width=axes_frame_linewidth)
        ax.tick_params(axis='y', which='minor', labelleft=False, direction='out', length=5,
                       width=axes_frame_linewidth * 0.5)

        try:
            ax.set_xticklabels(xlabels_display, fontsize=OTHER_FONTSIZE - 6, family='Times New Roman')
        except Exception: pass
        ax.tick_params(axis='x', labelbottom=True, which='major', direction='out',
                       length=10, width=axes_frame_linewidth, pad=10)

    if len(xticks_raw_sorted) > 1:
        xtick_minors_raw = (xticks_raw_sorted[:-1] + xticks_raw_sorted[1:]) / 2.0
        xtick_minors_plot = normalize_lon_to_center(xtick_minors_raw, central_lon)
        for ax in (ax1, ax2, ax3):
            try: ax.set_xticks(xtick_minors_plot, minor=True, crs=ccrs.PlateCarree())
            except Exception: ax.set_xticks(xtick_minors_plot, minor=True)
            ax.tick_params(axis='x', which='minor', labelbottom=False, direction='out', length=5)

    for ax in (ax1, ax2, ax3):
        if HAS_CARTOPY:
            gl = ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.4, color='gray', alpha=0.6)
            try:
                gl.xlocator = mticker.FixedLocator(xticks_plot_sorted)
                gl.ylocator = mticker.FixedLocator(y_major)
            except Exception: pass
        else:
            ax.grid(which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.6)

    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    axes_right = max(pos1.x1, pos2.x1, pos3.x1)
    center1 = 0.5 * (pos1.y0 + pos1.y1)
    center2 = 0.5 * (pos2.y0 + pos2.y1)
    combined_center = 0.5 * (center1 + center2)
    combined_top = pos1.y1
    combined_bottom = pos2.y0
    combined_span = combined_top - combined_bottom
    cbar1_h = max(0.001, combined_span * 0.92) * 1.2
    cbar1_y = combined_center - 0.5 * cbar1_h
    cbar1_y += CB1_Y_SHIFT
    min_y = 0.005; max_y = 0.995 - cbar1_h
    if cbar1_y < min_y: cbar1_y = min_y
    if cbar1_y > max_y: cbar1_y = max_y
    cbar1_x = axes_right + CB_GAP
    if cbar1_x + CB_WIDTH > 0.98: cbar1_x = axes_right - CB_GAP - CB_WIDTH
    cax1 = fig.add_axes([cbar1_x, cbar1_y, CB_WIDTH, cbar1_h])
    try:
        cb1 = fig.colorbar(cf1, cax=cax1, orientation='vertical', extend='both', boundaries=boundaries_top)
    except Exception:
        cb1 = fig.colorbar(ScalarMappable(norm=norm_top, cmap=cmap_top), cax=cax1,
                           orientation='vertical', extend='both')
    cb1.ax.tick_params(labelsize=OTHER_FONTSIZE - 4)
    try:
        tick_locs = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
        cb1.set_ticks(tick_locs)
        labels = []
        for v in tick_locs:
            if np.isclose(v, 0): labels.append("0")
            else: labels.append(f"{v:.2f}")
        cb1.set_ticklabels(labels)
    except Exception: pass

    sub_height = pos3.y1 - pos3.y0
    cbar2_h = max(0.001, sub_height * float(DIFF_CBAR_SCALE))
    cbar2_y = pos3.y0 + (sub_height - cbar2_h) / 2.0
    cbar2_y += CB2_Y_SHIFT
    if cbar2_y < min_y: cbar2_y = min_y
    if cbar2_y + cbar2_h > 0.995: cbar2_y = 0.995 - cbar2_h
    cbar2_x = axes_right + CB_GAP
    if cbar2_x + CB_WIDTH > 0.98: cbar2_x = axes_right - CB_GAP - CB_WIDTH
    cax2 = fig.add_axes([cbar2_x, cbar2_y, CB_WIDTH, cbar2_h])
    try:
        cb2 = fig.colorbar(cf3, cax=cax2, orientation='vertical', extend='both', boundaries=boundaries_diff)
    except Exception:
        cb2 = fig.colorbar(ScalarMappable(norm=norm_diff, cmap=cmap_diff), cax=cax2,
                           orientation='vertical', extend='both')
    cb2.ax.tick_params(labelsize=OTHER_FONTSIZE - 4)
    tick_locs = [vmin_diff, 0.0, vmax_diff]
    cb2.set_ticks(tick_locs)
    labels = []
    for v in tick_locs:
        if np.isclose(v, 0): labels.append("0")
        else: labels.append(f"{float(v):.2f}")
    cb2.set_ticklabels(labels)

    plt.subplots_adjust(top=0.95, bottom=BOTTOM_MARGIN, left=0.08, right=0.88)
    try:
        fig.savefig(out_png, dpi=FIG_DPI, bbox_inches='tight')
        print(f"Saved figure to {out_png}")
    except Exception: pass
    plt.show()
    print("Displaying composite figure")

# ---------------- Main Process ----------------

def main():
    print("Loading data from", DATA_DIR)
    nc_path = os.path.join(DATA_DIR, 'figure2_data.nc')
    
    if not os.path.exists(nc_path):
        print(f"Error: {nc_path} not found.")
        return

    try:
        ds = xr.open_dataset(nc_path)
        mean_ctrl = ds['mean_ctrl'].values
        mean_sens = ds['mean_sens'].values
        
        # Load Obs points
        # Assuming they are stored as variables 'obs_lat' and 'obs_lon'
        if 'obs_lat' in ds and 'obs_lon' in ds:
            obs_lats = ds['obs_lat'].values
            obs_lons = ds['obs_lon'].values
            obs_genesis_points = []
            for la, lo in zip(obs_lats, obs_lons):
                obs_genesis_points.append({'lat': float(la), 'lon': float(lo)})
            print(f"Loaded {len(obs_genesis_points)} observation points from NC")
        else:
            obs_genesis_points = []
            print("No observation points found in NC")
            
        ds.close()
        
    except Exception as e:
        print(f"Error loading NC file: {e}")
        return

    # Reconstruct edges for plotting
    lon_centers, lat_centers, lon_edges, lat_edges = make_global_grid(GRID_RES, CENTRAL_LON)

    plot_jas_control_sensitive_diff(mean_ctrl, mean_sens,
                                   LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                                   lon_edges, lat_edges,
                                   out_fig,
                                   central_lon=CENTRAL_LON,
                                   grid_res=GRID_RES,
                                   obs_genesis_points=obs_genesis_points)

if __name__ == "__main__":
    main()
