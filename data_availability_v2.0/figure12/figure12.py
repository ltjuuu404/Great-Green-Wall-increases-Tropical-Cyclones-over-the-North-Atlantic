# -*- coding: utf-8 -*-
"""
Plots 3x1 wind anomaly for 200, 500, 850hPa.
"""
import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import matplotlib.ticker as mticker
import math
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator


input_nc_path = Path(__file__).resolve().parent / 'figure12_wind_data.nc'


OTHER_FONT_SIZE = 20
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = OTHER_FONT_SIZE

QUIVERKEY_FONTSIZE = 16
FIGSIZE = (10, 24) 
AXES_FRAME_LINEWIDTH = 2.0
TICK_MAJOR_WIDTH = AXES_FRAME_LINEWIDTH
TICK_MINOR_WIDTH = 1.0
TICK_MAJOR_LENGTH = 8
TICK_MINOR_LENGTH = 4

quiver_max_arrows = 3000
min_stride = 2
arrow_color = 'black'

lat_min, lat_max = 0., 50.
lon_min, lon_max = -90., 0.
central_lon = -10.0

def auto_downsample_quiver_params(lon, lat, u, v, max_arrows=quiver_max_arrows):
    nx = len(lon)
    ny = len(lat)
    total = max(1, nx * ny)
    factor = int(np.ceil(np.sqrt(total / max(1, max_arrows))))
    sx = max(min_stride, factor)
    sy = max(min_stride, factor)
    mag = np.sqrt(np.nan_to_num(u)**2 + np.nan_to_num(v)**2)
    mag95 = float(np.nanpercentile(mag, 95)) if np.isfinite(mag).any() else 1.0
    scale_val = max(0.1, mag95 * 8.0)
    return sx, sy, scale_val, mag95

def format_lon_label(x):
    xv = int(np.round(x))
    if xv == 0:
        return '0°'
    if xv < 0:
        return f"{abs(xv)}°W"
    return f"{xv}°E"

def format_lat_label(y):
    yv = int(np.round(y))
    if yv == 0:
        return '0°'
    if yv < 0:
        return f"{abs(yv)}°S"
    return f"{yv}°N"

def normalize_lon_to_center(lon_arr, center):
    lon = np.array(lon_arr)
    lonc = ((lon - center + 180) % 360) - 180 + center
    return lonc

def plot_data():
    print("Plotting data...")
    ds = xr.open_dataset(input_nc_path)

    lon_use = ds['lon'].values
    lat_use = ds['lat'].values


    mask_lon = (lon_use >= lon_min) & (lon_use <= lon_max)
    mask_lat = (lat_use >= lat_min) & (lat_use <= lat_max)
    
    lon_sub = lon_use[mask_lon]
    lat_sub = lat_use[mask_lat]

    fig, axes = plt.subplots(
        3, 1,
        figsize=FIGSIZE,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_lon)}
    )

    plt.subplots_adjust(left=0.08, right=0.96, top=0.95,
                        bottom=0.04, hspace=0.37, wspace=0.08)

    plot_configs = [
        (200, ds['u_diff_200'].values[mask_lat, :][:, mask_lon], ds['v_diff_200'].values[mask_lat, :][:, mask_lon], '(a) Changes in 200 hPa wind'),
        (500, ds['u_diff_500'].values[mask_lat, :][:, mask_lon], ds['v_diff_500'].values[mask_lat, :][:, mask_lon], '(b) Changes in 500 hPa wind'),
        (850, ds['u_diff_850'].values[mask_lat, :][:, mask_lon], ds['v_diff_850'].values[mask_lat, :][:, mask_lon], '(c) Changes in 850 hPa wind')
    ]

    for ax, (level, u_plot, v_plot, title) in zip(axes, plot_configs):

        outline = getattr(ax, 'outline_patch', None) or getattr(ax, 'patch', None)
        if outline is not None:
            outline.set_linewidth(AXES_FRAME_LINEWIDTH)
            outline.set_edgecolor('black')
        else:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(AXES_FRAME_LINEWIDTH)
                spine.set_edgecolor('black')

        ax.text(0.0, 1.02, title, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=OTHER_FONT_SIZE+1, fontweight='bold', color='black')

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


        sx, sy, scale_val, _ = auto_downsample_quiver_params(
            lon_sub, lat_sub, u_plot, v_plot)

        lon_q, lat_q = np.meshgrid(lon_sub[::sx], lat_sub[::sy])
        u_q = u_plot[::sy, ::sx]
        v_q = v_plot[::sy, ::sx]

        this_scale = scale_val*1.3

        Q = ax.quiver(
            lon_q, lat_q, u_q, v_q,
            transform=ccrs.PlateCarree(),
            color=arrow_color,
            scale=this_scale,
            pivot='middle',
            width=0.0025,
            headwidth=12,
            headlength=8.5,
            headaxislength=2
        )

        key_kwargs = {
            'transform': ax.transAxes,
            'labelpos': 'N', 
            'labelsep': 0.05,
            'fontproperties': {'size': QUIVERKEY_FONTSIZE, 'family': 'Times New Roman'},
            'facecolor': 'none',
            'edgecolor': arrow_color,
            'linewidth': 1
        }
        ax.quiverkey(Q, 0.90, 1.03, 1.0, '1.0 m s$^{-1}$', **key_kwargs)

        ax.coastlines(resolution='50m', linewidth=0.6)


        xticks_raw = np.arange(-180 + central_lon, 180 + central_lon + 1, 30.0)
        xticks_plot = normalize_lon_to_center(xticks_raw, central_lon)
        xticks_sorted = np.sort(xticks_plot)

        xticks_sorted = xticks_sorted[(xticks_sorted >= lon_min) & (xticks_sorted <= lon_max)]
        
        yticks = np.arange(math.ceil(lat_min/10)*10, math.floor(lat_max/10)*10 + 1, 20.0)

        minor_xticks = (xticks_sorted[:-1] + xticks_sorted[1:]) / 2.0 if xticks_sorted.size >= 2 else np.array([])
        minor_yticks = (yticks[:-1] + yticks[1:]) / 2.0 if yticks.size >= 2 else np.array([])

        ax.set_xticks(xticks_sorted, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        if minor_xticks.size:
            ax.set_xticks(minor_xticks, minor=True, crs=ccrs.PlateCarree())
        if minor_yticks.size:
            ax.set_yticks(minor_yticks, minor=True, crs=ccrs.PlateCarree())

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_lon_label(x)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: format_lat_label(y)))

        ax.tick_params(axis='x', which='major', direction='out',
                       length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                       top=False, bottom=True, labelsize=OTHER_FONT_SIZE - 4,
                       labelbottom=True)
        ax.tick_params(axis='x', which='minor', direction='out',
                       length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH,
                       top=False, bottom=True, labelbottom=True)
        ax.tick_params(axis='y', which='major', direction='out',
                       length=TICK_MAJOR_LENGTH, width=TICK_MAJOR_WIDTH,
                       left=True, right=False, labelsize=OTHER_FONT_SIZE - 4,
                       labelleft=True)
        ax.tick_params(axis='y', which='minor', direction='out',
                       length=TICK_MINOR_LENGTH, width=TICK_MINOR_WIDTH,
                       left=True, right=False, labelleft=True)

        gl = ax.gridlines(draw_labels=False, linewidth=0.35, color='gray', linestyle='--', zorder=2)



    plt.show()
    plt.close(fig)
    ds.close()

if __name__ == "__main__":
    plot_data()
    print("=== Done ===")
