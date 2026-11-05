
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

mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
# White padding around all text and graphics when saving from the figure window.
FIGURE_PADDING_INCHES = 0.15
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = FIGURE_PADDING_INCHES
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 27
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 27
plt.rcParams['xtick.labelsize'] = 27
plt.rcParams['ytick.labelsize'] = 27
plt.rcParams['legend.fontsize'] = 27

QUIVERKEY_FONTSIZE = 24
ROW_VKEY_VAL_ROW12 = 5.0
FIGSIZE = (10, 16)
N_TICKS = 6
SNAP_TO_ROUND = False
DIFF_SCALE_FACTOR = 1.8
AXES_FRAME_LINEWIDTH = 2.0
TICK_MAJOR_WIDTH = AXES_FRAME_LINEWIDTH
TICK_MINOR_WIDTH = 1.0
TICK_MAJOR_LENGTH = 8
TICK_MINOR_LENGTH = 4
quiver_max_arrows = 3000
min_stride = 2
arrow_color_ctrl = 'black'
arrow_color_diff = 'black'

def lon_formatter_deg(x, pos=None):
    try:
        xv = float(x)
    except Exception:
        return ""
    iv = int(np.round(xv))
    if SNAP_TO_ROUND:
        iv = int(round(xv))
    if iv == 0:
        return "0°"
    if iv < 0:
        return f"{abs(iv)}°W"
    return f"{iv}°E"

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

def main():
    data_path = Path(__file__).resolve().parent / 'figure8_data.nc'
    ds = xr.open_dataset(data_path)

    u_c = ds['u_control'].values
    v_c = ds['v_control'].values
    u_diff = ds['u_difference'].values
    v_diff = ds['v_difference'].values
    lon_use = ds['lon'].values
    lat_use = ds['lat'].values

    fig, axes = plt.subplots(
        2, 1,
        figsize=FIGSIZE,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    plt.subplots_adjust(left=0.12, right=0.96, top=0.95,
                        bottom=0.04, hspace=0.25, wspace=0.08)

    lat_min, lat_max = lat_use.min(), lat_use.max()

    lmin, lmax = -90.0, 0.0
    lon_ticks_major = np.arange(-90.0, 1.0, 30.0)
    lon_ticks_minor = (lon_ticks_major[:-1] + lon_ticks_major[1:]) / 2.0

    yt0 = math.ceil(lat_min / 20.0) * 20
    yt1 = math.floor(lat_max / 20.0) * 20
    lat_ticks_major = np.arange(yt0, yt1 + 1, 20.0) if yt0 <= yt1 else np.linspace(lat_min, lat_max, max(2, N_TICKS))
    lat_ticks_minor = (lat_ticks_major[:-1] + lat_ticks_major[1:]) / 2.0 if lat_ticks_major.size >= 2 else np.array([])

    plot_data = [
        (u_c, v_c, arrow_color_ctrl, True, '(a) CTRL '),
        (u_diff, v_diff, arrow_color_diff, False, '(b) GGW-CTRL ')
    ]

    for ax, (u_plot, v_plot, arrow_color, use_fixed_key, title) in zip(axes, plot_data):

        for spine in ax.spines.values():
            spine.set_linewidth(AXES_FRAME_LINEWIDTH)

        ax.set_title(title,
                     fontsize=plt.rcParams['axes.titlesize'],
                     fontweight='bold',
                     loc='left')

        ax.set_extent([-90.0, 0.0,
                       0.0, 50.0],
                      crs=ccrs.PlateCarree())

        sx, sy, scale_val, _ = auto_downsample_quiver_params(
            lon_use, lat_use, u_plot, v_plot)

        lon_q, lat_q = np.meshgrid(lon_use[::sx], lat_use[::sy])
        u_q = u_plot[::sy, ::sx]
        v_q = v_plot[::sy, ::sx]

        this_scale = scale_val * (DIFF_SCALE_FACTOR if not use_fixed_key else 1.0)

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
            'labelpos': 'E',
            'fontproperties': {'size': QUIVERKEY_FONTSIZE},
            'facecolor': 'none',
            'edgecolor': arrow_color,
            'linewidth': 1
        }

        if use_fixed_key:
            ax.quiverkey(Q, 0.78, 1.04,
                         ROW_VKEY_VAL_ROW12,
                         f'{ROW_VKEY_VAL_ROW12:.1f} m s'+'$^{-1}$',
                         **key_kwargs)
        else:
            ax.quiverkey(Q, 0.78, 1.04,
                         1.0, '1.0 m s$^{-1}$',
                         **key_kwargs)

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.4)

        ax.set_xticks(lon_ticks_major, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks_major, crs=ccrs.PlateCarree())

        if lon_ticks_minor.size:
            ax.set_xticks(lon_ticks_minor, minor=True, crs=ccrs.PlateCarree())
        if lat_ticks_minor.size:
            ax.set_yticks(lat_ticks_minor, minor=True, crs=ccrs.PlateCarree())

        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lon_formatter_deg)
        )
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda y, pos:
                f"{int(round(y))}°N" if y >= 0
                else f"{abs(int(round(y)))}°S"
            )
        )

        ax.tick_params(axis='x', which='major',
                       direction='out',
                       length=TICK_MAJOR_LENGTH,
                       width=TICK_MAJOR_WIDTH,
                       top=False, bottom=True,
                       labelbottom=True)

        ax.tick_params(axis='x', which='minor',
                       direction='out',
                       length=TICK_MINOR_LENGTH,
                       width=TICK_MINOR_WIDTH)

        ax.tick_params(axis='y', which='major',
                       direction='out',
                       length=TICK_MAJOR_LENGTH,
                       width=TICK_MAJOR_WIDTH,
                       left=True, right=False,
                       labelleft=True)

        ax.tick_params(axis='y', which='minor',
                       direction='out',
                       length=TICK_MINOR_LENGTH,
                       width=TICK_MINOR_WIDTH)

    plt.savefig(_package_output / "figure9.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    ds.close()

if __name__ == "__main__":
    main()
