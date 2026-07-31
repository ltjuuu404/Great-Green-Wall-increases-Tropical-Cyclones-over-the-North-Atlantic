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

# --- Plotting Configuration ---
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22

# --- Constants ---
QUIVERKEY_FONTSIZE = 24
ROW_VKEY_VAL_ROW12 = 5.0
BOX_DEFS = [
    (10.0, 20.0, -70.0, -17.5, 'black', 'MDR'),
]
FIGSIZE = (10, 16)  # 两张图，适当缩小高度
N_TICKS = 6
SNAP_TO_ROUND = False
DIFF_SCALE_FACTOR = 1.8
AXES_FRAME_LINEWIDTH = 2.0
TICK_MAJOR_WIDTH = AXES_FRAME_LINEWIDTH
TICK_MINOR_WIDTH = 1.0
TICK_MAJOR_LENGTH = 8
TICK_MINOR_LENGTH = 4
quiver_max_arrows = 1000
min_stride = 3
arrow_color_ctrl = 'black'
arrow_color_diff = 'black'

# --- Helper Functions ---
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


# --- Main ---
def main():
    data_path = Path(__file__).resolve().parent / 'figure8_data.nc'
    ds = xr.open_dataset(data_path)

    u_c = ds['u_control'].values
    v_c = ds['v_control'].values
    u_diff = ds['u_difference'].values
    v_diff = ds['v_difference'].values
    lon_use = ds['lon'].values
    lat_use = ds['lat'].values

    # 只创建 2 行子图
    fig, axes = plt.subplots(
        2, 1,
        figsize=FIGSIZE,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    plt.subplots_adjust(left=0.08, right=0.96, top=0.95,
                        bottom=0.04, hspace=0.35, wspace=0.08)

    lat_min, lat_max = lat_use.min(), lat_use.max()

    # ---- Tick 设置（保持原样）----
    lmin, lmax = float(np.min(lon_use)), float(np.max(lon_use))
    start = int(math.ceil(lmin / 40.0) * 40)
    end = int(math.floor(lmax / 40.0) * 40)
    lon_ticks_major = np.arange(start, end + 1, 40.0) if start <= end else np.linspace(lmin, lmax, max(2, N_TICKS))
    lon_ticks_minor = (lon_ticks_major[:-1] + lon_ticks_major[1:]) / 2.0 if lon_ticks_major.size >= 2 else np.array([])

    yt0 = math.ceil(lat_min / 20.0) * 20
    yt1 = math.floor(lat_max / 20.0) * 20
    lat_ticks_major = np.arange(yt0, yt1 + 1, 20.0) if yt0 <= yt1 else np.linspace(lat_min, lat_max, max(2, N_TICKS))
    lat_ticks_minor = (lat_ticks_major[:-1] + lat_ticks_major[1:]) / 2.0 if lat_ticks_major.size >= 2 else np.array([])

    # 仅保留 CTRL 和 DIFF
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

        ax.set_extent([lon_use[0], lon_use[-1],
                       lat_use[0], lat_use[-1]],
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
            ax.quiverkey(Q, 0.78, 1.06,
                         ROW_VKEY_VAL_ROW12,
                         f'{ROW_VKEY_VAL_ROW12:.1f} m/s',
                         **key_kwargs)
        else:
            ax.quiverkey(Q, 0.78, 1.06,
                         1.0, '1.0 m/s',
                         **key_kwargs)

            import matplotlib.patches as mpatches
            for (lat1, lat2, lon1, lon2, color, name) in BOX_DEFS:
                rect = mpatches.Rectangle(
                    (lon1, lat1),
                    lon2 - lon1,
                    lat2 - lat1,
                    linewidth=2.5,
                    edgecolor=color,
                    facecolor='none',
                    transform=ccrs.PlateCarree(),
                    zorder=10
                )
                ax.add_patch(rect)

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

    plt.show()
    plt.close(fig)
    ds.close()


if __name__ == "__main__":
    main()
