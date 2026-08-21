import os
from pathlib import Path
import numpy as np
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("cartopy not available; plotting without coastlines/land mask.")

try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_AXGRID = True
except Exception:
    HAS_AXGRID = False

IN_NC = Path(__file__).resolve().parent / "figure10_data.nc"

LON_MIN, LON_MAX = -90.0, 0.0
LAT_MIN, LAT_MAX = 0.0, 50.0

CB_N_BINS = 9
CMAP_DIFF_NAME = 'RdBu_r'
DIFF_CB = (-1.8, 1.8)
DIFF_TICK_DECIMALS = 1

BOX_DEFS_DRAW = [
    (10.0, 30.0, -89.5, -60, 'black', 'Box1'),
    (10.0, 30.0, -59.5, -17.5, 'purple', 'Box2'),
    (9.5, 30.5, -90.0, -17.0, 'red', 'Box1+2')
]

FONT_FAMILY = "Times New Roman"
FS_TITLE = 24
FS_TICKLABEL = 18
FS_COLORBAR = 16
FS_ANNOT = 18

plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.serif'] = [FONT_FAMILY]

def compute_major_ticks(minv, maxv, step=30.0):
    """
    生成主刻度，但不会超过 maxv。
    例如 minv=0, maxv=50, step=20 -> [0, 20, 40, 50]
    """
    ticks = np.arange(minv, maxv + 1e-6, step)
    if ticks.size == 0:
        return np.array([minv, maxv])
    if not np.isclose(ticks[-1], maxv):
        ticks = np.append(ticks, maxv)
    return ticks

def midpoints(arr):
    if arr.size < 2:
        return np.array([])
    return (arr[:-1] + arr[1:]) / 2.0

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

def make_axes_bold(ax, lw=1.8):
    try:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw)
            spine.set_edgecolor('black')
    except Exception:
        try:
            outline = getattr(ax, 'outline_patch', None)
            if outline is not None:
                outline.set_linewidth(lw)
                outline.set_edgecolor('black')
        except Exception:
            pass

def plot_tc_diff(lon_centers, lat_centers, tc_diff):
    LonC, LatC = np.meshgrid(lon_centers, lat_centers)

    vmin, vmax = DIFF_CB
    boundaries = np.linspace(vmin, vmax, CB_N_BINS + 1)
    cmap = ListedColormap(plt.get_cmap(CMAP_DIFF_NAME)(np.linspace(0, 1, CB_N_BINS)))
    norm = BoundaryNorm(boundaries, ncolors=CB_N_BINS, clip=False)

    fig = plt.figure(figsize=(10, 6))
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        except Exception:
            try:
                ax.coastlines(resolution='50m', linewidth=1.5)
            except Exception:
                pass

        # 强制限制显示范围，避免自动扩展到 60N
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

    cf = ax.contourf(
        LonC, LatC, tc_diff,
        levels=boundaries, cmap=cmap, norm=norm, extend='both',
        transform=ccrs.PlateCarree() if HAS_CARTOPY else None
    )

    import matplotlib.patches as mpatches

    for lat1, lat2, lon1, lon2, color, name in BOX_DEFS_DRAW:
        width = lon2 - lon1
        height = lat2 - lat1
        xy = (lon1, lat1)
        rect = mpatches.Rectangle(
            xy, width, height,
            linewidth=2, edgecolor=color, facecolor='none',
            transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
            zorder=10
        )
        ax.add_patch(rect)

        if name == 'Box1':
            tx = lon2 - 8.0
            ty = lat2 - 1.0
            ax.text(
                tx, ty, name,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
                fontsize=16, color=color, fontweight='bold',
                ha='left', va='top'
            )
        elif name == 'Box2':
            tx = lon1 + 1.0
            ty = lat2 - 1.0
            ax.text(
                tx, ty, name,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
                fontsize=16, color=color, fontweight='bold',
                ha='left', va='top'
            )
        elif name == 'Box1+2':
            tx = lon2 - 10.5
            ty = lat2 + 0.7
            ax.text(
                tx, ty, name,
                transform=ccrs.PlateCarree() if HAS_CARTOPY else ax.transData,
                fontsize=16, color=color, fontweight='bold',
                ha='left', va='bottom'
            )

    ax.text(
        0.01, 1.02, 'TC track(GGW-CTL)',
        transform=ax.transAxes, va='bottom', ha='left',
        fontsize=FS_TITLE, fontweight='bold', fontname=FONT_FAMILY
    )

    lon_major = compute_major_ticks(LON_MIN, LON_MAX, step=30.0)
    lat_major = compute_major_ticks(LAT_MIN, LAT_MAX - 10.0, step=20.0)

    lon_minor = midpoints(lon_major)
    lat_minor = np.arange(LAT_MIN, LAT_MAX - 9.9, 10.0)

    try:
        ax.set_xticks(lon_major, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_xticks(lon_major)

    try:
        ax.set_yticks(lat_major, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_yticks(lat_major)

    if lon_minor.size > 0:
        try:
            ax.set_xticks(lon_minor, minor=True, crs=ccrs.PlateCarree())
        except Exception:
            ax.set_xticks(lon_minor, minor=True)

    if lat_minor.size > 0:
        try:
            ax.set_yticks(lat_minor, minor=True, crs=ccrs.PlateCarree())
        except Exception:
            ax.set_yticks(lat_minor, minor=True)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon_deg))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat_deg))

    ax.tick_params(which='major', width=1.8, length=6, labelsize=FS_TICKLABEL, color='black')
    ax.tick_params(which='minor', width=0.9, length=3, labelsize=FS_TICKLABEL)

    # 再次强制锁定范围，防止 cartopy 自动拉伸
    if HAS_CARTOPY:
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)

    make_axes_bold(ax, lw=1.8)

    if HAS_AXGRID:
        try:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='5%', pad=0.4, axes_class=plt.Axes)
            cb = plt.colorbar(cf, cax=cax, extend='both', orientation='horizontal')
            cb.ax.tick_params(labelsize=FS_COLORBAR)
        except Exception:
            pos = ax.get_position()
            x0 = pos.x0
            y0 = pos.y0 - 0.08
            w = pos.width
            h = 0.02
            cax = fig.add_axes([x0, y0, w, h])
            cb = plt.colorbar(cf, cax=cax, extend='both', orientation='horizontal')
            cb.ax.tick_params(labelsize=FS_COLORBAR)
    else:
        pos = ax.get_position()
        x0 = pos.x0
        y0 = pos.y0 - 0.08
        w = pos.width
        h = 0.02
        cax = fig.add_axes([x0, y0, w, h])
        cb = plt.colorbar(cf, cax=cax, extend='both', orientation='horizontal')
        cb.ax.tick_params(labelsize=FS_COLORBAR)

    try:
        ticks_arr = np.array(cb.get_ticks(), dtype=float)
        if DIFF_TICK_DECIMALS > 0:
            fmt_str = '%%.%df' % int(DIFF_TICK_DECIMALS)
            labels = [fmt_str % t for t in ticks_arr]
        else:
            labels = [str(int(round(t))) for t in ticks_arr]
        cb.set_ticks(ticks_arr)
        cb.ax.set_xticklabels(labels, fontsize=FS_COLORBAR)
    except Exception:
        pass

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.12)
    plt.show()

def main():
    if not os.path.exists(IN_NC):
        print(f"Error: {IN_NC} not found.")
        return

    with nc.Dataset(IN_NC, 'r') as ds:
        lat = ds.variables['lat'][:]
        lon = ds.variables['lon'][:]
        tc_diff = ds.variables['tc_diff'][:]

    plot_tc_diff(lon, lat, tc_diff)

if __name__ == "__main__":
    main()
