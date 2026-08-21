
# -*- coding: utf-8 -*-
from pathlib import Path
import warnings
import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4 as nc

matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 600
import matplotlib.ticker as mticker
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from matplotlib.patches import Rectangle
import logging

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# try cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

USE_DASK = False


processed_dir = Path(__file__).resolve().parent

lat_min, lat_max = -10., 50.
lon_min, lon_max = -100., 60.
central_lon = -10.0
SELECT_ROWS = [1, 2, 3, 4] # Corresponds to the four nc files

figsize = (6, 100)  # Increase height for 4 rows
OTHER_FONT_SIZE = 16
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = OTHER_FONT_SIZE

ROW_TITLE_Y_OFFSET = 0.09 
ROW_VSPACE = -0.5

AXES_FRAME_LINEWIDTH = 2.0
TICK_MAJOR_WIDTH = AXES_FRAME_LINEWIDTH
TICK_MINOR_WIDTH = 1.0
TICK_MAJOR_LENGTH = 8
TICK_MINOR_LENGTH = 4

CBAR_LIMITS = {
    1: {'common': None, 'diff': None},
    2: {'common': (-0.06,0.06), 'diff': None},
    3: {'common': (0, 80), 'diff': None},
    4: {'common': None, 'diff': None},
}

CB_THICKNESS = 0.01
CB_PAD = 0.11

ALPHA = 0.05



from matplotlib.colors import LinearSegmentedColormap

cmap_zeta = LinearSegmentedColormap.from_list(
    'zeta_custom', 
    ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', 'white', 
     '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02'],
    N=256
)

cmap_omega = LinearSegmentedColormap.from_list(
    'omega_custom',
    ['#00441b', '#1b7837', '#5aae61', '#a6dba0', '#d9f0d3', 'white',
     '#e7d4e8', '#c2a5cf', '#9970ab', '#762a83', '#40004b'],
    N=256
)

ROW_CMAPS = {
    1: cmap_zeta, 
    2: cmap_omega,  
    3: 'BrBG',       
    4: 'BrBG',
}
DIFF_CMAP = 'RdBu'

# Wind Vector Scale Keys (None = default 5 m/s)
WIND_SCALE_ROW1_CTRL = 15
WIND_SCALE_ROW1_DIFF = 1
WIND_SCALE_ROW2_CTRL = 15
WIND_SCALE_ROW2_DIFF = 1
WIND_SCALE_ROW4_CTRL = 15
WIND_SCALE_ROW4_DIFF = 1

# ----------------------------------------------------------------------------------------------

# ---------------- helpers ----------------
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

def compute_levels(arrs, is_diff=False, nlev=11, manual_limits=None):
    if manual_limits is not None:
        vmin, vmax = manual_limits
        return np.linspace(vmin, vmax, nlev)
    vals = np.concatenate([np.ravel(a) for a in arrs if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.linspace(-1.0, 1.0, nlev) if is_diff else np.linspace(0, 1, nlev)
    if is_diff:
        lim = np.percentile(np.abs(vals), 98)
        lim = max(lim, 1e-12)
        return np.linspace(-lim, lim, nlev)
    vmin, vmax = np.percentile(vals, [2, 98])
    if vmax <= vmin:
        vmin, vmax = vals.min(), vals.max()
    return np.linspace(vmin, vmax, nlev)

def grid_cell_size(lons_arr, lats_arr):
    dx = np.mean(np.diff(lons_arr))
    dy = np.mean(np.diff(lats_arr))
    return dx, dy

def get_cbar_ticks(levels, n_ticks):
    levels = np.asarray(levels)
    if levels.size == 0:
        return np.array([])
    n_levels = len(levels)
    n_ticks = min(max(2, int(n_ticks)), n_levels)
    idx = np.linspace(0, n_levels - 1, n_ticks)
    idx = np.round(idx).astype(int)
    idx = np.unique(idx)
    ticks = levels[idx]
    return np.array(ticks)

def load_data_from_nc(filepath):
    """Loads all variables from a NetCDF file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    try:
        with nc.Dataset(filepath, 'r') as ds:
            data = {}
            for var in ds.variables:
                data[var] = ds.variables[var][:]
            return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# formatting helpers for colorbar labels (mantissa formatter and non-scientific)
def make_non_scientific_formatter(decimals):
    """Return FuncFormatter that formats with 'decimals' and replaces '-0.0' with '0.0'."""
    fmt_str = f"{{:.{decimals}f}}"
    def _fmt(x, pos=None):
        s = fmt_str.format(x)
        # remove negative zero
        if s.startswith("-0") and float(s) == 0.0:
            s = s.replace("-0", "0", 1)
        return s
    return FuncFormatter(_fmt)

def make_scientific_mantissa_formatter(levels, decimals):
    """
    Return (formatter, exponent, offset_text) where:
    - formatter formats value -> mantissa string with given decimals
    - exponent is integer power-of-ten used for scaling (10**exponent)
    - offset_text is mathtext string for display (eg '$\\times10^{-5}$')
    """
    lv = np.asarray(levels)
    max_abs = np.nanmax(np.abs(lv)) if lv.size else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(max_abs)))
    scale = 10.0 ** exponent if exponent != 0 else 1.0
    fmt_str = f"{{:.{decimals}f}}"
    def _fmt(x, pos=None):
        # avoid division by zero issues
        m = x / scale
        s = fmt_str.format(m)
        # remove negative zero in mantissa
        if s.startswith("-0") and float(s) == 0.0:
            s = s.replace("-0", "0", 1)
        return s
    offset_text = r"$\times10^{{{}}}$".format(exponent) if exponent != 0 else ""
    return FuncFormatter(_fmt), exponent, offset_text

# ---------------- main (plot only) ----------------
def main():
    print("=== Start main (plot-only) - two-column version ===")
    
    files_to_load = [
        processed_dir / "figure6_row1_850_circulation.nc",
        processed_dir / "figure6_row2_500_omega.nc",
        processed_dir / "figure6_row3_700_rh.nc",
        processed_dir / "figure6_row4_200_shear.nc",
    ]

    rows_data = [load_data_from_nc(f) for f in files_to_load]
    rows_data = [d for d in rows_data if d is not None]

    if not rows_data:
        raise FileNotFoundError(f"No data loaded. Check files in {processed_dir}.")

    sample = rows_data[0]
    lats = sample.get('lat', None)
    lons = sample.get('lon', None)
    if lats is None or lons is None:
        lats = np.linspace(lat_min, lat_max, 61)
        lons = np.linspace(lon_min, lon_max, 161)
        print("Using fallback lats/lons grid.")

    lons = normalize_lon_to_center(lons, central_lon)

    nrows = len(rows_data)
    ncols = 2  # <-- two columns now: Control | Diff

    if CARTOPY_AVAILABLE:
        proj = ccrs.PlateCarree(central_longitude=central_lon)
        fig = plt.figure(figsize=figsize)
        axes = []
        for ri in range(nrows):
            row_axes = []
            for ci in range(ncols):
                ax = fig.add_subplot(nrows, ncols, ri * ncols + ci + 1, projection=proj)
                row_axes.append(ax)
            axes.append(row_axes)
        axes = np.array(axes)
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape((1, -1))

    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.07, right=0.96, hspace=0.20, wspace=0.2)

    letters = [f'({chr(97 + i)})' for i in range(nrows * ncols)]
    letter_idx = 0

    # decimals mapping by plot-row index (first plotted row -> 1 dec, second -> 1 dec, third -> int)
    decimals_by_plot_row = {0: 1, 1: 1, 2: 0, 3: 1}

    for irow, rinfo in enumerate(rows_data):
        row_num = SELECT_ROWS[irow]
        print(f"Plotting row {irow+1} (orig {row_num})")
        
        # Extract data based on row number
        if row_num == 1: # zeta
            ctrl_arr = rinfo.get('zeta_ctrl')
            sens_arr = rinfo.get('zeta_sens')
            diff_arr = rinfo.get('zeta_diff')
            pval = rinfo.get('zeta_pval')
            u_ctrl = rinfo.get('u850_ctrl')
            v_ctrl = rinfo.get('v850_ctrl')
            u_sens = rinfo.get('u850_sens')
            v_sens = rinfo.get('v850_sens')
            units = 's$^{-1}$'
        elif row_num == 2: # omega
            ctrl_arr = rinfo.get('omega_ctrl')
            sens_arr = rinfo.get('omega_sens')
            diff_arr = rinfo.get('omega_diff')
            pval = rinfo.get('omega_pval')
            u_ctrl = rinfo.get('u500_ctrl')
            v_ctrl = rinfo.get('v500_ctrl')
            u_sens = rinfo.get('u500_sens')
            v_sens = rinfo.get('v500_sens')
            units = 'Pa s$^{-1}$'
        elif row_num == 3: # rh
            ctrl_arr = rinfo.get('rh_ctrl')
            sens_arr = rinfo.get('rh_sens')
            diff_arr = rinfo.get('rh_diff')
            pval = rinfo.get('rh_pval')
            u_ctrl, v_ctrl, u_sens, v_sens = None, None, None, None
            units = '%'
        elif row_num == 4: # shear
            ctrl_arr = rinfo.get('shear_ctrl')
            sens_arr = rinfo.get('shear_sens')
            diff_arr = rinfo.get('shear_diff')
            pval = rinfo.get('shear_pval')
            u_ctrl = rinfo.get('u200_ctrl')
            v_ctrl = rinfo.get('v200_ctrl')
            u_sens = rinfo.get('u200_sens')
            v_sens = rinfo.get('v200_sens')
            units = 'm s$^{-1}$'
        else:
            continue

        dx, dy = grid_cell_size(lons, lats)

        row_limits = CBAR_LIMITS.get(row_num, {'common': None, 'diff': None})
        common_manual = row_limits.get('common', None)
        diff_manual = row_limits.get('diff', None)

        levels_common = compute_levels([ctrl_arr, sens_arr], is_diff=False, manual_limits=common_manual)
        levels_diff = compute_levels([diff_arr], is_diff=True, manual_limits=diff_manual)

        # --- determine the common colormap for this row (ctrl & sens) ---
        cmap_common_spec = ROW_CMAPS.get(row_num, 'viridis')
        try:
            cmap_common = plt.cm.get_cmap(cmap_common_spec) if isinstance(cmap_common_spec, str) else cmap_common_spec
        except Exception:
            cmap_common = plt.cm.viridis

        # diff cmap - 对row 2和4反转
        try:
            if row_num in [2, 4]: 
                cmap_diff = plt.cm.get_cmap(DIFF_CMAP + '_r') if isinstance(DIFF_CMAP, str) else DIFF_CMAP.reversed()
            else:
                cmap_diff = plt.cm.get_cmap(DIFF_CMAP) if isinstance(DIFF_CMAP, str) else DIFF_CMAP
        except Exception:
            cmap_diff = plt.cm.RdBu_r

        # keys for two columns only
        for icol, key in enumerate(['ctrl', 'diff']):
            ax = axes[irow, icol] if nrows > 1 else axes[icol]

            # set bold border
            try:
                if CARTOPY_AVAILABLE:
                    outline = getattr(ax, 'outline_patch', None) or getattr(ax, 'patch', None)
                    if outline is not None:
                        outline.set_linewidth(AXES_FRAME_LINEWIDTH)
                        outline.set_edgecolor('black')
                else:
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(AXES_FRAME_LINEWIDTH)
                        spine.set_edgecolor('black')
            except Exception:
                pass

            is_bottom = (irow == nrows - 1)
            is_left = (icol == 0)
            try:
                if CARTOPY_AVAILABLE:
                    xticks_raw = np.arange(-180 + central_lon, 180 + central_lon + 1, 30.0)
                    xticks_plot = normalize_lon_to_center(xticks_raw, central_lon)
                    xticks_sorted = np.sort(xticks_plot)
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

                    try:
                        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
                    except Exception:
                        pass

                    gl = ax.gridlines(draw_labels=False, linewidth=0.35, color='gray', linestyle='--', zorder=2)
                else:
                    ax.set_xticks(np.linspace(lon_min, lon_max, 6))
                    ax.set_yticks(np.linspace(lat_min, lat_max, 6))
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_lon_label(x)))
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: format_lat_label(y)))
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

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

                    ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.6)

            except Exception as e:
                print("Tick/label setup warning:", e)

            col_label = ['CTL', 'GGW-CTL'][icol]
            panel_letter = letters[letter_idx]
            
            row_titles = {
                1: "ζ & 850 hPa wind",
                2: "500 hPa Omega & Wind",
                3: "700 hPa RH",
                4: "VWS & 200 hPa Wind"
            }
            
            title_text = f"{panel_letter} {col_label} {row_titles.get(row_num, '')}"
            ax.text(0.01, 1.02, title_text, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=OTHER_FONT_SIZE-2, fontweight='bold', color='black')
            letter_idx += 1

            if key == 'ctrl':
                data_arr = ctrl_arr
                uvec, vvec = u_ctrl, v_ctrl
            else:  # 'diff'
                data_arr = diff_arr
                if (u_sens is not None) and (u_ctrl is not None):
                    uvec = (u_sens - u_ctrl)
                    vvec = (v_sens - v_ctrl)
                else:
                    uvec = vvec = None

            if data_arr is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
                continue

            levels = levels_diff if key == 'diff' else levels_common

            if key == 'diff':
                cmap = cmap_diff
            else:
                cmap = cmap_common

            try:
                cf = ax.contourf(lons, lats, data_arr, levels=levels, extend='both',
                                 transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None, cmap=cmap, zorder=1)
                if CARTOPY_AVAILABLE:
                    ax.coastlines(resolution='50m', linewidth=0.6)
            except Exception:
                cf = ax.contourf(lons, lats, data_arr, transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None, cmap=cmap, zorder=1)

            try:
                ax.contour(lons, lats, data_arr, levels=levels, linewidths=0.4, colors='gray',
                           transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None, zorder=1.5)
            except Exception:
                pass

            if (uvec is not None) and (vvec is not None):
                try:
                    u = uvec
                    v = vvec
                    ny, nx = u.shape
                    total = max(1, nx * ny)
                    factor = int(np.ceil(np.sqrt(total / max(1, 200))))
                    sx = max(3, factor)
                    sy = max(3, factor)
                    lon2, lat2 = np.meshgrid(lons[::sx], lats[::sy])
                    u_sub = u[::sy, ::sx]
                    v_sub = v[::sy, ::sx]
                    scale_val = max(1.0, np.nanpercentile(np.sqrt(u_sub**2 + v_sub**2), 95) * 12.0)
                    transform = ccrs.PlateCarree() if CARTOPY_AVAILABLE else None
                    Q = ax.quiver(lon2, lat2, u_sub, v_sub, transform=transform, width=0.006, pivot='middle', scale=scale_val, zorder=4)
                    
                    if irow == 0 or irow == 1 or irow == 3: # Rows 1, 2, 4
                        key_val = 5
                        if irow == 0:
                            key_val = WIND_SCALE_ROW1_CTRL if key == 'ctrl' else WIND_SCALE_ROW1_DIFF
                        elif irow == 1:
                            key_val = WIND_SCALE_ROW2_CTRL if key == 'ctrl' else WIND_SCALE_ROW2_DIFF
                        elif irow == 3:
                            key_val = WIND_SCALE_ROW4_CTRL if key == 'ctrl' else WIND_SCALE_ROW4_DIFF
                        
                        ax.quiverkey(Q, 0.9, 1.05, key_val, '{} m s$^{{-1}}$'.format(key_val),
             labelpos='N', labelsep=0.01, coordinates='axes',
             fontproperties={'size': OTHER_FONT_SIZE-4, 'family': 'Times New Roman'})

                        # ax.quiverkey(Q, 0.85, 1.05, key_val, '{}m s$^{{-1}}$'.format(key_val), labelpos='N',
                        #              coordinates='axes', fontproperties={'size': OTHER_FONT_SIZE-4, 'family': 'Times New Roman'})

                except Exception as e:
                    print("quiver issue:", e)

            if key == 'diff' and (pval is not None):
                try:
                    p_arr = np.array(pval)
                    data_np = np.array(data_arr)
                    if p_arr.shape != data_np.shape:
                        if p_arr.T.shape == data_np.shape:
                            p_arr = p_arr.T
                        else:
                            p_arr = None
                    if p_arr is not None:
                        data_mask = np.isnan(data_np)
                        sig_mask = (p_arr < ALPHA) & np.isfinite(p_arr) & (~data_mask)
                        if np.any(sig_mask):
                            ny = len(lats)
                            nx = len(lons)
                            for ii in range(ny):
                                for jj in range(nx):
                                    if sig_mask[ii, jj]:
                                        x0 = lons[jj] - dx / 2.0
                                        y0 = lats[ii] - dy / 2.0
                                        rect = Rectangle((x0, y0), dx*1.02, dy*1.02,
                                                         facecolor='none', edgecolor='white', linewidth=0.0,
                                                         hatch='///',
                                                         transform=ccrs.PlateCarree()._as_mpl_transform(ax) if CARTOPY_AVAILABLE else ax.transData,
                                                         zorder=5)
                                        ax.add_patch(rect)
                except Exception as e:
                    print("hatch issue:", e)

            try:
                box_x, box_y, box_w, box_h = -70, 10, 52.5, 10
                rect_box = Rectangle(
                    (box_x, box_y), box_w, box_h,
                    linewidth=2.0, edgecolor='black', facecolor='none',
                    transform=ccrs.PlateCarree() if CARTOPY_AVAILABLE else None,
                    zorder=10
                )
                ax.add_patch(rect_box)
            except Exception as e:
                print("box issue:", e)

            ax_pos = ax.get_position()
            cb_w = ax_pos.width * 0.72
            cb_x = ax_pos.x0 + (ax_pos.width - cb_w) / 2.0
            cb_y = ax_pos.y0 - CB_PAD
            cax = fig.add_axes([cb_x, cb_y, cb_w, CB_THICKNESS])

            ticks = get_cbar_ticks(levels, 5) if key == 'diff' else np.asarray(levels[::2])

            cb = fig.colorbar(cf, cax=cax, orientation='horizontal', boundaries=levels, ticks=ticks)

            lv = np.asarray(levels)
            use_scientific = np.all(np.abs(lv) < 1.0) and np.any(np.abs(lv) > 0)

            decimals = decimals_by_plot_row.get(irow, 1)

            unit_text = units
            if use_scientific:
                fmt, exponent, offset_text = make_scientific_mantissa_formatter(levels, decimals)
                cb.ax.xaxis.set_major_formatter(fmt)
                cb.formatter = fmt
                cb.update_ticks()
                if offset_text:
                    unit_text = f"{offset_text} {units}"
            else:
                fmt = make_non_scientific_formatter(decimals)
                cb.ax.xaxis.set_major_formatter(fmt)
                cb.formatter = fmt
                cb.update_ticks()

            cb.ax.text(1.02, -0.5, f"({unit_text})", transform=cb.ax.transAxes,
                       ha='left', va='center', fontsize=OTHER_FONT_SIZE - 6)

            cb.ax.tick_params(labelsize=OTHER_FONT_SIZE - 6)


    print("=== Displaying figure (use Ctrl+S in figure window to save) ===")
    plt.show()
    print("=== Done ===")

if __name__ == "__main__":
    main()
