#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2 plotting script with MDR boxes on panels (b), (d), and (f)."""
from pathlib import Path
import warnings
import traceback
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from shapely.geometry import Point, shape
from shapely.ops import unary_union
import re
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import math

# DPI settings
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 600
warnings.filterwarnings("ignore")

# Input directory
INPUT_DIR = Path(__file__).resolve().parent

LAT_SLICE = (-10, 50)
LON_SLICE = (-90, 50)
central_lon = -10.0
FIGSIZE = (15, 18)
CMAPS = {
    'sst': 'RdBu_r',
    'precip': 'Blues',
    'precip_diff': 'BrBG',
    'slp': 'viridis',
    'slp_diff': 'RdBu_r'
}

ALPHA = 0.05
USE_LAND_MASK = False

FONT_NAME = "Times New Roman"
FONT_SIZE = 18
TITLE_FONT_SIZE = FONT_SIZE 
mpl.rcParams.update({
    'font.family': 'Times New Roman',
    'font.serif': ['Times New Roman', 'Times'],
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 18,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})
plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['font.size'] = FONT_SIZE

axes_frame_linewidth = 1.8
LAND_OVERLAY_COLOR = '#cccccc'
CONTOUR_ZORDER = 1
SIG_ZORDER = 10
LAND_OVERLAY_ZORDER = 20
COAST_ZORDER = 50
SPINE_ZORDER = 100

# Main development region (MDR): 10-20 degrees N, 70-17.5 degrees W.
MDR_LON_MIN = -70.0
MDR_LON_MAX = -17.5
MDR_LAT_MIN = 10.0
MDR_LAT_MAX = 20.0
MDR_BOX_LINEWIDTH = 2.0

# ------------------ Global colorbar limit controls ------------------
SST_SENS_VMIN = 285
SST_SENS_VMAX = 305
SST_DIFF_VMIN = -1
SST_DIFF_VMAX = 1

PRECIP_SENS_VMIN = 0
PRECIP_SENS_VMAX = 10
PRECIP_DIFF_VMIN = -1
PRECIP_DIFF_VMAX = 1

SLP_SENS_VMIN = 100300
SLP_SENS_VMAX = 102500
SLP_DIFF_VMIN = -100
SLP_DIFF_VMAX = 100

PANEL_TITLES = {
    'rows': [None, None, None],
    'cols': [None, None],
    'cells': ["SST (CTL)", "SST (GGW-CTL)",
              "SLP (CTL)", "SLP (GGW-CTL)",
              "Precip (CTL)", "Precip (GGW-CTL)"]
}

WIND_SCALE_SENS_KEY = 10
WIND_SCALE_DIFF_KEY = 1
WIND_VECTOR_SCALE_MULTIPLIER = 16.0

# ------------------ Helper Functions ------------------
def get_land_mask_and_geoms(lon_vals, lat_vals, resolution='50m'):
    try:
        shp_path = shapereader.natural_earth(resolution=resolution, category='physical', name='land')
        reader = shapereader.Reader(shp_path)
        geoms = list(reader.geometries())
        land_geom = unary_union([shape(g) for g in geoms])
    except Exception as e:
        warnings.warn(f"Could not build land geometry ({e}). Land overlay disabled.")
        return None, None
    lon_vals = np.asarray(lon_vals)
    lat_vals = np.asarray(lat_vals)
    if lon_vals.ndim != 1 or lat_vals.ndim != 1:
        try:
            lon_vals = np.unique(lon_vals)
            lat_vals = np.unique(lat_vals)
        except Exception:
            warnings.warn("lon_vals/lat_vals not 1D and cannot be uniqued; aborting land mask.")
            return None, geoms
    nlat = lat_vals.size
    nlon = lon_vals.size
    mask = np.zeros((nlat, nlon), dtype=bool)
    for i in range(nlat):
        lat_i = float(lat_vals[i])
        for j in range(nlon):
            lon_j = float(lon_vals[j])
            try:
                if land_geom.contains(Point(lon_j, lat_i)):
                    mask[i, j] = True
            except Exception:
                mask[i, j] = False
    return mask, geoms

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

def grid_cell_size(lons_arr, lats_arr):
    dx = np.mean(np.diff(lons_arr))
    dy = np.mean(np.diff(lats_arr))
    return dx, dy

def make_axes_bold(ax, lw=axes_frame_linewidth):
    try:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(lw)
            spine.set_edgecolor('black')
            try:
                spine.set_zorder(SPINE_ZORDER)
            except Exception:
                pass
    except Exception:
        try:
            outline = getattr(ax, 'outline_patch', None)
            if outline is not None:
                outline.set_linewidth(lw)
                outline.set_edgecolor('black')
                try:
                    outline.set_zorder(SPINE_ZORDER)
                except Exception:
                    pass
        except Exception:
            pass

def align_mask_to_shape(mask, target_shape):
    if mask is None:
        return None
    try:
        if mask.shape == target_shape:
            return mask
        if mask.T.shape == target_shape:
            return mask.T
        if mask.size == np.prod(target_shape):
            return mask.reshape(target_shape)
    except Exception:
        pass
    return None

def format_cb_value_custom(val, row_key, col_key):
    if abs(val) < 1e-12:
        return "0"
    if row_key == 'sst' and col_key == 'ctrl':
        return str(int(round(val)))
    elif row_key == 'sst' and col_key == 'diff':
        return f"{val:.1f}"
    elif row_key == 'precip' and col_key == 'ctrl':
        return f"{val:.1f}"
    elif row_key == 'precip' and col_key == 'diff':
        return f"{val:.1f}"
    elif row_key == 'slp' and col_key == 'ctrl':
        val_hpa = val / 100.0
        return str(int(round(val_hpa)))
    elif row_key == 'slp' and col_key == 'diff':
        return str(int(round(val)))
    else:
        return str(int(round(val)))

def apply_cb_tick_format_custom(cb, row_key, col_key):
    try:
        ticks = cb.get_ticks()
        labels = [format_cb_value_custom(t, row_key, col_key) for t in ticks]
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels)
    except Exception:
        pass

# ---------------- Plotting Function ----------------
def plot_3x2(sst_res, precip_res, slp_res, wind_res=None, land_mask=None, land_geoms=None, outfile='figure4_reproduced.png'):
    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.14, hspace=0.35, wspace=0.15)

    row_keys = ['sst', 'slp', 'precip']

    rows = [
        (row_keys[0], 'SST', sst_res, CMAPS['sst'], CMAPS['sst']),
        (row_keys[1], 'SLP', slp_res, CMAPS['slp'], CMAPS['slp_diff']),
        (row_keys[2], 'Precip (mm/day)', precip_res, CMAPS['precip'], CMAPS['precip_diff'])
    ]

    letters = [f'({chr(97+i)})' for i in range(6)]
    letter_idx = 0

    for irow, (row_key, row_default_title, res, cmap_main, cmap_diff) in enumerate(rows):
        if res is None:
            for icol in range(2):
                ax = axes[irow, icol]
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            continue

        lats = np.asarray(res.get('lats'))
        lons = np.asarray(res.get('lons'))
        lons = normalize_lon_to_center(lons, central_lon)
        
        dx, dy = grid_cell_size(lons, lats)

        sens = res.get('sens')
        diff = res.get('diff')
        pval = res.get('pval')
        units = res.get('units', '')

        row_title = PANEL_TITLES.get('rows', [None, None, None])[irow] if PANEL_TITLES.get('rows') else None
        if row_title is None:
            row_title = row_default_title
        col_titles_override = PANEL_TITLES.get('cols', [None, None]) if PANEL_TITLES.get('cols') else [None, None]
        cells_override = PANEL_TITLES.get('cells', [None]*6)

        if row_key == 'sst':
            vmin_sens, vmax_sens = SST_SENS_VMIN, SST_SENS_VMAX
            vmin_diff, vmax_diff = SST_DIFF_VMIN, SST_DIFF_VMAX
        elif row_key == 'precip':
            vmin_sens, vmax_sens = PRECIP_SENS_VMIN, PRECIP_SENS_VMAX
            vmin_diff, vmax_diff = PRECIP_DIFF_VMIN, PRECIP_DIFF_VMAX
        elif row_key == 'slp':
            vmin_sens, vmax_sens = SLP_SENS_VMIN, SLP_SENS_VMAX
            vmin_diff, vmax_diff = SLP_DIFF_VMIN, SLP_DIFF_VMAX
        else:
            vmin_sens = vmax_sens = vmin_diff = vmax_diff = None

        if vmin_sens is None or vmax_sens is None:
            try:
                auto_vmin = np.nanpercentile(sens, 2)
                auto_vmax = np.nanpercentile(sens, 98)
                if not np.isfinite(auto_vmin) or not np.isfinite(auto_vmax) or auto_vmax == auto_vmin:
                    auto_vmin = np.nanmin(sens)
                    auto_vmax = np.nanmax(sens)
                if vmin_sens is None: vmin_sens = auto_vmin
                if vmax_sens is None: vmax_sens = auto_vmax
            except Exception:
                vmin_sens = vmax_sens = None

        if vmin_diff is None or vmax_diff is None:
            try:
                vlim = np.nanmax(np.abs(diff))
                if not np.isfinite(vlim) or vlim == 0:
                    vlim = np.nanpercentile(np.abs(diff), 98)
                if vmin_diff is None: vmin_diff = -vlim
                if vmax_diff is None: vmax_diff = vlim
            except Exception:
                vmin_diff = vmax_diff = None

        try:
            levels_sens = np.linspace(vmin_sens, vmax_sens, 21) if (vmin_sens is not None and vmax_sens is not None) else 21
        except Exception:
            levels_sens = 21
        try:
            levels_diff = np.linspace(vmin_diff, vmax_diff, 21) if (vmin_diff is not None and vmax_diff is not None) else 21
        except Exception:
            levels_diff = 21

        c_sens = c_diff = None

        for icol, key in enumerate(['ctrl', 'diff']):
            ax = axes[irow, icol]
            ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=ccrs.PlateCarree())
            ax.coastlines(resolution='110m', linewidth=0.5, zorder=CONTOUR_ZORDER + 0.1)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, zorder=CONTOUR_ZORDER + 0.1)

            cell_idx = irow * 2 + icol
            cell_override = cells_override[cell_idx] if (cells_override and len(cells_override) > cell_idx) else None
            col_default = {0: 'Control', 1: 'Diff'}[icol]
            col_title_override = col_titles_override[icol] if (col_titles_override and len(col_titles_override) > icol) else None

            if cell_override is not None:
                display_col_title = cell_override
            else:
                display_col_title = col_title_override if col_title_override is not None else col_default

            ax.set_title(f"{letters[letter_idx]} {display_col_title}", loc='left', fontsize=TITLE_FONT_SIZE, fontweight='bold')
            letter_idx += 1

            data = res.get('ctrl') if key == 'ctrl' else diff
            if key == 'ctrl':
                levels = levels_sens
                cmap = cmap_main
            else:
                levels = levels_diff
                cmap = cmap_diff

            if data is None:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
                make_axes_bold(ax)
                continue

            try:
                cf = ax.contourf(lons, lats, data, levels=levels, cmap=cmap, extend='both',
                                 transform=ccrs.PlateCarree(), zorder=CONTOUR_ZORDER)
            except Exception:
                cf = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                                   shading='auto', cmap=cmap, zorder=CONTOUR_ZORDER)

            sig_plotted = False
            if key == 'diff' and pval is not None and pval.shape == data.shape:
                sig_mask = (pval < ALPHA)
                if land_mask is not None:
                    aligned = align_mask_to_shape(land_mask, sig_mask.shape)
                    if aligned is not None:
                        sig_mask = sig_mask & (~aligned)
                if np.any(sig_mask):
                    intmask = sig_mask.astype(int)
                    try:
                        cf_sig = ax.contourf(lons, lats, intmask, levels=[0.5, 1.5],
                                             colors='none', hatches=['///'],
                                             transform=ccrs.PlateCarree(), zorder=SIG_ZORDER)
                        for coll in cf_sig.collections:
                            coll.set_edgecolor('white')
                            coll.set_linewidth(0.2)
                        sig_plotted = True
                    except Exception:
                        ys, xs = np.where(sig_mask)
                        ax.scatter(lons[xs], lats[ys], s=1, transform=ccrs.PlateCarree(),
                                   color='white', alpha=0.7, zorder=SIG_ZORDER)
                        sig_plotted = True

            if row_key == 'slp' and wind_res is not None:
                u_vec = None
                v_vec = None
                if key == 'ctrl':
                    u_vec = wind_res.get('u_ctrl')
                    v_vec = wind_res.get('v_ctrl')
                elif key == 'diff':
                    u_vec = wind_res.get('u_diff')
                    v_vec = wind_res.get('v_diff')
                
                if u_vec is not None and v_vec is not None:
                    try:
                        u = u_vec
                        v = v_vec
                        ny, nx = u.shape
                        total = max(1, nx * ny)
                        factor = int(np.ceil(np.sqrt(total / max(1, 200))))
                        sx = max(3, factor)
                        sy = max(3, factor)
                        
                        lon2, lat2 = np.meshgrid(lons[::sx], lats[::sy])
                        u_sub = u[::sy, ::sx]
                        v_sub = v[::sy, ::sx]
                        
                        mag = np.sqrt(u_sub**2 + v_sub**2)
                        mag_95 = np.nanpercentile(mag, 95)
                        
                        if mag_95 < 1e-6:
                            scale_val = 1.0
                        else:
                            scale_val = mag_95 * WIND_VECTOR_SCALE_MULTIPLIER
                        
                        transform = ccrs.PlateCarree()
                        
                        Q = ax.quiver(lon2, lat2, u_sub, v_sub, transform=transform,
                                  width=0.005, pivot='middle', scale=scale_val, zorder=102, minlength=0)
                        
                        if key == 'ctrl':
                            key_val = WIND_SCALE_SENS_KEY if WIND_SCALE_SENS_KEY is not None else 5
                        else:
                            key_val = WIND_SCALE_DIFF_KEY if WIND_SCALE_DIFF_KEY is not None else 5
                            
                        ax.quiverkey(Q, 0.80, 1.05, key_val, r'{} m s$^{{-1}}$'.format(key_val), labelpos='E',
                                     coordinates='axes', fontproperties={'size': FONT_SIZE-4, 'family': 'Times New Roman'})
                    except Exception as e:
                        print(f"Error plotting wind vectors: {e}")

            if land_geoms is not None and USE_LAND_MASK and row_key == 'sst':
                try:
                    ax.add_geometries(land_geoms, crs=ccrs.PlateCarree(),
                                      facecolor=LAND_OVERLAY_COLOR, edgecolor='none',
                                      zorder=LAND_OVERLAY_ZORDER)
                except Exception:
                    try:
                        land_feature = cfeature.NaturalEarthFeature('physical', 'land', '50m')
                        ax.add_feature(land_feature, facecolor=LAND_OVERLAY_COLOR, edgecolor='none',
                                       zorder=LAND_OVERLAY_ZORDER)
                    except Exception:
                        pass

            try:
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=1.2, zorder=COAST_ZORDER)
            except Exception:
                try:
                    ax.coastlines(resolution='50m', linewidth=1.2)
                except Exception:
                    pass

            # Panels (b), (d), and (f) occupy the right column.
            if icol == 1:
                ax.add_patch(
                    Rectangle(
                        (MDR_LON_MIN, MDR_LAT_MIN),
                        MDR_LON_MAX - MDR_LON_MIN,
                        MDR_LAT_MAX - MDR_LAT_MIN,
                        facecolor='none',
                        edgecolor='black',
                        linewidth=MDR_BOX_LINEWIDTH,
                        transform=ccrs.PlateCarree(),
                        zorder=COAST_ZORDER + 5,
                    )
                )

            make_axes_bold(ax)

            try:
                xticks_sorted = np.arange(LON_SLICE[0], LON_SLICE[1] + 0.1, 30.0)
                yticks = np.arange(math.ceil(LAT_SLICE[0]/10)*10, math.floor(LAT_SLICE[1]/10)*10 + 1, 20.0)

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
                               length=8, width=axes_frame_linewidth,
                               top=False, bottom=True, labelsize=FONT_SIZE - 4,
                               labelbottom=True)
                ax.tick_params(axis='x', which='minor', direction='out',
                               length=4, width=1.0,
                               top=False, bottom=True, labelbottom=True)
                ax.tick_params(axis='y', which='major', direction='out',
                               length=8, width=axes_frame_linewidth,
                               left=True, right=False, labelsize=FONT_SIZE - 4,
                               labelleft=True)
                ax.tick_params(axis='y', which='minor', direction='out',
                               length=4, width=1.0,
                               left=True, right=False, labelleft=True)

                try:
                    ax.set_extent([LON_SLICE[0], LON_SLICE[1], LAT_SLICE[0], LAT_SLICE[1]], crs=ccrs.PlateCarree())
                except Exception:
                    pass

                gl = ax.gridlines(draw_labels=False, linewidth=0.35, color='gray', linestyle='--', zorder=2)

            except Exception as e:
                print("Tick/label setup warning:", e)

            if key == 'ctrl':
                c_sens = cf
            else:
                c_diff = cf

        try:
            pos_sens = axes[irow, 0].get_position()
            cb_w = pos_sens.width * 0.9
            cb_x = pos_sens.x0 + (pos_sens.width - cb_w) / 2.0
            cb_y = pos_sens.y0 - 0.065
            cax_sens = fig.add_axes([cb_x, cb_y, cb_w, 0.015])
            if c_sens is not None:
                cb_sens = fig.colorbar(c_sens, cax=cax_sens, orientation='horizontal')
                unit_label = units if units else ''
                if row_key == 'slp':
                    unit_label = 'hPa'
                elif row_key == 'precip':
                    unit_label = r'mm day$^{-1}$'
                if unit_label:
                    cb_sens.set_label(unit_label, fontsize=FONT_SIZE-4)
                cb_sens.ax.tick_params(labelsize=FONT_SIZE-4)
                apply_cb_tick_format_custom(cb_sens, row_key, 'ctrl')
        except Exception:
            pass

        try:
            pos_diff = axes[irow, 1].get_position()
            cb_w = pos_diff.width * 0.9
            cb_x = pos_diff.x0 + (pos_diff.width - cb_w) / 2.0
            cb_y = pos_diff.y0 - 0.065
            cax_diff = fig.add_axes([cb_x, cb_y, cb_w, 0.015])
            if c_diff is not None:
                cb_diff = fig.colorbar(c_diff, cax=cax_diff, orientation='horizontal')
                unit_label = units if units else ''
                if row_key == 'slp':
                    unit_label = 'Pa'
                elif row_key == 'precip':
                    unit_label = r'mm day$^{-1}$'
                if unit_label:
                    cb_diff.set_label(unit_label, fontsize=FONT_SIZE-4)
                cb_diff.ax.tick_params(labelsize=FONT_SIZE-4)
                apply_cb_tick_format_custom(cb_diff, row_key, 'diff')
        except Exception:
            pass

    plt.show()

def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    def load_res(row_key):
        res = {}
        try:
            ds_ctrl = xr.open_dataset(INPUT_DIR / f"{row_key}_ctrl.nc")
            res['ctrl'] = ds_ctrl['data'].values
            res['lats'] = ds_ctrl['lat'].values
            res['lons'] = ds_ctrl['lon'].values
            ds_ctrl.close()
            
            if row_key == 'precip': res['units'] = 'mm/day'
            elif row_key == 'slp': res['units'] = 'Pa'
            elif row_key == 'sst': res['units'] = 'K'
        except Exception as e:
            print(f"Error loading {row_key}_ctrl: {e}")
            return None

        try:
            ds_diff = xr.open_dataset(INPUT_DIR / f"{row_key}_diff.nc")
            res['diff'] = ds_diff['data'].values
            if 'pval' in ds_diff:
                res['pval'] = ds_diff['pval'].values
            else:
                res['pval'] = None
            ds_diff.close()
        except Exception as e:
            print(f"Error loading {row_key}_diff: {e}")
            return None
        
        res['sens'] = None 
        return res

    sst_res = load_res('sst')
    precip_res = load_res('precip')
    slp_res = load_res('slp')
    
    wind_res = {}
    try:
        ds_ctrl = xr.open_dataset(INPUT_DIR / "slp_ctrl.nc")
        if 'u' in ds_ctrl and 'v' in ds_ctrl:
            wind_res['u_ctrl'] = ds_ctrl['u'].values
            wind_res['v_ctrl'] = ds_ctrl['v'].values
        ds_ctrl.close()
        
        ds_diff = xr.open_dataset(INPUT_DIR / "slp_diff.nc")
        if 'u' in ds_diff and 'v' in ds_diff:
            wind_res['u_diff'] = ds_diff['u'].values
            wind_res['v_diff'] = ds_diff['v'].values
        ds_diff.close()
    except Exception as e:
        print(f"Error loading wind: {e}")
        wind_res = None
    
    if not wind_res: wind_res = None

    land_geoms = None
    land_mask = None
    if USE_LAND_MASK and sst_res is not None:
        lats = sst_res['lats']
        lons = sst_res['lons']
        print("Building land geometry (Natural Earth 50m) and mask for overlay...")
        mask, geoms = get_land_mask_and_geoms(lons, lats, resolution='50m')
        if geoms is None:
            warnings.warn("Land geometries could not be constructed; continuing without overlay.")
        else:
            land_geoms = geoms
            land_mask = mask

    plot_3x2(sst_res, precip_res, slp_res, wind_res=wind_res, 
             land_mask=land_mask, land_geoms=land_geoms, 
             outfile=str(INPUT_DIR / 'figure2.png'))

if __name__ == '__main__':
    main()
