#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import math

# ------------------ Style Config ------------------
FONT_NAME = "Times New Roman"
FONT_SIZE = 18
plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['savefig.dpi'] = 600

AXES_FRAME_LINEWIDTH = 2.0
SPINE_ZORDER = 100
CONTOUR_ZORDER = 1
SIG_ZORDER = 10
COAST_ZORDER = 50

# Input Directory
DATA_DIR = Path(__file__).resolve().parent

# Plot Config
LAT_SLICE = (-10, 50)
LON_SLICE = (-90, 50)
CENTRAL_LON = -10.0
FIGSIZE = (16, 14)

ALPHA = 0.05

# ------------------ Helper Functions ------------------

def make_axes_bold(ax, lw=AXES_FRAME_LINEWIDTH):
    """Make axis spines bold and push them to a high zorder."""
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
        pass

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

def apply_style(ax):
    # Ticks
    xticks_sorted = np.arange(LON_SLICE[0], LON_SLICE[1] + 0.1, 30.0)
    # Y-ticks every 20 degrees, starting from 10S (-10) to 50N
    yticks = np.arange(-10, 51, 20.0)

    minor_xticks = (xticks_sorted[:-1] + xticks_sorted[1:]) / 2.0 if xticks_sorted.size >= 2 else np.array([])

    minor_yticks = (yticks[:-1] + yticks[1:]) / 2.0

    ax.set_xticks(xticks_sorted, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    
    ax.set_xticks(minor_xticks, minor=True, crs=ccrs.PlateCarree())
    ax.set_yticks(minor_yticks, minor=True, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: format_lon_label(x)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, pos: format_lat_label(y)))

    ax.tick_params(axis='both', which='major', direction='out',
                   length=8, width=AXES_FRAME_LINEWIDTH,
                   top=False, bottom=True, left=True, right=False,
                   labelsize=FONT_SIZE - 4)
    ax.tick_params(axis='both', which='minor', direction='out',
                   length=4, width=1.0,
                   top=False, bottom=True, left=True, right=False)
                   
    try:
        ax.set_extent([LON_SLICE[0], LON_SLICE[1], LAT_SLICE[0], LAT_SLICE[1]], crs=ccrs.PlateCarree())
    except Exception:
        pass

    # Coastlines and Borders
    ax.coastlines(resolution='110m', linewidth=1.2, zorder=COAST_ZORDER)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.4, zorder=CONTOUR_ZORDER + 0.1)
    
    ax.gridlines(draw_labels=False, linewidth=0.35, color='gray', linestyle='--', zorder=2)

    make_axes_bold(ax)

def format_cb_tick(val):
    if abs(val) < 1e-12:
        return "0"
    if abs(val) >= 1:
        return str(int(round(val)))
    else:
         return f"{val:.1f}"

def style_colorbar(cb, unit_label):
    cb.ax.tick_params(labelsize=FONT_SIZE-4)
    # Custom tick formatting
    ticks = cb.get_ticks()
    labels = [format_cb_tick(t) for t in ticks]
    cb.set_ticks(ticks)
    cb.set_ticklabels(labels)
    if unit_label:
        cb.set_label(unit_label, fontsize=FONT_SIZE-4)

# ------------------ Main Plotting ------------------

def plot_all():
    # Load Data
    wind_nc = DATA_DIR / 'wind_1000_stats.nc'
    lh_nc = DATA_DIR / 'lhflx_stats.nc'
    cld_nc = DATA_DIR / 'cldtot_stats.nc'

    if not wind_nc.exists() or not lh_nc.exists() or not cld_nc.exists():
        print("Error: Missing NC files. Run figure11.py first to generate data.")
        return

    ds_wind = xr.open_dataset(wind_nc)
    ds_lh = xr.open_dataset(lh_nc)
    ds_cld = xr.open_dataset(cld_nc)

    lons = ds_wind['lon'].values
    lats = ds_wind['lat'].values
    
    # Prepare Plot
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, hspace=0.3, wspace=0.15)
    
    letters = ['(a)', '(b)', '(c)', '(d)']
    titles = [
        'Climatology 1000 hPa Wind', 
        'Changes in 1000 hPa Wind', 
        'Changes in Latent Heat Flux', 
        'Changes in Total Cloud Cover'
    ]
    
    # ---------------- (a) Climatology Wind ----------------
    ax = axes[0, 0]
    data_a = ds_wind['ws_ctrl'].values
    levels_a = np.linspace(0, 10, 21)
    # Changed from YlGnBu to Blues for a lighter/simpler look
    cmap_a = 'Blues'
    
    cf_a = ax.contourf(lons, lats, data_a, levels=levels_a, cmap=cmap_a, extend='both', transform=ccrs.PlateCarree(), zorder=CONTOUR_ZORDER)
    
    # Vectors
    skip = 2
    u_a = ds_wind['u_ctrl'].values
    v_a = ds_wind['v_ctrl'].values
    # Scale increased to 80 to make arrows shorter
    q_a = ax.quiver(lons[::skip], lats[::skip], u_a[::skip, ::skip], v_a[::skip, ::skip], 
                    transform=ccrs.PlateCarree(), scale=160, width=0.004, zorder=102, minlength=0)
    ax.quiverkey(q_a, 0.9, 1.05, 5, '5 m s$^{-1}$', labelpos='E', coordinates='axes', fontproperties={'size': FONT_SIZE-4})
    
    apply_style(ax)
    ax.set_title(f"{letters[0]} {titles[0]}", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    
    # Colorbar
    cax_a = ax.inset_axes([0.15, -0.2, 0.7, 0.05]) # [x, y, width, height]
    cb_a = fig.colorbar(cf_a, cax=cax_a, orientation='horizontal')
    style_colorbar(cb_a, 'm s$^{-1}$')

    # ---------------- (b) Wind Change ----------------
    ax = axes[0, 1]
    data_b = ds_wind['ws_diff'].values
    levels_b = np.linspace(-1, 1, 21)
    cmap_b = 'RdBu_r'
    
    cf_b = ax.contourf(lons, lats, data_b, levels=levels_b, cmap=cmap_b, extend='both', transform=ccrs.PlateCarree(), zorder=CONTOUR_ZORDER)
    
    # Significance
    pval_ws = ds_wind['pval_ws'].values
    sig_mask_b = pval_ws < ALPHA
    cs_b = ax.contourf(lons, lats, sig_mask_b, levels=[0.5, 1.5], colors='none', hatches=['///'], transform=ccrs.PlateCarree(), zorder=SIG_ZORDER)
    for collection in cs_b.collections:
        collection.set_edgecolor('white')
        collection.set_linewidth(0.5)
    
    # Vectors
    u_b = ds_wind['u_diff'].values
    v_b = ds_wind['v_diff'].values
    # Skip matches (a) for consistency, or kept denser if needed? User said "ab too sparse". So use skip=2 here too.
    q_b = ax.quiver(lons[::skip], lats[::skip], u_b[::skip, ::skip], v_b[::skip, ::skip], 
                    transform=ccrs.PlateCarree(), scale=20, width=0.004, zorder=102, minlength=0)
    ax.quiverkey(q_b, 0.9, 1.05, 1, '1 m s$^{-1}$', labelpos='E', coordinates='axes', fontproperties={'size': FONT_SIZE-4})
    
    apply_style(ax)
    ax.set_title(f"{letters[1]} {titles[1]}", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    
    cax_b = ax.inset_axes([0.15, -0.2, 0.7, 0.05])
    cb_b = fig.colorbar(cf_b, cax=cax_b, orientation='horizontal')
    style_colorbar(cb_b, 'm s$^{-1}$')

    # ---------------- (c) LH Change ----------------
    ax = axes[1, 0]
    data_c = ds_lh['diff'].values
    levels_c = np.linspace(-15, 15, 21)
    cmap_c = 'BrBG'
    
    cf_c = ax.contourf(lons, lats, data_c, levels=levels_c, cmap=cmap_c, extend='both', transform=ccrs.PlateCarree(), zorder=CONTOUR_ZORDER)
    
    pval_lh = ds_lh['pval'].values
    sig_mask_c = pval_lh < ALPHA
    cs_c = ax.contourf(lons, lats, sig_mask_c, levels=[0.5, 1.5], colors='none', hatches=['///'], transform=ccrs.PlateCarree(), zorder=SIG_ZORDER)
    for collection in cs_c.collections:
        collection.set_edgecolor('white')
        collection.set_linewidth(0.5)
    
    apply_style(ax)
    ax.set_title(f"{letters[2]} {titles[2]}", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    
    cax_c = ax.inset_axes([0.15, -0.2, 0.7, 0.05])
    cb_c = fig.colorbar(cf_c, cax=cax_c, orientation='horizontal')
    style_colorbar(cb_c, 'W m$^{-2}$')

    # ---------------- (d) Cloud Change ----------------
    ax = axes[1, 1]
    # CLDTOT is fraction (0-1), convert to %
    data_d = ds_cld['diff'].values * 100
    levels_d = np.linspace(-5, 5, 21)
    cmap_d = 'RdBu_r'
    
    cf_d = ax.contourf(lons, lats, data_d, levels=levels_d, cmap=cmap_d, extend='both', transform=ccrs.PlateCarree(), zorder=CONTOUR_ZORDER)
    
    pval_cld = ds_cld['pval'].values
    sig_mask_d = pval_cld < ALPHA
    cs_d = ax.contourf(lons, lats, sig_mask_d, levels=[0.5, 1.5], colors='none', hatches=['///'], transform=ccrs.PlateCarree(), zorder=SIG_ZORDER)
    for collection in cs_d.collections:
        collection.set_edgecolor('white')
        collection.set_linewidth(0.5)
    
    apply_style(ax)
    ax.set_title(f"{letters[3]} {titles[3]}", loc='left', fontsize=FONT_SIZE+2, fontweight='bold')
    
    cax_d = ax.inset_axes([0.15, -0.2, 0.7, 0.05])
    cb_d = fig.colorbar(cf_d, cax=cax_d, orientation='horizontal')
    style_colorbar(cb_d, '%')

    plt.show()

if __name__ == '__main__':
    plot_all()
