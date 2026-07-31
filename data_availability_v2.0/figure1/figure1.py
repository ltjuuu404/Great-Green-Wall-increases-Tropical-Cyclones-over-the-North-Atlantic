import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import matplotlib.ticker as mticker
import matplotlib

plt.switch_backend('Agg')

# Set DPI
matplotlib.rcParams['figure.dpi'] = 600
matplotlib.rcParams['savefig.dpi'] = 600
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42

# ------------------ Parameters from pft_choosetype_retick.py ------------------
fig_width = 12.0
fig_height = 15.0

# Global adjustments
title_y_offset = -0.03
cbar_x_offset = 0.0
axes_frame_linewidth = 1.5
cbar_label_x_offset = 0
x_ticklabel_pad_offset = -10
extra_bottom_space = 0.04

# Lat/Lon ranges (0-360 for internal logic, converted for plotting)
lat_min, lat_max = 0, 30
lon_min, lon_max = 340, 40

# Visualization settings
levels = np.linspace(0, 100, 11)
n_intervals = len(levels) - 1
base_cmap = plt.cm.Greens

# Fonts
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
title_fontsize = 20
other_fontsize = 16

# Construct colormap
if n_intervals < 2:
    raise ValueError("Need at least two intervals.")
green_colors = base_cmap(np.linspace(0.15, 1.0, n_intervals - 1))
colors = np.vstack(([1.0, 1.0, 1.0, 1.0], green_colors))
cmap = ListedColormap(colors)
norm = BoundaryNorm(boundaries=levels, ncolors=cmap.N, clip=False)

# Diff Color Settings
diff_pos_color_options = [
    ("DarkGreen", "#006400"),
    ("ForestGreen", "#228B22"),
    ("SeaGreen", "#2E8B57"),
    ("LimeGreen", "#32CD32"),
    ("SpringGreen", "#00FF7F")
]
diff_choice_index = 2
diff_pos_color_name, diff_pos_color_hex = diff_pos_color_options[diff_choice_index]
print(f"Diff positive color chosen: {diff_pos_color_name} ({diff_pos_color_hex})")

# ------------------ Load all Figure 1 data from one prepared NC file ------------------
data_dir = Path(__file__).resolve().parent
plot_data_file = data_dir / "figure1_plot_data.nc"
if not plot_data_file.exists():
    raise FileNotFoundError(
        f"File not found: {plot_data_file}. Run prepare_figure1_plot_data.py first."
    )

print("Loading prepared Figure 1 data...")
with xr.open_dataset(plot_data_file) as ds:
    arrL_sub = ds['ctl_vegetation_percent'].values
    arrR_sub = ds['ggw_vegetation_percent'].values
    arr_diff = ds['vegetation_change_percent'].values
    latsL_1d = ds['lat'].values
    lonsL_1d = ds['lon'].values
    latsR_1d = latsL_1d.copy()
    lonsR_1d = lonsL_1d.copy()
    gpcp_annual_smoothed = ds['gpcp_annual_precip_smoothed'].values
    gpcp_lats = ds['gpcp_lat'].values
    gpcp_lons = ds['gpcp_lon'].values
    isohyet_levels = ds['isohyet_level'].values

# Create 2D meshgrids for plotting (assuming all files have same grid, which they should)
# Note: subset_by_latlon in make_data.py returned sorted -180..180 lons. 
# The NC files preserved this.
lonsL_sub2d, latsL_sub2d = np.meshgrid(lonsL_1d, latsL_1d)
lonsR_sub2d, latsR_sub2d = np.meshgrid(lonsR_1d, latsR_1d)
# For diff, use the same grid (they are identical)

# ------------------ Plotting ------------------
print("Plotting...")
fig = plt.figure(figsize=(fig_width, fig_height), dpi=600)

gs_bottom = 0.06 + extra_bottom_space
if gs_bottom >= 0.45:
    gs_bottom = 0.45
gs = fig.add_gridspec(3, 1, left=0.08, right=0.86, top=0.93, bottom=gs_bottom, hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=0))
ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree(central_longitude=0))

title_base_y = 1.02
ax1.set_title("(a) CTL", fontsize=title_fontsize, fontfamily='Times New Roman',
              fontweight='bold', loc='left', y=title_base_y + title_y_offset)
ax2.set_title("(b) GGW", fontsize=title_fontsize, fontfamily='Times New Roman',
              fontweight='bold', loc='left', y=title_base_y + title_y_offset)
ax3.set_title("(c) GGW-CTL", fontsize=title_fontsize, fontfamily='Times New Roman',
              fontweight='bold', loc='left', y=title_base_y + title_y_offset)

for ax in (ax1, ax2, ax3):
    ax.add_feature(cfeature.LAND.with_scale('10m'), zorder=0, alpha=0.7)
    ocean_mask = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                                              edgecolor='face',
                                              facecolor='white')
    ax.add_feature(ocean_mask, zorder=2)
    ax.coastlines(resolution='10m', zorder=3, linewidth=0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(axes_frame_linewidth)
        spine.set_edgecolor('black')

# Helper for ticks
def lon360_to_lon180(l):
    return ((l + 180.0) % 360.0) - 180.0

def nicestep(range_width, n_desired):
    if range_width <= 0: return 1.0
    raw = range_width / max(1, (n_desired - 1))
    exp = np.floor(np.log10(raw))
    base = 10.0 ** exp
    candidates = np.array([1.0, 2.0, 5.0, 10.0]) * base
    counts = np.ceil(range_width / candidates) + 1
    idx = np.argmin(np.abs(counts - n_desired))
    return float(candidates[idx])

lon_min180 = lon360_to_lon180(lon_min)
lon_max180 = lon360_to_lon180(lon_max)
if lon_min180 > lon_max180:
    lon_min180, lon_max180 = lon_max180, lon_min180

# Ticks configuration
step_lon = 20.0
xticks_major = np.arange(np.floor(lon_min180 / step_lon) * step_lon,
                         np.ceil(lon_max180 / step_lon) * step_lon + 0.1 * step_lon,
                         step_lon)
xticks_major = xticks_major[(xticks_major >= lon_min180 - 1e-6) & (xticks_major <= lon_max180 + 1e-6)]

step_lon_minor = 10.0
xticks_minor = np.arange(np.floor(lon_min180 / step_lon_minor) * step_lon_minor,
                         np.ceil(lon_max180 / step_lon_minor) * step_lon_minor + 1e-6,
                         step_lon_minor)
xticks_minor = xticks_minor[(xticks_minor >= lon_min180 - 1e-6) & (xticks_minor <= lon_max180 + 1e-6)]
xticks_minor = np.setdiff1d(xticks_minor, xticks_major)

step_lat = 10.0
yticks_major = np.arange(np.floor(lat_min / step_lat) * step_lat,
                         np.ceil(lat_max / step_lat) * step_lat + 0.1 * step_lat,
                         step_lat)
yticks_major = yticks_major[(yticks_major >= lat_min - 1e-6) & (yticks_major <= lat_max + 1e-6)]
if 0.0 >= lat_min and 0.0 <= lat_max:
     if not np.any(np.isclose(yticks_major, 0.0)):
         yticks_major = np.sort(np.concatenate((yticks_major, [0.0])))

step_lat_minor = 5.0
yticks_minor = np.arange(np.floor(lat_min / step_lat_minor) * step_lat_minor,
                         np.ceil(lat_max / step_lat_minor) * step_lat_minor + 0.1 * step_lat_minor,
                         step_lat_minor)
yticks_minor = yticks_minor[(yticks_minor >= lat_min - 1e-6) & (yticks_minor <= lat_max + 1e-6)]
yticks_minor = np.setdiff1d(yticks_minor, yticks_major)

# Gridlines setup
grid_label_size = max(6, other_fontsize - 2)
gls = []
for ax in (ax1, ax2, ax3):
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.xlabel_style = {'size': grid_label_size, 'family': 'Times New Roman', 'weight': 'normal'}
    gl.ylabel_style = {'size': grid_label_size, 'family': 'Times New Roman', 'weight': 'normal'}
    gl.xlocator = mticker.FixedLocator(xticks_major)
    gl.ylocator = mticker.FixedLocator(yticks_major)
    gl.xminor_locator = mticker.FixedLocator(xticks_minor)
    gl.yminor_locator = mticker.FixedLocator(yticks_minor)
    gl.yformatter = mticker.FuncFormatter(
        lambda x, pos: f"{int(x)}°" if np.isfinite(x) else ""
    )
    gls.append(gl)

# Data Plotting
im1 = ax1.pcolormesh(lonsL_sub2d, latsL_sub2d, arrL_sub, cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), shading='auto')
im2 = ax2.pcolormesh(lonsR_sub2d, latsR_sub2d, arrR_sub, cmap=cmap, norm=norm,
                     transform=ccrs.PlateCarree(), shading='auto')

# Diff Plotting
vmin = np.nanmin(arr_diff)
vmax = np.nanmax(arr_diff)
if not np.isfinite(vmax): vmax = 1.0
if not np.isfinite(vmin): vmin = 0.0

if vmin >= 0:
    cmap_diff = LinearSegmentedColormap.from_list('white_to_pos', ['#ffffff', diff_pos_color_hex], N=256)
    levels_diff = np.linspace(0, vmax if vmax > 0 else 1.0, 11)
    norm_diff = BoundaryNorm(boundaries=levels_diff, ncolors=cmap_diff.N, clip=False)
    extend_option = 'max'
else:
    neg_color = '#2166ac'
    pos_color = diff_pos_color_hex
    cmap_diff = LinearSegmentedColormap.from_list('blue_white_pos', [neg_color, '#ffffff', pos_color], N=512)
    levels_diff = np.linspace(vmin, vmax if vmax != 0 else 1.0, 21)
    norm_diff = BoundaryNorm(boundaries=levels_diff, ncolors=cmap_diff.N, clip=False)
    extend_option = 'both'

im3 = ax3.pcolormesh(lonsL_sub2d, latsL_sub2d, arr_diff, cmap=cmap_diff, norm=norm_diff,
                     transform=ccrs.PlateCarree(), shading='auto')

# GPCP Version 2.3 annual precipitation isohyets for 1991-2020.
isohyet_colors = ('#2166ac', '#b2182b')
isohyet_linestyles = ('--', '-')
contours = ax3.contour(
    gpcp_lons,
    gpcp_lats,
    gpcp_annual_smoothed,
    levels=isohyet_levels,
    colors=isohyet_colors,
    linestyles=isohyet_linestyles,
    linewidths=(2.0, 2.0),
    transform=ccrs.PlateCarree(),
    zorder=4,
)
contour_labels = ax3.clabel(
    contours,
    levels=isohyet_levels,
    fmt={100.0: '100', 400.0: '400'},
    inline=True,
    inline_spacing=4,
    fontsize=other_fontsize,
)
for label in contour_labels:
    label.set_fontfamily('Times New Roman')
    label.set_fontweight('normal')

legend_handles = [
    Line2D(
        [0], [0], color=isohyet_colors[0], linestyle=isohyet_linestyles[0],
        linewidth=2.0, label=r'100 mm yr$^{-1}$'
    ),
    Line2D(
        [0], [0], color=isohyet_colors[1], linestyle=isohyet_linestyles[1],
        linewidth=2.0, label=r'400 mm yr$^{-1}$'
    ),
]
ax3.legend(
    handles=legend_handles,
    loc='upper right',
    frameon=False,
    facecolor='white',
    prop={'family': 'Times New Roman', 'size': other_fontsize, 'weight': 'normal'},
)

# Ticks Application
for ax in (ax1, ax2, ax3):
    try:
        ax.set_xticks(xticks_major, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks_major, crs=ccrs.PlateCarree())
        ax.set_xticks(xticks_minor, minor=True, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks_minor, minor=True, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_xticks(xticks_major)
        ax.set_yticks(yticks_major)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_yticks(yticks_minor, minor=True)

    ax.tick_params(axis='x', which='both', labelbottom=False)
    ax.tick_params(axis='y', which='both', labelleft=False)
    ax.tick_params(axis='both', which='major', direction='out', length=10, width=axes_frame_linewidth)
    ax.tick_params(axis='both', which='minor', direction='out', length=5, width=axes_frame_linewidth * 0.4)

# Colorbars
fig.canvas.draw()
pos1 = ax1.get_position()
pos2 = ax2.get_position()
pos3 = ax3.get_position()
axes_right = max(pos1.x1, pos2.x1, pos3.x1)
base_cbar_gap = 0.008
cbar_width = 0.015
cbar_gap = base_cbar_gap + cbar_x_offset

cbar1_x = axes_right + cbar_gap
cbar1_y = pos2.y0
cbar1_height = pos1.y1 - pos2.y0
cax1 = fig.add_axes([cbar1_x, cbar1_y, cbar_width, cbar1_height])
cb1 = fig.colorbar(mappable=im1, cax=cax1, orientation='vertical', boundaries=levels, ticks=np.linspace(0, 100, 6))
cb1.set_label("%", fontsize=other_fontsize, fontfamily='Times New Roman', rotation=90, labelpad=8)
cb1.ax.tick_params(labelsize=other_fontsize)
cb1.set_ticklabels([f"{int(t)}" for t in cb1.get_ticks()])
cb1.ax.yaxis.label.set_fontweight('normal')
for label in cb1.ax.get_yticklabels():
    label.set_fontweight('normal')

cbar2_x = axes_right + cbar_gap
cbar2_y = pos3.y0
cbar2_height = pos3.y1 - pos3.y0
cax2 = fig.add_axes([cbar2_x, cbar2_y, cbar_width, cbar2_height])
cb2 = fig.colorbar(mappable=im3, cax=cax2, orientation='vertical',
                   boundaries=levels_diff, ticks=np.linspace(levels_diff[0], levels_diff[-1], 6),
                   extend=extend_option)
cb2.set_label('%', fontsize=other_fontsize, fontfamily='Times New Roman', rotation=90, labelpad=8)
cb2.ax.tick_params(labelsize=other_fontsize)
cb2.ax.yaxis.label.set_fontweight('normal')
for label in cb2.ax.get_yticklabels():
    label.set_fontweight('normal')

# Colorbar label offset
cb_label_x_base = 1.05
label_offset_axis = (cbar_label_x_offset * 0.01)
try:
    cb1.ax.yaxis.set_label_coords(cb_label_x_base + label_offset_axis, 0.5)
    cb2.ax.yaxis.set_label_coords(cb_label_x_base + label_offset_axis, 0.5)
except Exception:
    pass

# Tick-label formatting
def format_lat_label(lat):
    if np.isclose(lat, 0.0): return "0°"
    elif lat > 0: return f"{int(np.round(lat))}°N"
    else: return f"{int(np.round(abs(lat)))}°S"


def format_lon_label(lon):
    if np.isclose(lon, 0.0): return "0°"
    elif lon > 0: return f"{int(np.round(lon))}°E"
    else: return f"{int(np.round(abs(lon)))}°W"


lat_range = lat_max - lat_min
pad = max(0.1, lat_range * 0.01)
lat_min_plot = lat_min
lat_max_plot = lat_max

if abs(lat_min_plot - 0.0) < 1e-8: lat_min_plot = max(-90.0, lat_min_plot - pad)
if abs(lat_max_plot - 0.0) < 1e-8: lat_max_plot = min(90.0, lat_max_plot + pad)

xticklabels_strings = [format_lon_label(x) for x in xticks_major]
base_xtick_pad = 14
pad_val = base_xtick_pad + x_ticklabel_pad_offset
if pad_val is None: pad_val = base_xtick_pad

for ax, gl in zip((ax1, ax2, ax3), gls):
    ax.title.set_fontsize(title_fontsize)
    ax.set_extent([lon_min180, lon_max180, lat_min_plot, lat_max_plot], crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', which='major', labelsize=other_fontsize)
    for t in ax.xaxis.get_ticklabels():
        t.set_fontfamily('Times New Roman')
        t.set_fontweight('normal')
    for t in ax.yaxis.get_ticklabels():
        t.set_fontfamily('Times New Roman')
        t.set_fontweight('normal')

    try:
        gl.left_labels = False
        gl.bottom_labels = False
        gl.top_labels = False
        gl.right_labels = False
    except Exception: pass

    yticks_in = np.array([t for t in yticks_major if (lat_min_plot - 1e-8) <= t <= (lat_max_plot + 1e-8)])
    if (lat_min_plot - 1e-8) <= 0.0 <= (lat_max_plot + 1e-8) and 0.0 not in yticks_in:
        yticks_in = np.unique(np.concatenate((yticks_in, [0.0])))
        yticks_in.sort()

    if yticks_in.size == 0:
        n_yt = 5
        step = nicestep(lat_max_plot - lat_min_plot, n_yt)
        yticks_in = np.arange(np.ceil(lat_min_plot / step) * step, lat_max_plot + 1e-6, step)
        yticks_in = yticks_in[(yticks_in >= lat_min_plot - 1e-6) & (yticks_in <= lat_max_plot + 1e-6)]

    try:
        ax.set_yticks(yticks_in, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_yticks(yticks_in)

    ylabels = [format_lat_label(y) for y in yticks_in]
    ax.set_yticklabels(
        ylabels,
        fontsize=grid_label_size,
        family='Times New Roman',
        weight='normal',
    )
    ax.tick_params(axis='y', labelleft=True)

    try:
        ax.set_xticks(xticks_major, crs=ccrs.PlateCarree())
    except Exception:
        ax.set_xticks(xticks_major)

    ax.tick_params(axis='x', which='major', labelbottom=True, pad=pad_val)
    ax.set_xticklabels(
        xticklabels_strings,
        fontsize=grid_label_size,
        family='Times New Roman',
        weight='normal',
    )

output_stem = data_dir / "figure1"
fig.savefig(output_stem.with_suffix('.png'), dpi=600, facecolor='white')
fig.savefig(output_stem.with_suffix('.pdf'), facecolor='white')
fig.savefig(output_stem.with_suffix('.svg'), facecolor='white')
plt.close(fig)
print(f"Saved: {output_stem.with_suffix('.png')}")
print(f"Saved: {output_stem.with_suffix('.pdf')}")
print(f"Saved: {output_stem.with_suffix('.svg')}")
