# -*- coding: utf-8 -*-
"""
按原论文Figure 9的画法出图：直接contourf，不额外高斯平滑。
主要修改第1部分；直接运行本文件即可。

耦合DGPI/ENGPI直接使用原论文RH700版本；非耦合保留当前30年结果。
第三行计算为：原论文耦合响应减去非耦合响应。
原论文的色块：连续256色图按区间中点取色；不是先取10种颜色再BoundaryNorm。
若要核对旧图本身，把FIGURE_TYPE改为"paper_check"，此时使用与你截图吻合的旧论文RH700版本。
"""

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

from pathlib import Path
import json
import hashlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, NullLocator
from matplotlib.patches import Rectangle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# %% 1. 参数区：平时只改这里 ========================================

HERE = Path(__file__).resolve().parent
# 非耦合用当前30年未平滑NC；耦合用下面的原论文RH700 NC。
NC_FILE = HERE / "gpi_comparison_all30y_raw.nc"
PAPER_REFERENCE = HERE / "paper_figure9_rh700_reference.nc"

FIGURE_TYPE = "response"       # 六面板：耦合响应、非耦合响应、两者之差
# FIGURE_TYPE = "dgpi"         # 四面板：四组DGPI绝对值
# FIGURE_TYPE = "engpi"        # 四面板：四组ENGPI绝对值
# FIGURE_TYPE = "paper_check"  # 四面板：原论文RH700的两种气候态和两种差值

OUTPUT_DIR = HERE / "图件"
FILE_TAG = "3x2_bold_ticks_ggw_ctl_titles"
DPI = 600
SHOW_FIGURE = False
SAVE_PNG, SAVE_PDF, SAVE_SVG = True, True, True

FONT_NAME = "Times New Roman"
FONT_SIZE = 10                # 全部文字统一字号，只有子图标题加粗
TITLE_PAD = 5
BORDER_WIDTH = 0.75
MAP_BORDER_WIDTH = 1.5
MINOR_TICK_WIDTH = 1.25
TICK_LENGTH, TICK_PAD = 5.0, 2
MINOR_TICK_LENGTH = 2.5
X_LABEL, Y_LABEL = "", ""

LON_MIN, LON_MAX = -90, -10
LAT_MIN, LAT_MAX = 0, 35
LON_TICK_START, LON_TICK_END, LON_TICK_STEP = -90, -10, 30
LAT_TICK_START, LAT_TICK_END, LAT_TICK_STEP = 0, 35, 20
LON_MINOR_COUNT, LAT_MINOR_COUNT = 1, 1   # 每个主刻度间隔内次刻度数量，不标标签

SHOW_GRID = False              # 原论文图没有灰色网格；需要时改True
GRID_COLOR, GRID_WIDTH, GRID_STYLE = "0.60", 0.5, "--"
SHOW_COAST, SHOW_BORDERS = True, True
COAST_WIDTH, COUNTRY_BORDER_WIDTH = 0.45, 0.3  # 恢复50m海岸线与国界
SHOW_MDR = True
MDR_BOX = (-70, -17.5, 10, 20)
MDR_COLOR, MDR_WIDTH, MDR_STYLE = "black", 0.75, "-"

# ----- 色标：严格恢复原论文的数值范围、分级和取色方法 -----
RESPONSE_CMAP = "RdBu_r"
ABSOLUTE_CMAP = "YlOrRd"
RESPONSE_LEVELS = np.linspace(-1.0, 1.0, 11)   # 10个色块，每0.2一档
ABSOLUTE_LEVELS = np.linspace(0.0, 5.0, 11)    # 10个色块，每0.5一档
# None：沿用原脚本的自动刻度。差值显示-0.8,-0.4,0.0,0.4,0.8；
# 气候态显示0,1,2,3,4,5。需要自定时可改为一个列表。
RESPONSE_TICKS = None
ABSOLUTE_TICKS = None
COLORBAR_LABEL = ""             # 两指数无量纲，沿用原图不单写Dimensionless
COLORBAR_TICK_LENGTH, COLORBAR_TICK_PAD = 2, 1.5

# ----- 排版：毫米为单位，经纬度保持1:1 -----
FIGURE_WIDTH_MM = 183
LEFT_MARGIN_MM, RIGHT_MARGIN_MM, TOP_MARGIN_MM = 11.5, 3.5, 7.5
COLUMN_GAP_MM = 11
ROW_EXTRA_GAP_MM = 6
COLORBAR_GAP_MM = 7.0  # Leave room for the thicker ticks and their labels.
COLORBAR_BOTTOM_SPACE_MM = 7
COLORBAR_WIDTH_RATIO = 1.0      # 恢复原图：色标含两端三角与地图等宽
COLORBAR_ASPECT = 40            # 原colorbar参数aspect=30，宽/高；越大越细

RESPONSE_TITLES = [
    "(a) Uncoupled DGPI (GGW-CTL)", "(b) Uncoupled ENGPI (GGW-CTL)",
    "(c) Coupled DGPI (GGW-CTL)", "(d) Coupled ENGPI (GGW-CTL)",
    "(e) DGPI Response Diff (GGW-CTL)", "(f) ENGPI Response Diff (GGW-CTL)",
]
ABSOLUTE_TITLES = [
    "(a) Coupled CTL", "(b) Coupled GGW",
    "(c) Uncoupled CTL", "(d) Uncoupled GGW",
]
PAPER_TITLES = [
    "(a) DGPI Climatology", "(b) ENGPI Climatology",
    "(c) DGPI Difference (GGW-CTL)", "(d) ENGPI Difference (GGW-CTL)",
]

# %% 2. 直接取原论文耦合场；非耦合取现有30年数据 ====================

with xr.open_dataset(PAPER_REFERENCE) as source:
    paper = source.load()
sources = {"coupled": str(PAPER_REFERENCE)}
source_hashes = {"coupled": hashlib.sha256(PAPER_REFERENCE.read_bytes()).hexdigest()}

if FIGURE_TYPE == "paper_check":
    data = paper.copy()
    output_stem = "paper_reference_rh700"
    print("原论文RH700核对模式：直接绘制原来的四张地图。")
else:
    with xr.open_dataset(NC_FILE) as source:
        current = source.load()
    periods = json.loads(current.attrs.get("periods_json", "{}"))
    noncoupled_cases = ("noncoupled_ctl", "noncoupled_ggw")
    if any(periods.get(case, {}).get("n_years") != 30 for case in noncoupled_cases):
        raise ValueError("非耦合CTL和GGW必须均为30年数据。")
    if current.attrs.get("data_kind") != "raw_unsmoothed":
        raise ValueError("非耦合场必须使用未平滑数据。")
    if (current.attrs.get("rh_hpa") != 700 or current.attrs.get("months") != "7,8,9"
            or current.attrs.get("month_basis") != "filename"):
        raise ValueError("非耦合场要求RH700及实际7—9月。")

    sources["noncoupled"] = str(NC_FILE)
    source_hashes["noncoupled"] = hashlib.sha256(NC_FILE.read_bytes()).hexdigest()
    data = xr.Dataset()
    for index in ("dgpi", "engpi"):
        # 原论文CTL和响应逐值保留，连同其原始网格，不滤波、不插值。
        # reset_coords仅移除重复的辅助坐标，不改变数值或维度坐标。
        ctl = paper[f"{index}_ctrl"].reset_coords(drop=True).copy(deep=True)
        response = paper[f"{index}_diff"].reset_coords(drop=True).copy(deep=True)
        ctl.attrs.update(source_file=str(PAPER_REFERENCE), source_variable=f"{index}_ctrl")
        response.attrs.update(source_file=str(PAPER_REFERENCE), source_variable=f"{index}_diff")
        data[f"{index}_coupled_ctl"] = ctl
        data[f"{index}_coupled_response"] = response
        # 原NC只存CTL和GGW-CTL；GGW由两者相加恢复。
        data[f"{index}_coupled_ggw"] = ctl.astype("float64") + response.astype("float64")
        data[f"{index}_coupled_ggw"].attrs.update(
            source_file=str(PAPER_REFERENCE), definition=f"{index}_ctrl + {index}_diff")

        for case in ("ctl", "ggw", "response"):
            name = f"{index}_noncoupled_{case}"
            data[name] = current[name].copy(deep=True)
            data[name].attrs.update(source_file=str(NC_FILE), source_variable=name)

        # 只为第三列对齐坐标：原网格精确包含非耦合绘图网格，不需插值。
        noncoupled = data[f"{index}_noncoupled_response"]
        coupled_aligned = response.rename(
            {f"lat_{index}": "lat", f"lon_{index}": "lon"}
        ).sel(lat=noncoupled.lat, lon=noncoupled.lon)
        coupled_aligned, noncoupled = xr.align(coupled_aligned, noncoupled, join="exact")
        data[f"{index}_response_difference"] = coupled_aligned - noncoupled
        data[f"{index}_response_difference"].attrs.update(
            definition="Original paper coupled response minus current 30-year atmosphere-only response",
            grid_alignment="Exact coordinate selection; no interpolation",
        )

    data.attrs.update(
        rh_hpa=700,
        data_kind="original_paper_coupled_plus_all30y_noncoupled",
        coupled_month_definition="Original paper time.month selection retained",
        noncoupled_month_definition="Physical July-August-September from filenames",
        period_note="Coupled original 30-year paper fields; atmosphere-only years 0001-0030",
        response_definition="GGW minus CTL; third column = paper coupled response minus atmosphere-only response",
        comparison_note="Source periods are retained as requested; seasonal definitions differ.",
    )
    output_stem = "paper_coupled_rh700_30y_noncoupled"
    print("耦合：直接使用原论文RH700数据；非耦合：保留现有30年数据；第三列已重新相减。")

# 每项依次为：NC变量名、子图标题、色标类型。
if FIGURE_TYPE == "response":
    nrows, ncols = 3, 2
    names = [
        "dgpi_noncoupled_response", "engpi_noncoupled_response",
        "dgpi_coupled_response", "engpi_coupled_response",
        "dgpi_response_difference", "engpi_response_difference",
    ]
    panels = list(zip(names, RESPONSE_TITLES, ["response"] * 6))
elif FIGURE_TYPE in ("dgpi", "engpi"):
    nrows, ncols = 2, 2
    names = [f"{FIGURE_TYPE}_{case}" for case in
             ("coupled_ctl", "coupled_ggw", "noncoupled_ctl", "noncoupled_ggw")]
    panels = list(zip(names, ABSOLUTE_TITLES, ["absolute"] * 4))
elif FIGURE_TYPE == "paper_check":
    nrows, ncols = 2, 2
    panels = list(zip(["dgpi_ctrl", "engpi_ctrl", "dgpi_diff", "engpi_diff"],
                      PAPER_TITLES, ["absolute", "absolute", "response", "response"]))
else:
    raise ValueError("FIGURE_TYPE应为response、dgpi、engpi或paper_check。")

if len(panels) != nrows * ncols:
    raise ValueError("子图标题数量必须与子图数量一致。")

# 按你的要求，将实际绘图数据另存NC到脚本旁；数值不进行额外处理。
used_fields = [variable for variable, _, _ in panels]
plot_data = data[used_fields].copy()
plot_data.attrs.update(
    plotting_method="Original paper contourf; no added Gaussian smoothing",
    sources_json=json.dumps(sources, ensure_ascii=False),
    source_hashes_json=json.dumps(source_hashes),
)
plot_data_file = HERE / f"plot_data_{FIGURE_TYPE}_paper_coupled.nc"
plot_data.to_netcdf(plot_data_file)

# 只用本机已缓存的50m底图，避免触发自动下载。
# 使用随文件夹附带的底图，整体复制后无需原电脑的Cartopy缓存。
cartopy.config["data_dir"] = str(HERE.parent / "map_data")
cartopy.config["pre_existing_data_dir"] = str(HERE.parent / "map_data")
map_root = Path(cartopy.config["data_dir"]) / "shapefiles" / "natural_earth"
for enabled, relative in [
    (SHOW_COAST, "physical/ne_50m_coastline.shp"),
    (SHOW_BORDERS, "cultural/ne_50m_admin_0_boundary_lines_land.shp"),
]:
    if enabled and not (map_root / relative).exists():
        raise FileNotFoundError(f"缺少本地50m底图：{map_root / relative}")

font_manager.findfont(FONT_NAME, fallback_to_default=False)
plt.rcParams.update({
    "font.family": FONT_NAME, "font.size": FONT_SIZE, "font.weight": "normal",
    "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42, "mathtext.fontset": "stix",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

panel_width = (FIGURE_WIDTH_MM - LEFT_MARGIN_MM - RIGHT_MARGIN_MM -
               (ncols - 1) * COLUMN_GAP_MM) / ncols
panel_height = panel_width * (LAT_MAX - LAT_MIN) / (LON_MAX - LON_MIN)
bar_width = panel_width * COLORBAR_WIDTH_RATIO
bar_height = bar_width / COLORBAR_ASPECT
below_panel = COLORBAR_GAP_MM + bar_height + COLORBAR_BOTTOM_SPACE_MM
row_gap = below_panel + ROW_EXTRA_GAP_MM
figure_height = TOP_MARGIN_MM + nrows * panel_height + (nrows - 1) * row_gap + below_panel
fig = plt.figure(figsize=(FIGURE_WIDTH_MM / 25.4, figure_height / 25.4))
panel_audit = []

# %% 3. 画图：contourf的核心调用与原论文一致 =========================

for i, (variable, title, color_kind) in enumerate(panels):
    row, col = divmod(i, ncols)
    left = LEFT_MARGIN_MM + col * (panel_width + COLUMN_GAP_MM)
    bottom = figure_height - TOP_MARGIN_MM - (row + 1) * panel_height - row * row_gap
    ax = fig.add_axes([left / FIGURE_WIDTH_MM, bottom / figure_height,
                       panel_width / FIGURE_WIDTH_MM, panel_height / figure_height],
                      projection=ccrs.PlateCarree())

    field = data[variable]
    # 耦合场保留原图专属坐标；非耦合和第三列使用lat/lon。
    lat_name = next(d for d in field.dims if d.startswith("lat"))
    lon_name = next(d for d in field.dims if d.startswith("lon"))
    lat, lon = field[lat_name].values, field[lon_name].values
    values = field.transpose(lat_name, lon_name).values
    if color_kind == "response":
        levels, cmap_name, ticks = RESPONSE_LEVELS, RESPONSE_CMAP, RESPONSE_TICKS
    else:
        levels, cmap_name, ticks = ABSOLUTE_LEVELS, ABSOLUTE_CMAP, ABSOLUTE_TICKS
    levels = np.asarray(levels, dtype=float)  # 也支持手工填写普通列表
    if (levels.ndim != 1 or len(levels) < 2 or not np.isfinite(levels).all()
            or np.any(np.diff(levels) <= 0)):
        raise ValueError("LEVELS需为至少两个有限数值，且严格递增。")

    # 关键：直接传色图名称，由contourf用连续色图的区间中点取色。
    # 不使用get_cmap(cmap_name, 10)，也不使用BoundaryNorm。
    # 不进行Gaussian滤波、插值加密、补零或数值裁剪。
    filled = ax.contourf(lon, lat, values, levels=levels, cmap=cmap_name,
                         transform=ccrs.PlateCarree(), extend="both", zorder=1)

    if SHOW_COAST:
        ax.coastlines("50m", linewidth=COAST_WIDTH, zorder=6)
    if SHOW_BORDERS:
        ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                        linewidth=COUNTRY_BORDER_WIDTH, linestyle=":", zorder=6)
    if SHOW_MDR:
        x0, x1, y0, y1 = MDR_BOX
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                               edgecolor=MDR_COLOR, linewidth=MDR_WIDTH,
                               linestyle=MDR_STYLE, transform=ccrs.PlateCarree(), zorder=7))

    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(LON_TICK_START, LON_TICK_END + 1e-8, LON_TICK_STEP),
                   crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(LAT_TICK_START, LAT_TICK_END + 1e-8, LAT_TICK_STEP),
                   crs=ccrs.PlateCarree())
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.xaxis.set_minor_locator(AutoMinorLocator(LON_MINOR_COUNT+1) if LON_MINOR_COUNT else NullLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(LAT_MINOR_COUNT+1) if LAT_MINOR_COUNT else NullLocator())
    ax.tick_params(which="major", direction="out", length=TICK_LENGTH, pad=TICK_PAD,
                    width=MAP_BORDER_WIDTH, bottom=True, left=True, top=False, right=False,
                    labelbottom=True, labelleft=True, labeltop=False, labelright=False)
    ax.tick_params(which="minor", direction="out", length=MINOR_TICK_LENGTH,
                    width=MINOR_TICK_WIDTH, bottom=True, left=True, top=False, right=False,
                    labelbottom=False, labelleft=False, labeltop=False, labelright=False)
    if SHOW_GRID:
        ax.grid(which="major", color=GRID_COLOR, linewidth=GRID_WIDTH, linestyle=GRID_STYLE)
    for border in ax.spines.values():
        border.set_color("black")
        border.set_linewidth(MAP_BORDER_WIDTH)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title, loc="left", fontsize=FONT_SIZE+4, fontweight="bold", pad=TITLE_PAD)

    bar_left = left + (panel_width - bar_width) / 2
    bar_bottom = bottom - COLORBAR_GAP_MM - bar_height
    cax = fig.add_axes([bar_left / FIGURE_WIDTH_MM, bar_bottom / figure_height,
                        bar_width / FIGURE_WIDTH_MM, bar_height / figure_height])
    # 原脚本使用colorbar默认刻度和数字格式，自动得到论文图中的标注。
    cbar = fig.colorbar(filled, cax=cax, orientation="horizontal",
                        ticks=ticks, extend="both", spacing="uniform")
    cbar.outline.set_linewidth(BORDER_WIDTH)
    cbar.outline.set_edgecolor("black")
    cbar.ax.tick_params(which="major", length=COLORBAR_TICK_LENGTH, pad=COLORBAR_TICK_PAD,
                        width=BORDER_WIDTH, bottom=True, top=False, labelsize=FONT_SIZE)
    cbar.ax.tick_params(which="minor", length=0, bottom=False, top=False)
    cbar.set_label(COLORBAR_LABEL, fontsize=FONT_SIZE)
    valid = values[np.isfinite(values)]
    panel_audit.append({
        "title": title, "row": row + 1, "column": col + 1,
        "map_border_pt": MAP_BORDER_WIDTH, "major_tick_pt": MAP_BORDER_WIDTH,
        "minor_tick_pt": MINOR_TICK_WIDTH,
        "longitude_major_ticks": ax.get_xticks().tolist(),
        "latitude_major_ticks": ax.get_yticks().tolist(),
        "longitude_minor_ticks": ax.get_xticks(minor=True).tolist(),
        "latitude_minor_ticks": ax.get_yticks(minor=True).tolist(),
        "variable": variable, "levels": levels.tolist(), "ticks": cbar.get_ticks().tolist(),
        "cmap": filled.cmap.name, "cmap_N": filled.cmap.N, "norm": type(filled.norm).__name__,
        "rgba_including_extensions": filled.to_rgba(filled.cvalues).tolist(),
        "finite_count": int(valid.size), "minimum": float(valid.min()), "maximum": float(valid.max()),
        "below_colorbar": int((valid < levels[0]).sum()), "above_colorbar": int((valid > levels[-1]).sum()),
    })

# %% 4. 保存：不会覆盖之前的平滑版本 =================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_name = f"{FIGURE_TYPE}_{output_stem}_{FILE_TAG}"
if SAVE_PNG:
    fig.savefig(OUTPUT_DIR / f"{output_name}.png", dpi=DPI, facecolor="white")
if SAVE_PDF:
    fig.savefig(OUTPUT_DIR / f"{output_name}.pdf", facecolor="white")
if SAVE_SVG:
    fig.savefig(OUTPUT_DIR / f"{output_name}.svg", facecolor="white")
audit = {
    "sources": sources, "source_sha256": source_hashes,
    "extra_smoothing": False, "plot_data_nc": str(plot_data_file),
    "figure_type": FIGURE_TYPE, "panels": panel_audit,
    "layout": {"rows": nrows, "columns": ncols, "width_mm": FIGURE_WIDTH_MM,
               "height_mm": figure_height},
}
(OUTPUT_DIR / f"{output_name}_绘图核查.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已保存到：{OUTPUT_DIR / output_name}")
if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)
