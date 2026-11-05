#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot_tc_generation_genesis_only_fixed_jas_composite.py
（已修改）
- 修复 x 轴最左侧经度标签不显示的问题（将刻度位置正规化到以 CENTRAL_LON 为中心的区间，并按从左到右排序后设置）
- 将 colorbar 修改为“分级设色”（discrete / boundary colormap）
- 其它与原始脚本一致的改进（contourf+contour、land mask、lat cutoff、grid/smooth 等）
- 新增：将绘图经纬度范围设为 120W 到 30E，并使所有子图的经纬标识与 diff 子图一致
- 新增：读取 IBTrACS CSV 并把 JAS（7/8/9）生成点标注在 control 图上
- 新增：全局变量 IBTRACS_YEAR_RANGE 用于按年份区间过滤 CSV（格式：(min_year, max_year)，值为 None 表示不限制）
- 新增（2025-11-20）：将 diff colorbar 垂直居中并缩短使其不超出 diff 子图。可通过 DIFF_CBAR_SCALE 调整缩放比例。
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

import os, glob, re
import numpy as np
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# [新] 导入平滑所需的工具
import matplotlib as mpl
# 保持交互窗口显示为常规分辨率（不要把 figure.dpi 设得太大）
mpl.rcParams['figure.dpi'] = 100  # 交互窗口显示 DPI，按需改为 100-150

# 指定默认保存分辨率（如果你用 fig.savefig 不指定 dpi，这里会被用到）
mpl.rcParams['savefig.dpi'] = 600  # 文件保存时默认 600 dpi（高分辨率）
box_lon_min, box_lon_max = -70.0, -17.5
box_lat_min, box_lat_max = 10.0, 20.0

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy.ndimage 未找到，将跳过高斯平滑。")
# try cartopy for nicer maps / land masking
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    print("cartopy not available; plotting without coastlines/land-mask.")

# -------------- 用户参数 ----------------
PROCESSED_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JAS_TC_genesis_2p5deg_processed.nc")

# --- 修改：设置绘图经纬度为 120W 到 30E ---
LON_MIN, LON_MAX = -90.0, 30.0  # 120W .. 30E
LAT_MIN, LAT_MAX = 0.0, 40.0

# --- [新增] 自动计算中心经度 ---
CENTRAL_LON = (LON_MIN + LON_MAX) / 2.0  # e.g., -45.0 for [-120, 30]

# --- [新增] 网格分辨率与平滑参数 ---
GRID_RES = 2.5  # 统计网格的分辨率 (单位: 度)
SMOOTH_SIGMA = 1.0  # 高斯平滑的标准差 (单位: 网格点), 设为 0 或 None 则不平滑

# --- 只在统计中保留纬度 <= LAT_STAT_MAX（去掉北于北纬30度） ---
LAT_STAT_MAX = None  # 南北限的统计上限（单位：度北），设为 None 则不做额外裁剪

# IBTrACS CSV 路径（你给定的文件）
# IBTrACS genesis points are included in PROCESSED_DATA_FILE.

# 新增全局变量：按年份区间过滤 IBTrACS CSV（inclusive）
# 设为 (min_year, max_year)。例如 (1950, 2010)；若不想按年份筛选请设置为 (None, None)
IBTRACS_YEAR_RANGE = (1991, 2020)

# number of discrete bins for top colorbars (useful for BoundaryNorm)
CB_N_BINS = 11     # for diff colorbar
CB_N_BINS_TOP = 15 # for top shared colorbar (control & sensitive)
FIG_DPI = 180

# ---------------- Global colorbar limits (set to None for auto) ----------------
# For top shared colorbar (control & sensitive)
TOP_CBAR_VMIN = 0  # e.g. 0.00 or None
TOP_CBAR_VMAX = 0.15  # e.g. 0.20 or None
# diff colorbar: alongside ax3 (延长colorbar)
extension_factor = 2  # 延长 factor (legacy — 不再用于 diff colorbar 实际高度计算)
# For diff colorbar (sensitive - control)
DIFF_CBAR_VMIN = -0.03  # e.g. -0.05 or None
DIFF_CBAR_VMAX = 0.03  # e.g. 0.05 or None
# -----------------------------------------------------------------------

# colorbar layout (vertical)
CB_WIDTH = 0.01  # width in figure coords for vertical colorbars
CB_GAP = 0.01  # gap between axes and colorbar

# ---------------- New: vertical manual offsets for colorbars (figure coords) ----------------
# 修改这两个值可以手动垂直平移 colorbar（正值向上移动，负值向下移动）
CB1_Y_SHIFT = 0.05  # shared top colorbar vertical shift (figure coordinate units)
CB2_Y_SHIFT = 0.015  # diff colorbar vertical shift (figure coordinate units)
# -----------------------------------------------------------------------------------------

# 新增：diff colorbar 相对于 diff 子图高度的缩放比例（0..1），用于缩短 colorbar 使其不超出子图高度
DIFF_CBAR_SCALE = 0.92  # 1.0 表示与 diff 子图等高，<1 表示短一些

out_fig = os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_sensitive_diff_jas_composite.svg")
PALETTE = ['white', '#fff7bc', '#fee391', '#fdae61', '#f46d43', '#d73027', '#a50026']
CMAP_DIFF_NAME = 'RdBu_r'

# fonts / sizes
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22
TITLE_FONTSIZE = 34
OTHER_FONTSIZE = 32
ANNOT_FONTNAME = "Times New Roman"
ANNOT_FONTSIZE = 28

# layout
FIG_WIDTH = 12
FIG_HEIGHT = 11
BOTTOM_MARGIN = 0.12
axes_frame_linewidth = 1.5
Y_MAJOR_TICK_STEP = 20.0
Y_MINOR_TICK_STEP = 10.0

# helper
def normalize_lon(lon, central_lon):
    """
    将经度归一化到 [central_lon-180, central_lon+180] 的区间内（可用于投影/定位等）。
    适用于标量或 numpy 数组。
    """
    lon_arr = np.asarray(lon, dtype=float)
    out = lon_arr.copy()
    low = central_lon - 180.0
    high = central_lon + 180.0
    out = (out - central_lon + 180.0) % 360.0 - 180.0 + central_lon
    return out

def normalize_lon_to_180(lon):
    """
    将经度标准化到 [-180, 180] 便于格式化为 E/W（不改变 central reference）。
    """
    lon_arr = np.asarray(lon, dtype=float)
    return (lon_arr + 180.0) % 360.0 - 180.0

def normalize_lon_to_center(lon, central_lon):
    """
    更明确的将 lon 缩放到 [central_lon-180, central_lon+180] 区间（返回同样形状）。
    该实现确保数值落在该闭区间，便于设置 tick 位置（避免在 Cartopy 中出现“缺失”刻度）。
    """
    lon_arr = np.asarray(lon, dtype=float)
    # 先标准化到 [-180,180]
    lon_norm180 = (lon_arr + 180.0) % 360.0 - 180.0
    offset = ((central_lon + 180.0) % 360.0) - 180.0
    return normalize_lon(lon_arr, central_lon)

def make_global_grid(resolution=5.0, central_lon=0.0):
    """
    创建一个以 central_lon 为中心、指定分辨率的全球网格。
    """
    lon_start = central_lon - 180.0
    lon_end = central_lon + 180.0
    lon_edges = np.arange(lon_start, lon_end + resolution, resolution)
    lat_edges = np.arange(-90.0, 90.0 + resolution, resolution)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    return lon_centers, lat_centers, lon_edges, lat_edges

def find_systems_files_recursive(root_dir, patterns):
    found = []
    for path, dirs, files in os.walk(root_dir):
        for fn in files:
            for pat in patterns:
                if glob.fnmatch.fnmatch(fn, pat):
                    found.append(os.path.join(path, fn))
    if not found:
        for path, dirs, files in os.walk(root_dir):
            for fn in files:
                ln = fn.lower()
                if 'systems' in ln or 'all_year' in ln or 'system' in ln:
                    found.append(os.path.join(path, fn))
    return sorted(list(set(found)))

def is_integer_like_array(a):
    try:
        arr = np.array(a)
        if np.issubdtype(arr.dtype, np.integer): return True
        return np.all(np.mod(arr.astype(float), 1.0) == 0.0)
    except Exception:
        return False

def detect_month_year_from_vars(ds, npoints, exclude_vars=set()):
    months = None;
    years = None
    for v in ds.variables:
        if v in exclude_vars: continue
        try:
            arr = np.array(ds.variables[v][:]).ravel()
        except Exception:
            continue
        if arr.size != npoints: continue
        lname = v.lower()
        if 'month' in lname or lname == 'mon':
            if is_integer_like_array(arr): return arr.astype(int).tolist(), years
        if is_integer_like_array(arr):
            valsf = arr.astype(float)
            if np.nanmin(valsf) >= 1 and np.nanmax(valsf) <= 12:
                return arr.astype(int).tolist(), years
        units = getattr(ds.variables[v], 'units', None)
        if isinstance(units, str) and 'since' in units:
            try:
                dates = nc.num2date(arr, units)
                months = [d.month for d in dates]
                years = [d.year for d in dates]
                return months, years
            except Exception:
                pass
    return months, years

# (原始的读取/处理函数 — 保持不变)
def read_genesis_from_systems_nc(fn):
    ds = nc.Dataset(fn)
    try:
        varnames = [v for v in ds.variables]
        lat_candidates = [v for v in varnames if 'lat' in v.lower()]
        lon_candidates = [v for v in varnames if 'lon' in v.lower()]
        varnames_lower = {vn.lower(): vn for vn in varnames}
        jet_names_try = ['jetcore', 'jet_core', 'jetCore', 'jet_core_flag', 'jetcore_flag', 'jet_coreflag']
        for la in lat_candidates:
            for lo in lon_candidates:
                try:
                    latv = np.array(ds.variables[la][:])
                    lonv = np.array(ds.variables[lo][:])
                except Exception:
                    continue
                if latv.shape != lonv.shape:
                    continue
                flat_lat = latv.ravel();
                flat_lon = lonv.ravel()
                npoints = flat_lat.size
                if npoints == 0:
                    continue
                jetcore_arr = None
                jetcore_name = None
                for try_name in jet_names_try:
                    if try_name.lower() in varnames_lower:
                        jetcore_name = varnames_lower[try_name.lower()]
                        try:
                            tmp = np.array(ds.variables[jetcore_name][:]).ravel()
                            if tmp.size == npoints:
                                try:
                                    jetcore_arr = np.asarray(tmp).astype(np.int8)
                                except Exception:
                                    jetcore_arr = (np.asarray(tmp) != 0).astype(np.int8)
                        except Exception:
                            jetcore_arr = None
                        break
                id_var = None
                id_arr = None
                for v in varnames:
                    if v in (la, lo): continue
                    lname = v.lower()
                    if any(k in lname for k in ('id', 'storm', 'track', 'sys', 'cyc', 'sid', 'system_id')):
                        try:
                            tmp = np.array(ds.variables[v][:]).ravel()
                            if tmp.size == npoints:
                                id_var = v
                                id_arr = tmp
                                break
                        except Exception:
                            continue
                exclude_ids = set()
                if jetcore_arr is not None and id_arr is not None:
                    try:
                        unique_ids = np.unique(id_arr[~np.isnan(id_arr)]) if np.issubdtype(id_arr.dtype, np.number) else np.unique(id_arr)
                        for uid in unique_ids:
                            if np.issubdtype(id_arr.dtype, np.number):
                                idxs = np.where(id_arr == uid)[0]
                            else:
                                idxs = np.where(np.array(id_arr) == uid)[0]
                            if idxs.size == 0: continue
                            if np.any(jetcore_arr[idxs] == 1):
                                exclude_ids.add(uid)
                    except Exception:
                        exclude_ids = set()
                genesis_mask = None
                used_rule = None
                flag_vars = [v for v in varnames if v not in (la, lo)
                             and any(k in v.lower() for k in ('gen', 'birth', 'start', 'isgen', 'origin', 'first'))]
                for v in flag_vars:
                    try:
                        arr = np.array(ds.variables[v][:]).ravel()
                    except Exception:
                        continue
                    if arr.size != npoints: continue
                    if arr.dtype.kind in ('S', 'U', 'O'):
                        try:
                            s = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr])
                        except Exception:
                            s = arr.astype(str)
                        s = np.char.lower(s)
                        mask = np.array([('gen' in ss) or ('birth' in ss) or ('start' in ss) for ss in s])
                        if mask.any():
                            genesis_mask = mask;
                            used_rule = f"flag_str:{v}";
                            break
                    else:
                        try:
                            valid = ~np.isnan(arr)
                            uniq = np.unique(arr[valid]) if valid.any() else np.unique(arr)
                        except Exception:
                            uniq = np.unique(arr)
                        try:
                            uniqf = uniq.astype(float)
                        except Exception:
                            uniqf = uniq
                        if uniqf.size > 0 and np.all(np.isin(uniqf, [0, 1])):
                            mask = (arr == 1)
                            if mask.any():
                                genesis_mask = mask;
                                used_rule = f"flag_num:{v}";
                                break
                        if is_integer_like_array(uniqf) and uniqf.size > 0:
                            minv = np.nanmin(uniqf)
                            mask = (arr == minv)
                            if mask.any():
                                genesis_mask = mask;
                                used_rule = f"flag_min_stage:{v}";
                                break
                if genesis_mask is None:
                    id_var_local = None;
                    time_var = None
                    for v in varnames:
                        if v in (la, lo): continue
                        if any(k in v.lower() for k in ('id', 'storm', 'track', 'sys', 'cyc', 'sid', 'system_id')):
                            try:
                                arr = np.array(ds.variables[v][:]).ravel()
                                if arr.size == npoints:
                                    id_var_local = v;
                                    break
                            except Exception:
                                continue
                    for v in varnames:
                        if v in (la, lo): continue
                        units = getattr(ds.variables[v], 'units', '')
                        if isinstance(units, str) and 'since' in units:
                            try:
                                arr = np.array(ds.variables[v][:]).ravel()
                                if arr.size == npoints:
                                    time_var = v;
                                    break
                            except Exception:
                                continue
                    if id_var_local is not None and time_var is not None:
                        try:
                            id_arr_local = np.array(ds.variables[id_var_local][:]).ravel()
                            time_arr = np.array(ds.variables[time_var][:]).ravel()
                            try:
                                times = nc.num2date(time_arr, getattr(ds.variables[time_var], 'units'))
                            except Exception:
                                times = time_arr.astype(float)
                            genesis_mask = np.zeros(npoints, dtype=bool)
                            for uid in np.unique(id_arr_local):
                                try:
                                    if np.isnan(uid): continue
                                except Exception:
                                    pass
                                idxs = np.where(id_arr_local == uid)[0]
                                if idxs.size == 0: continue
                                try:
                                    sub_times = np.array([t for t in times[idxs]])
                                    argmin_local = idxs[np.argmin(sub_times)]
                                except Exception:
                                    argmin_local = idxs[np.argmin(np.array(times[idxs]).astype(float))]
                                genesis_mask[argmin_local] = True
                            if genesis_mask.any():
                                used_rule = f"id_earliest:{id_var_local}+{time_var}"
                            else:
                                genesis_mask = None
                        except Exception:
                            genesis_mask = None
                if genesis_mask is None:
                    age_var = None
                    for v in varnames:
                        if v in (la, lo): continue
                        if any(k in v.lower() for k in ('age', 'step', 'life', 'itime', 'age_days', 'age_hours')):
                            try:
                                arr_test = np.array(ds.variables[v][:]).ravel()
                                if arr_test.size == npoints:
                                    age_var = v;
                                    break
                            except Exception:
                                continue
                    if age_var is not None:
                        try:
                            arr = np.array(ds.variables[age_var][:]).ravel()
                            mask0 = (arr == 0)
                            if mask0.any():
                                genesis_mask = mask0;
                                used_rule = f"age_eq0:{age_var}"
                            else:
                                if id_arr is not None:
                                    genesis_mask = np.zeros(npoints, dtype=bool)
                                    for uid in np.unique(id_arr):
                                        try:
                                            if np.isnan(uid): continue
                                        except Exception:
                                            pass
                                        idxs = np.where(id_arr == uid)[0]
                                        if idxs.size == 0: continue
                                        argmin_local = idxs[np.argmin(arr[idxs])]
                                        genesis_mask[argmin_local] = True
                                    if genesis_mask.any():
                                        used_rule = f"age_min_per_id:{age_var}"
                                    else:
                                        genesis_mask = None
                                else:
                                    genesis_mask = None
                        except Exception:
                            genesis_mask = None
                if genesis_mask is None or not np.any(genesis_mask):
                    continue
                months, years = detect_month_year_from_vars(ds, npoints, exclude_vars={la, lo})
                pts = []
                for k in range(npoints):
                    if not bool(genesis_mask[k]): continue
                    if jetcore_arr is not None:
                        try:
                            if int(jetcore_arr[k]) == 1:
                                continue
                        except Exception:
                            pass
                    if id_arr is not None and len(exclude_ids) > 0:
                        try:
                            uidk = id_arr[k]
                            if (uidk is not None) and (uidk in exclude_ids):
                                continue
                        except Exception:
                            pass
                    la_v = flat_lat[k];
                    lo_v = flat_lon[k]
                    if np.ma.is_masked(la_v) or np.isnan(la_v): continue
                    if np.ma.is_masked(lo_v) or np.isnan(lo_v): continue
                    m = months[k] if (months is not None and len(months) > k) else None
                    y = years[k] if (years is not None and len(years) > k) else None
                    pts.append({'lat': float(la_v), 'lon': float(lo_v),
                                'month': int(m) if m is not None else None,
                                'year': int(y) if y is not None else None})
                if pts:
                    print(f"从 {os.path.basename(fn)} 识别 {len(pts)} 个 genesis 点（排除 jetcore==1），规则: {used_rule}")
                ds.close()
                return pts
        ds.close()
        print(
            f"[警告] {os.path.basename(fn)}: 未识别到任何 genesis 点（或均被 jetcore 排除），已跳过该文件（不会使用全部轨迹点）")
        return []
    except Exception as e:
        try:
            ds.close()
        except Exception:
            pass
        print(f"跳过 {os.path.basename(fn)}: {e}")
        return []

def read_genesis_from_csv(fn):
    df = pd.read_csv(fn)
    cols = {c.lower(): c for c in df.columns}
    if 'lat' not in cols and 'latitude' not in cols:
        raise ValueError("CSV 必须含 lat 或 Latitude 列")
    if 'lon' not in cols and 'longitude' not in cols:
        raise ValueError("CSV 必须含 lon 或 Longitude 列")
    lat_col = cols.get('lat', cols.get('latitude'))
    lon_col = cols.get('lon', cols.get('longitude'))
    time_col = None
    if 'time' in cols:
        time_col = cols['time']
    elif 'date' in cols:
        time_col = cols['date']
    pts = []
    for _, row in df.iterrows():
        try:
            latv = float(row[lat_col])
            lonv = float(row[lon_col])
        except Exception:
            continue
        m = None
        y = None
        if time_col is not None:
            try:
                t = pd.to_datetime(row[time_col], errors='coerce')
                if not pd.isna(t):
                    m = int(t.month)
                    y = int(t.year)
            except Exception:
                m = None
        pts.append({
            'lat': latv,
            'lon': lonv,
            'month': m,
            'year': y,
            # also keep SID/Number if present
            'SID': row[cols['sid']] if 'sid' in cols else None,
            'Number': row[cols['number']] if 'number' in cols else None,
            'Name': row[cols['name']] if 'name' in cols else None
        })
    return pts

def collect_points_from_files_sorted(filelist):
    points = []
    for idx, fn in enumerate(sorted(filelist)):
        base = os.path.basename(fn)
        try:
            if fn.lower().endswith('.csv'):
                pts = read_genesis_from_csv(fn)
            else:
                pts = read_genesis_from_systems_nc(fn)
            for p in pts:
                p['sim_year'] = idx
                if p.get('year', None) is None:
                    m = re.search(r'(\d{4})', base)
                    if m: p['year'] = int(m.group(1))
            points.extend(pts)
            has_month = any(p.get('month') is not None for p in pts)
            has_year = any(p.get('year') is not None for p in pts)
            print(
                f"读取 {base} -> {len(pts)} genesis points (sim_year={idx}) month_info={'yes' if has_month else 'no'} year_info={'yes' if has_year else 'no'})")
        except Exception as e:
            print(f"跳过 {base}: {e}")
    return points, len(filelist)

def spatial_filter(points, lon_min, lon_max, lat_min, lat_max, central_lon):
    """
    使用归一化经度进行空间过滤。
    """
    out = []
    for p in points:
        lon_norm = normalize_lon(p['lon'], central_lon)
        lat = p['lat']
        if lon_min <= lon_norm <= lon_max and lat_min <= lat <= lat_max:
            np_ = p.copy()
            np_['lon'] = lon_norm
            out.append(np_)
    return out

def bin_points_to_grid(points, lon_edges, lat_edges):
    """
    点已经归一化后的 histogram2d（经度为第一维）。
    """
    if len(points) == 0:
        return np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=float)
    lats = np.array([p['lat'] for p in points])
    lons = np.array([p['lon'] for p in points])
    H, xedges, yedges = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    return H.T

def build_counts_by_sim_year(points, n_sim_years, lon_edges, lat_edges):
    ny = len(lat_edges) - 1;
    nx = len(lon_edges) - 1
    counts = np.zeros((n_sim_years, ny, nx), dtype=float)
    bysim = {}
    for p in points:
        sy = p.get('sim_year', 0)
        bysim.setdefault(sy, []).append(p)
    for sy in range(n_sim_years):
        pts = bysim.get(sy, [])
        counts[sy] = bin_points_to_grid(pts, lon_edges, lat_edges)
    return counts

def selection_mask_for_month(points, months):
    if months is None:
        return np.array([True] * len(points))
    if isinstance(months, int): months = [months]
    mask = []
    for p in points:
        m = p.get('month', None)
        mask.append((m is not None) and (m in months))
    return np.array(mask)

# formatting helpers
def format_ticks_two_decimal(vals):
    """Format a list/array of tick values as two-decimal strings."""
    return [f"{float(v):.2f}" for v in np.asarray(vals)]

def format_lat_label(lat):
    if np.isclose(lat, 0.0):
        return "0°"
    elif lat > 0:
        return f"{int(np.round(lat))}°N"
    else:
        return f"{int(np.round(abs(lat)))}°S"

def format_lon_label(lon):
    """
    格式化经度标签。
    始终将经度标准化到 [-180, 180] 范围以显示 E/W。
    """
    lon_norm = normalize_lon_to_180(lon)
    if np.isclose(lon_norm, 0.0):
        return "0°"
    elif lon_norm > 0:
        return f"{int(np.round(lon_norm))}°E"
    else:
        return f"{int(np.round(abs(lon_norm)))}°W"

# plotting
def plot_jas_control_sensitive_diff(mean_ctrl_full, mean_sens_full,
                                   lon_min, lon_max, lat_min, lat_max,
                                   lon_edges_full, lat_edges_full,
                                   out_png, central_lon=0.0, grid_res=5.0,
                                   obs_genesis_points=None):  # 新增 obs_genesis_points 参数
    # 网格切片逻辑更新，以匹配以 central_lon 为中心的网格
    grid_lon_start = central_lon - 180.0
    grid_lat_start = -90.0
    lon_idx0 = int(round((lon_min - grid_lon_start) / grid_res))
    lon_idx1 = int(round((lon_max - grid_lon_start) / grid_res))
    lat_idx0 = int(round((lat_min - grid_lat_start) / grid_res))
    lat_idx1 = int(round((lat_max - grid_lat_start) / grid_res))
    lon_idx0 = max(0, lon_idx0);
    lat_idx0 = max(0, lat_idx0)
    lon_idx1 = min(len(lon_edges_full) - 1, lon_idx1);
    lat_idx1 = min(len(lat_edges_full) - 1, lat_idx1)
    data_ctrl = mean_ctrl_full[lat_idx0:lat_idx1, lon_idx0:lon_idx1]
    data_sens = mean_sens_full[lat_idx0:lat_idx1, lon_idx0:lon_idx1]
    nx = data_ctrl.shape[1];
    ny = data_ctrl.shape[0]
    # Lon_e 和 Lat_e 现在将反映 lon_min, lon_max
    Lon_e = np.linspace(lon_min, lon_max, nx + 1)
    Lat_e = np.linspace(lat_min, lat_max, ny + 1)
    # 计算网格中心点 (C for Centers)
    Lon_c = (Lon_e[:-1] + Lon_e[1:]) / 2.0
    Lat_c = (Lat_e[:-1] + Lat_e[1:]) / 2.0
    Lon_C, Lat_C = np.meshgrid(Lon_c, Lat_c)
    # top vmin/vmax logic
    vals_top = np.concatenate([data_ctrl.ravel(), data_sens.ravel()])
    finite_top = vals_top[np.isfinite(vals_top)]
    if finite_top.size > 0:
        if TOP_CBAR_VMIN is None:
            if np.nanmin(finite_top) >= 0:
                vmin_top = 0.0
            else:
                vmin_top = float(np.nanmin(finite_top))
        else:
            vmin_top = float(TOP_CBAR_VMIN)
        if TOP_CBAR_VMAX is None:
            vmax_top = float(np.nanmax(finite_top))
        else:
            vmax_top = float(TOP_CBAR_VMAX)
    else:
        vmin_top = 0.0 if TOP_CBAR_VMIN is None else float(TOP_CBAR_VMIN)
        vmax_top = 1.0 if TOP_CBAR_VMAX is None else float(TOP_CBAR_VMAX)
    if vmin_top > vmax_top:
        vmin_top, vmax_top = vmax_top, vmin_top
    if np.isclose(vmin_top, vmax_top):
        vmin_top -= 1e-6;
        vmax_top += 1e-6
    # 为 contourf 创建离散的 boundaries 与 colormap
    boundaries_top = np.linspace(vmin_top, vmax_top, CB_N_BINS_TOP + 1)
    try:
        cmap_top_lin = LinearSegmentedColormap.from_list('white_warm_interp', PALETTE, N=CB_N_BINS_TOP)
        colors = cmap_top_lin(np.linspace(0, 1, CB_N_BINS_TOP))
        colors[0] = np.array([1.0, 1.0, 1.0, 1.0])
        cmap_top = ListedColormap(colors)
        cmap_top.set_bad('white')
    except Exception:
        base_colors = PALETTE[:]
        if base_colors[0].lower() != 'white':
            base_colors = ['white'] + base_colors
        cmap_top = ListedColormap(base_colors)
        cmap_top.set_bad('white')
    norm_top = BoundaryNorm(boundaries_top, ncolors=cmap_top.N, clip=False)
    # diff colormap boundaries & norm
    diff = data_sens - data_ctrl
    finite_diff = diff.ravel()[np.isfinite(diff.ravel())]
    if finite_diff.size > 0:
        auto_maxabs = float(np.nanmax(np.abs(finite_diff)))
    else:
        auto_maxabs = max(abs(vmin_top), abs(vmax_top), 1.0)
    if DIFF_CBAR_VMIN is None and DIFF_CBAR_VMAX is None:
        maxabs = auto_maxabs
        vmin_diff = -maxabs;
        vmax_diff = maxabs
    elif DIFF_CBAR_VMIN is None and DIFF_CBAR_VMAX is not None:
        vmax_diff = abs(float(DIFF_CBAR_VMAX));
        vmin_diff = -vmax_diff
    elif DIFF_CBAR_VMIN is not None and DIFF_CBAR_VMAX is None:
        vmin_diff = float(DIFF_CBAR_VMIN);
        vmax_diff = abs(vmin_diff);
        vmin_diff = -abs(vmin_diff)
    else:
        vmin_diff = float(DIFF_CBAR_VMIN);
        vmax_diff = float(DIFF_CBAR_VMAX)
    if vmin_diff > vmax_diff:
        vmin_diff, vmax_diff = vmax_diff, vmin_diff
    if np.isclose(vmin_diff, vmax_diff):
        vmin_diff -= 1e-6;
        vmax_diff += 1e-6
    boundaries_diff = np.linspace(vmin_diff, vmax_diff, CB_N_BINS + 1)
    try:
        cmap_diff_full = plt.get_cmap(CMAP_DIFF_NAME)
        diff_colors = cmap_diff_full(np.linspace(0, 1, CB_N_BINS))
        cmap_diff = ListedColormap(diff_colors)
        cmap_diff.set_bad('white')
    except Exception:
        cmap_diff = plt.get_cmap(CMAP_DIFF_NAME)
        try:
            cmap_diff.set_bad('white')
        except Exception:
            pass
    norm_diff = BoundaryNorm(boundaries_diff, ncolors=CB_N_BINS, clip=False)
    # figure & axes: 3x1, increase hspace
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    HSPACE = 0.47
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.0, 1.0], hspace=HSPACE)
    if HAS_CARTOPY:
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
        ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
        ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree(central_longitude=central_lon))
    else:
        ax1 = fig.add_subplot(gs[0, 0]);
        ax2 = fig.add_subplot(gs[1, 0]);
        ax3 = fig.add_subplot(gs[2, 0])
    # plot control (top) — 使用分级 contourf
    if HAS_CARTOPY:
        cf1 = ax1.contourf(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax1.contour(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try:
            ax1.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except Exception:
            pass
        ax1.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf1 = ax1.contourf(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', zorder=1)
        ax1.contour(Lon_C, Lat_C, data_ctrl, levels=boundaries_top, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax1.set_xlim(lon_min, lon_max);
        ax1.set_ylim(lat_min, lat_max)
    ax1.set_title("(a) CTL", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')

    # 如果传入了观测 IBTrACS 的 genesis 点，绘制在 control 上
    if obs_genesis_points is not None and len(obs_genesis_points) > 0:
        # obs_genesis_points: list of dict {'lat':..., 'lon':..., 'SID':..., 'month':...}
        obs_lons = np.array([p['lon'] for p in obs_genesis_points])
        obs_lats = np.array([p['lat'] for p in obs_genesis_points])
        # 归一化到 central_lon 的范围
        obs_lons_plot = normalize_lon(obs_lons, central_lon)
        try:
            ax1.scatter(obs_lons_plot, obs_lats, s=25, edgecolors=(0, 0, 0, 0.8), facecolors='black',
                        linewidths=0, alpha=0.6, transform=ccrs.PlateCarree(), zorder=6)
        except Exception:
            # fallback for non-cartopy axes
            ax1.scatter(obs_lons_plot, obs_lats, s=60, edgecolors='k', facecolors='black', linewidths=1.2, zorder=6)

    # plot sensitive (middle)
    if HAS_CARTOPY:
        cf2 = ax2.contourf(Lon_C, Lat_C, data_sens, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax2.contour(Lon_C, Lat_C, data_sens, levels=boundaries_top, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try:
            ax2.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except Exception:
            pass
        ax2.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf2 = ax2.contourf(Lon_C, Lat_C, data_sens, levels=boundaries_top, cmap=cmap_top, norm=norm_top,
                           extend='both', zorder=1)
        ax2.contour(Lon_C, Lat_C, data_sens, levels=boundaries_top, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax2.set_xlim(lon_min, lon_max);
        ax2.set_ylim(lat_min, lat_max)
    ax2.set_title("(b) GGW", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')

    # plot diff (bottom)
    if HAS_CARTOPY:
        cf3 = ax3.contourf(Lon_C, Lat_C, diff, levels=boundaries_diff, cmap=cmap_diff, norm=norm_diff,
                           extend='both', transform=ccrs.PlateCarree(), zorder=1)
        ax3.contour(Lon_C, Lat_C, diff, levels=boundaries_diff, colors='gray',
                    linewidths=0.5, transform=ccrs.PlateCarree(), zorder=1.5)
        try:
            ax3.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', zorder=2)
        except Exception:
            pass
        ax3.coastlines(resolution='50m', linewidth=1.5, color='black', zorder=3)
        ax3.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    else:
        cf3 = ax3.contourf(Lon_C, Lat_C, diff, levels=boundaries_diff, cmap=cmap_diff, norm=norm_diff,
                           extend='both', zorder=1)
        ax3.contour(Lon_C, Lat_C, diff, levels=boundaries_diff, colors='gray',
                    linewidths=0.5, zorder=1.5)
        ax3.set_xlim(lon_min, lon_max);
        ax3.set_ylim(lat_min, lat_max)
    ax3.set_title("(c) GGW-CTL", fontsize=TITLE_FONTSIZE, fontweight='bold', loc='left')
        # -------------------------------------------------------------------------
    # [新增] 在 Diff 子图上添加指定经纬度范围的黑框
    # Range: Lon [-50.0, -12.5], Lat [5.0, 20.0]
    # -------------------------------------------------------------------------

    box_w = box_lon_max - box_lon_min
    box_h = box_lat_max - box_lat_min

    if HAS_CARTOPY:
        # Cartopy 模式：使用 PlateCarree 变换
        rect_box = mpl.patches.Rectangle(
            (box_lon_min, box_lat_min), box_w, box_h,
            linewidth=2.5, edgecolor='black', facecolor='none',
            transform=ccrs.PlateCarree(), zorder=20
        )
        ax3.add_patch(rect_box)
    else:
        # 非 Cartopy 模式：直接使用坐标
        rect_box = mpl.patches.Rectangle(
            (box_lon_min, box_lat_min), box_w, box_h,
            linewidth=2.5, edgecolor='black', facecolor='none', zorder=20
        )
        ax3.add_patch(rect_box)
    # -------------------------------------------------------------------------
    # bold spines
    for ax in (ax1, ax2, ax3):
        try:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(axes_frame_linewidth)
                spine.set_edgecolor('black')
        except Exception:
            pass

    # ticks: xticks every 30°, yticks: majors every 20°, minors every 10°
    step = 30.0
    xticks_raw = np.arange(int(np.ceil(lon_min / step)) * int(step),
                           int(np.floor(lon_max / step)) * int(step) + 1, int(step)).astype(float)
    if xticks_raw.size == 0:
        xticks_raw = np.array([lon_min, lon_max], dtype=float)
    xticks_plot = normalize_lon_to_center(xticks_raw, central_lon)
    order = np.argsort(xticks_plot)
    xticks_plot_sorted = xticks_plot[order]
    xticks_raw_sorted = xticks_raw[order]

    # y majors / minors
    y_major0 = int(np.ceil(lat_min / Y_MAJOR_TICK_STEP) * Y_MAJOR_TICK_STEP)
    y_major1 = int(np.floor(lat_max / Y_MAJOR_TICK_STEP) * Y_MAJOR_TICK_STEP)
    if y_major0 > y_major1:
        y_major = np.array([lat_min, lat_max])
    else:
        y_major = np.arange(y_major0, y_major1 + 1, Y_MAJOR_TICK_STEP)
    y_minor0 = int(np.ceil(lat_min / Y_MINOR_TICK_STEP) * Y_MINOR_TICK_STEP)
    y_minor1 = int(np.floor(lat_max / Y_MINOR_TICK_STEP) * Y_MINOR_TICK_STEP)
    if y_minor0 > y_minor1:
        y_minor = np.array([])
    else:
        y_minor = np.arange(y_minor0, y_minor1 + 1, Y_MINOR_TICK_STEP)
    if (lat_min <= 0.0 <= lat_max) and (0 not in y_major):
        y_major = np.unique(np.sort(np.concatenate((y_major, [0]))))
        y_major = np.sort(y_major)

    # 为所有子图统一显示与 diff 相同的经纬标签（用户要求）
    # 生成 xlabels_display（每隔2个刻度显示一个标签，若需全部显示可改为 i%1）
    xlabels_display = []
    for i, x_raw in enumerate(xticks_raw_sorted):
        if i % 2 == 0:
            xlabels_display.append(format_lon_label(x_raw))
        else:
            xlabels_display.append('')

    for ax in (ax1, ax2, ax3):
        try:
            ax.set_xticks(xticks_plot_sorted, crs=ccrs.PlateCarree())
            ax.set_yticks(y_major, crs=ccrs.PlateCarree())
            ax.set_yticks(y_minor, minor=True, crs=ccrs.PlateCarree())
        except Exception:
            ax.set_xticks(xticks_plot_sorted)
            ax.set_yticks(y_major)
            ax.set_yticks(y_minor, minor=True)

        # Y major labels formatted with N/S
        ylabels_major = [format_lat_label(y) for y in y_major]
        ax.set_yticklabels(ylabels_major, fontsize=OTHER_FONTSIZE - 6, family='Times New Roman')
        ax.tick_params(axis='y', which='major', labelleft=True, labelsize=OTHER_FONTSIZE - 6,
                       direction='out', length=10, width=axes_frame_linewidth)
        ax.tick_params(axis='y', which='minor', labelleft=False, direction='out', length=5,
                       width=axes_frame_linewidth * 0.5)

        # 对所有子图都显示 x 标签（和 diff 一致）
        try:
            ax.set_xticklabels(xlabels_display, fontsize=OTHER_FONTSIZE - 6, family='Times New Roman')
        except Exception:
            # cartopy 有时需要先 draw() 再设置 xticklabels；如果异常则忽略
            pass
        ax.tick_params(axis='x', labelbottom=True, which='major', direction='out',
                       length=10, width=axes_frame_linewidth, pad=10)

    # 次刻度为相邻主刻度的中点（基于原始 xticks_raw_sorted）
    if len(xticks_raw_sorted) > 1:
        xtick_minors_raw = (xticks_raw_sorted[:-1] + xticks_raw_sorted[1:]) / 2.0
        xtick_minors_plot = normalize_lon_to_center(xtick_minors_raw, central_lon)
        for ax in (ax1, ax2, ax3):
            try:
                ax.set_xticks(xtick_minors_plot, minor=True, crs=ccrs.PlateCarree())
            except Exception:
                ax.set_xticks(xtick_minors_plot, minor=True)
            ax.tick_params(axis='x', which='minor', labelbottom=False, direction='out', length=5)

    # dashed grids (no labels)
    for ax in (ax1, ax2, ax3):
        if HAS_CARTOPY:
            gl = ax.gridlines(draw_labels=False, linestyle='--', linewidth=0.4, color='gray', alpha=0.6)
            try:
                gl.xlocator = mticker.FixedLocator(xticks_plot_sorted)
                gl.ylocator = mticker.FixedLocator(y_major)
            except Exception:
                pass
        else:
            ax.grid(which='major', linestyle='--', linewidth=0.4, color='gray', alpha=0.6)

    # colorbars: shared for top two (centered & slightly shorter), separate for diff
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
    min_y = 0.005
    max_y = 0.995 - cbar1_h
    if cbar1_y < min_y:
        cbar1_y = min_y
    if cbar1_y > max_y:
        cbar1_y = max_y
    cbar1_x = axes_right + CB_GAP
    if cbar1_x + CB_WIDTH > 0.98:
        cbar1_x = axes_right - CB_GAP - CB_WIDTH
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
            if np.isclose(v, 0):
                labels.append("0")
            else:
                labels.append(f"{v:.2f}")
        cb1.set_ticklabels(labels)
    except Exception:
        pass

    # -------------------- 修改点：使 diff colorbar 垂直居中并缩短 --------------------
    # 现在使用 DIFF_CBAR_SCALE（0..1）基于 diff 子图高度来计算 colorbar 的高度，
    # 并使其垂直居中（可再通过 CB2_Y_SHIFT 做微调）。
    sub_height = pos3.y1 - pos3.y0
    cbar2_h = max(0.001, sub_height * float(DIFF_CBAR_SCALE))
    # 垂直居中：子图底部 + (子图高度 - colorbar高度)/2
    cbar2_y = pos3.y0 + (sub_height - cbar2_h) / 2.0
    # 应用额外的垂直偏移（正向上）
    cbar2_y += CB2_Y_SHIFT
    # 保证不超出边界
    if cbar2_y < min_y:
        cbar2_y = min_y
    if cbar2_y + cbar2_h > 0.995:
        cbar2_y = 0.995 - cbar2_h
    # 水平位置（与上面的逻辑一致）
    cbar2_x = axes_right + CB_GAP
    if cbar2_x + CB_WIDTH > 0.98:
        cbar2_x = axes_right - CB_GAP - CB_WIDTH
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
        if np.isclose(v, 0):
            labels.append("0")
        else:
            labels.append(f"{float(v):.2f}")
    cb2.set_ticklabels(labels)
    # ---------------------------------------------------------------------------

    # finalize layout and show/save
    plt.subplots_adjust(top=0.95, bottom=BOTTOM_MARGIN, left=0.08, right=0.88)
    try:
        fig.savefig(out_png, dpi=FIG_DPI, bbox_inches='tight')
        print(f"Saved figure to {out_png}")
    except Exception:
        pass
    plt.show()
    print("Displaying composite figure")

# main
def main():
    """Read the extracted plot inputs; keep the original plotting call below."""
    if not os.path.isfile(PROCESSED_DATA_FILE):
        raise FileNotFoundError(f"Processed data not found: {PROCESSED_DATA_FILE}")
    print(f"Loading processed plot data: {PROCESSED_DATA_FILE}")
    with nc.Dataset(PROCESSED_DATA_FILE, 'r') as ds:
        if not np.isclose(float(ds.grid_resolution_degrees), GRID_RES):
            raise ValueError('GRID_RES differs from the saved grid; rerun extract_plot_data.py.')
        if not np.isclose(float(ds.central_longitude_degrees), CENTRAL_LON):
            raise ValueError('CENTRAL_LON differs from the saved grid; rerun extract_plot_data.py.')
        mean_ctrl = np.array(ds.variables['mean_ctrl'][:])
        mean_sens = np.array(ds.variables['mean_sens'][:])
        lon_edges = np.array(ds.variables['lon_edges'][:])
        lat_edges = np.array(ds.variables['lat_edges'][:])
        obs_lats = np.array(ds.variables['obs_lat'][:])
        obs_lons = np.array(ds.variables['obs_lon'][:])
        obs_months = np.array(ds.variables['obs_month'][:])
        obs_sids = list(ds.variables['obs_sid'][:])
        obs_genesis_points = [
            {'lat': float(lat), 'lon': float(lon), 'SID': sid, 'month': int(month)}
            for lat, lon, sid, month in zip(obs_lats, obs_lons, obs_sids, obs_months)
        ]
        for label, prefix in [('Control (CTL)', 'control'), ('Sensitive (GGW)', 'sensitive')]:
            total = int(ds.getncattr(prefix + '_jas_total'))
            years = int(ds.getncattr(prefix + '_n_sim_years'))
            print(f"{label} JAS Total: {total}, Years: {years}, Annual Avg: {total / years:.2f}")
        print(f"IBTrACS JAS points: {len(obs_genesis_points)}, Years: {int(ds.obs_n_years)}")

    plot_jas_control_sensitive_diff(mean_ctrl, mean_sens,
                                   LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
                                   lon_edges, lat_edges,
                                   out_fig,
                                   central_lon=CENTRAL_LON,
                                   grid_res=GRID_RES,
                                   obs_genesis_points=obs_genesis_points)

if __name__ == "__main__":
    main()
