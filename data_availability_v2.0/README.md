# Data availability package v2.0

This package contains the processed NetCDF inputs and standalone Python plotting scripts for all figures in the revised manuscript and its Supplementary Material. The folder numbering follows the final figure order in `GGW-NATC_0728.docx` and `supplements.docx`.

## Contents

| Folder | Figure content | Script | NetCDF input(s) |
|---|---|---|---|
| `figure1` | Experimental design and prescribed vegetation modification | `figure1.py` | `figure1_plot_data.nc` |
| `figure2` | SST, sea-level pressure/circulation, and precipitation | `figure2.py` | `sst_ctrl.nc`; `sst_diff.nc`; `slp_ctrl.nc`; `slp_diff.nc`; `precip_ctrl.nc`; `precip_diff.nc` |
| `figure3` | Large-scale circulation, ascent, midlevel humidity, and vertical wind shear | `figure3.py` | `row1_850_circulation.nc`; `row2_500_omega.nc`; `row3_700_rh.nc`; `row4_200_shear.nc` |
| `figure4` | Meridional atmospheric circulation cross-section | `figure4.py` | `figure6.nc` |
| `figure5` | Surface albedo and surface-energy flux responses | `figure5.py` | `albedo_ctl.nc`; `albedo_diff.nc`; `fsns_diff.nc`; `latent_heat_diff.nc`; `sensible_heat_diff.nc` |
| `figure6` | Moist-static-energy cross-section | `figure6.py` | `JAS_MSE_smoothed_data.nc` |
| `figure7` | Surface-wind, latent-heat-flux, and cloud responses | `figure7.py` | `wind_1000_stats.nc`; `lhflx_stats.nc`; `cldtot_stats.nc` |
| `figure8` | Observed and simulated tropical-cyclone genesis frequency | `figure8.py` | `figure2_data.nc` |
| `figure9` | Tropical-cyclone genesis indices and factor decomposition | `figure9.py` | `figure3_data_ocean_masked_new.nc` |
| `figure10` | Tropical-cyclone track-density response | `figure10.py` | `figure7_data.nc` |
| `figure11` | Pressure-weighted tropical-cyclone steering flow | `figure11.py` | `figure8_data.nc` |
| `figure12` | 850-, 500-, and 200-hPa circulation responses | `figure12.py` | `figure12_wind_data.nc` |
| `figureS1` | GPCP precipitation validation | `figureS1.py` | `figureS1_data.nc` |
| `figureS2` | Coupled SST, current, advection, and Sahel-rainfall mechanism | `figureS2.py` | `figureS2_data.nc` |
| `figureS3` | Prescribed-SST versus coupled precipitation and circulation responses | `figureS3.py` | `figureS3_data.nc` |
| `figureS4` | African easterly-wave seed response | `figureS4.py` | `figureS4_data.nc` |
| `figureS5` | Dust, radiation, surface-temperature, and TC-genesis responses | `figureS5.py` | `figureS5_data.nc` |
| `figureS6` | Tropical-cyclone intensity validation | `figureS6.py` | `figureS6_data.nc` |

## Reproduction

Each script reads only the NetCDF file or files in its own folder. Paths are resolved relative to the script location, so the scripts can be launched from any working directory.

Example:

```bash
python figure8/figure8.py
python figureS2/figureS2.py
```

The original v1.4 display/export behavior is retained for every main figure. Scripts that originally used `plt.show()` still open an interactive figure window for manual saving. Scripts that originally saved files retain the same export mechanism, with only the manuscript figure identifier updated.

## Software

The scripts were verified with Python 3 and the packages listed in `requirements.txt`. Cartopy requires access to its Natural Earth coastline and boundary datasets. All NetCDF files contain processed figure-ready data; the original raw model and observational archives are not required for reproducing these figures.
