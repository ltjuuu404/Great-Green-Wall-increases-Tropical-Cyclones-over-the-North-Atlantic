# Data availability v3.5

Figure source data and plotting scripts for Figures 1–11 and S1–S8 in the 17 November 2026 manuscript and supplement. The [figure manifest](manifest.csv) lists each plotting script, archived image and input filename.

Each figure has its own folder. Its processed inputs are named by figure and are resolved from the script location. The `outputs/` subfolder holds the archived artwork used for the version check. Some archived panels were cropped or assembled after plotting, so a fresh render may differ in its margins or panel spacing.

Install the dependencies in [requirements.txt](requirements.txt), then run the relevant script, for example `python figure8/figure8.py`. Figures with an interactive plotting script can be saved from the plot window. Keep [map_data](map_data/) beside the figure folders for offline map rendering.

Figure S4 uses the 850/200 hPa wind panels shown in the archived artwork. Figure S7 panel (b) shows basin-wide AEW frequency. Figure S8 contains the dust aerosol optical depth response.
