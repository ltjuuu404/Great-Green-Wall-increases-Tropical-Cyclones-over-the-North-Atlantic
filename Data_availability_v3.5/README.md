# Data availability v3.5

Figure source data and plotting scripts for Figures 1–11 and S1–S8 in the 17 November 2026 manuscript and supplement. The [figure manifest](manifest.csv) lists each plotting script and input filename.

Each figure has its own folder. Its processed inputs are named by figure and are resolved from the script location. Run the plotting scripts to generate figures locally.

Install the dependencies in [requirements.txt](requirements.txt), then run the relevant script, for example `python figure8/figure8.py`. Figures with an interactive plotting script can be saved from the plot window. Keep [map_data](map_data/) beside the figure folders for offline map rendering.

Figure S4 uses 850/200 hPa wind panels. Figure S7 panel (b) shows basin-wide AEW frequency. Figure S8 contains the dust aerosol optical depth response.
