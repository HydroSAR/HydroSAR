# HydroSAR
## About the HydroSAR Project
HydroSAR is a project funded by the NASA Applied Sciences Program focused on the development of algorithms for the monitoring of hydrological hazards using data from Synthetic Aperture Radar (SAR) sensors. Algorithms are being developed for the following SAR-derived products:
- **RTC30:** Time series of Radiometrically Terrain Corrected (RTC) SAR images provided at 30 m resolution.
- **RTC30-Color:** RGB compositions of from dual-pol (VV/VH) Sentinel-1 SAR RTC image products provided at 30 m resolution.
- **CCD30:** Change detection products relative to the first image of an event subscription (30 m resolution).
- **HYDRO30:** Surface water extent maps per Sentinel-1 SAR image acquisition (30 m resolution).
- **AG100:** Agriculture extent maps derived by statistical analyses of RTC30 time series stacks (provided at 1 ha resolution).
- **AG30-IN:** Maps of inundated agriculture (1 ha resolution).

## The HydroSAR Team
HydroSAR includes an interdisciplinary team from the following universities and NASA Centers:
- University of Alaska Fairbanks (UAF; project lead); Fairbanks, AK
- NASA Alaska Satellite Facility (ASF) Distributed Active Archive Center; Fairbanks, AK
- NASA Marshall Space Flight Center (MSFC); Huntsville, AL
- University of Alabama, Huntsville (UAH); Huntsville, AL
- NASA Goddard Space Flight Center (GSFC); Greenbelt, MD
- Jet Propulsion Laboratory (JPL); Pasadena, CA

## HydroSAR Python Package

[`hydrosar`](src/hydrosar) is a Python package developed for generating HydroSAR products, both locally and in [ASF's HyP3](https://hyp3-docs.asf.alaska.edu/).

`hydrosar` is distributed on [PyPI](https://pypi.org/) and [conda-forge](https://conda-forge.org/docs/user/introduction.html). See the `hydrosar` [README](src/hydrosar/README.md) for installation and usage.


## FIERpy Pipeline Instructions

### Order RTC jobs and launch your server
The pipeline was made to work directly with ASF’s Vertex RTC images. 
- Order jobs on [Vertex](https://search.asf.alaska.edu/#/) (wait maybe an hour before they have been processed). **MAKE SURE YOU ORDER THE DEMS WITH YOUR RTC IMAGES**
- Start your [OpenScienceLab](http://opensciencelab.asf.alaska.edu/) server to access high RAM performance and the appropriate python environments (you can also directly download the notebooks on [Github](https://github.com/fjmeyer/HydroSAR/tree/Workflow_Forecast) but be aware that FIERpy needs A LOT of RAM (often more than 40 Gb).

### Download the RTCs and precipitation data.
- Launch Notebook 1. It might take some time to download all the RTC images. You can delete the zips but **do not delete the main RTCs once unzipped**.
- Launch Notebook 1bis. This might take a few hours, depending on the queue of [Copernicus](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) servers. *Once launched, let the notebook opened until you have a file in the ERA folder.* **You can run Notebooks 2 to 4 while waiting for Notebook 1bis to run**

### Choose your AOI, generate its HAND DEM and Water Masks
- Launch Notebook 2. Crop the RTCs to your AOI. Trial-and-error might have to happen in order to find an AOI that does not blow-up your RAM during the REOF (Notebook 5).
- Launch Notebook 3. It should be fast (5-10min).
- Launch Notebook 4. If you activate the post-processing, it might take a few hours. If not, it might take up to 30min if you have around 90 RTC images.

### Compute the forecast
**Only after you have downloaded [ERA5 precipitation data](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) and computed HAND DEM with Water Masks**
- Launch Notebook 5. It can take a few hours to compute. Follow the instructions, make sure the indices you select for the forecast time-window are 12 days total. An error message should pop-up if it's not the case.
*Remember: this forecast does not actually forecast past your present day. GEOGLOWS (discharge input) is generally stops 2 months before your current date, and 1 month for ERA5.*

### Workflow
<img src="docs/HydroSAR Workflow.jpg" align="right" width="1500" />
