![HydroSAR Banner](https://radar.community.uaf.edu/wp-content/uploads/sites/667/2021/03/HydroSARbanner.jpg)

# HydroSAR Transition Workshop Notebooks

This is a collection of Notebooks to facilitate the HydroSAR Data Products and Web Application training.
Through the notebooks, users will understand the current HydroSAR service as implemented by the Alaska Satellite Facility (ASF), 
as well as how to run/use each component. Detailing this process will allow for a successful transition
of ownership from ASF/HydroSAR to the SERVIR-HKH science team. By the end of the 2022 ICIMOD Transition workshop, 
ICIMOD participants will have sufficient understanding of the various ways the HydroSAR service can be run and maintained.
For more information, please see:
* [The Transition Plan](https://drive.google.com/file/d/1VJnj6vjKvi_D556zBXLGb7-GdMNrdK9j/view?usp=sharing)
* [The 2022 ICIMOD Transition Workshop agenda](https://drive.google.com/file/d/1Up6ZUcN8JiDxASIaELw3jR6749tZqudp/view?usp=sharing)

## Overview of HydroSAR Service
![HydroSAR Service Diagram](https://docs.google.com/drawings/d/e/2PACX-1vS8Pg7F6qXVRez6rAbUI97eetkBHUjPUatLRfj0AtUemKVYf9XxyZ2twV3HMBbvk_vjCI7l0GU4RcSc/pub?w=905&h=432)

The HydroSAR service consists of 4 major components, which are detailed in their associated notebooks:
1. Monitoring an Area of Interest (AOI) for new Sentinel-1 acquisitions: [HydroSAR_service_monitoring.ipynb](./HydroSAR_service_monitoring.ipynb)
2. Processing new Sentinel-1 Level-1 SLC products into HydroSAR products [HydroSAR_service_creating_water_products.ipynb](./HydroSAR_service_creating_water_products.ipynb)
3. Archiving new HydroSAR products: [HydroSAR_service_archiving.ipynb](./HydroSAR_service_archiving.ipynb)
4. Updating ESRI Image Services with any newly archived products

### Earth Data Cloud Account

Before working with these notebooks, users must have [an Earth Data Cloud (EDC) account](https://urs.earthdata.nasa.gov/users/new?client_id=BO_n7nTIlMljdvU6kRRB3g&redirect_uri=https%3A%2F%2Fauth.asf.alaska.edu%2Flogin&response_type=code&state=https%3A%2F%2Fsearch.asf.alaska.edu)
to submit SAR products for processing with HyP3 via the [ASF Search portal](https://search.asf.alaska.edu/),
the [HyP3 SDK](https://hyp3-docs.asf.alaska.edu/using/sdk/), or the [HyP3 API](https://hyp3-api.asf.alaska.edu/ui/).


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
