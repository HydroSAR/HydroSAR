# Start Here

HydroSAR is a project funded by the NASA Applied Sciences Program focused on the development of algorithms for the monitoring of hydrological hazards using data from Synthetic Aperture Radar (SAR) sensors. This Jupyter Book demonstrates how to create HydroSAR products. 

This Jupyter Book currently supports:
- **HYDRO30:** Surface water extent maps per Sentinel-1 SAR image acquisition (30 m resolution)

See the [HydroSAR README](../../../README.md) for a complete list of algorithms, still under development.


<div class="alert alert-success" style="display: flex; align-items: center; font-family: 'Times New Roman', Times, serif; background-color: 'rgba(200,0,0,0.2)'">
  <div style="width: 95%;">
    <h2><b>Important Note About Your Jupyter Environment if not in OpenSARLab</b></h2>
    <b><i>Tip: Run the notebooks in this Jupyter Book from Jupyter Lab, launched from a conda environment containing the required Jupyter packages. 
        <br/>
        This can be accomplished with the following commands:</i></b>
    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px; border: 1px solid #ccc; overflow: auto;">
      <code>mamba create -n jbook -c conda-forge jupyterlab notebook ipywidgets ipympl nb_conda_kernels</code>
      <code>mamba activate jbook</code>
      <code>python -m pip install jupyterlab-jupyterbook-navigation</code>
      <code>jupyter lab</code>
    </pre>
    <ul>
        <li>Jupyter selection widgets used in the notebooks require the packages <code>ipywidgets</code> and <code>notebook</code>.</li>
        <li>In order to use multiple conda environments in Jupyter Lab at the same time, you must install <code>nb_conda_kernels</code>.</li>
        <li>Interactive matplotlib plots requires the package <code>ipympl</code> in the environment running Jupyter Lab</li>
    </ul>
  </div>
</div>

## How To Use This Jupyter Book

>1. ### Install the software environments needed to run the notebooks
>
>    - [Install Required Software with Conda](Software_Environments.ipynb)
>    - Rerun this step periodically to pull in environment updates.
>
>1. ### Access and Prepare RTCs for HydroSAR
>
>    - Not necessary if ordering water maps directly from ASF HyP3
>    - [Download RTCs from ASF HyP3](Prepare_HydroSAR_RTC_Stack.ipynb)
>  
>1. ### Subset Data (Optional)
>
>    - [Subset RTCs](Subset_HydroSAR_Stack.ipynb)
>  
>1. ### Create Water Maps
>
>    - 
