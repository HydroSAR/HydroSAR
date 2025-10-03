# Creating a `hydrosar` prototype environment in OpenScienceLab

<https://opensciencelab.asf.alaska.edu/>

## Initial (one-time) setup of an `hydrosar` development environment

When you log into your server on OpenScienceLab (OSL), launch a terminal and run these commands to set up a development environment:

```bash
git clone https://github.com/HydroSAR/HydroSAR
cd HydroSAR

conda env create -f environment.yml
conda activate hydrosar

# install an editable/develop version of the `hydrosar` python package
python -m pip install -e .

# test `hydrosar` is installed:
python -c 'import hydrosar; print(hydrosar.__version__)'
```

To run the on demand notebook and make this environment discoverable in OpenSienceLab, you'll need to also install some additional packages:
```bash
conda install --file docs/hydrosar-on-demand-requirements.txt
```

Refresh your browser and the new environment should be available!

Now you can:
* Create a new notebook using the `hydrosar` kernel
  
  ![new_notebook_kernel](https://github.com/HydroSAR/HydroSAR/assets/7882693/b0395e7d-d056-4e20-bb5a-5154adfbe61c)

* Change the kernel of an existing notebook to the `hydrosar` kernel
  
  ![change_kernel](https://github.com/HydroSAR/HydroSAR/assets/7882693/e008a252-80a5-480e-b8af-9d057b79c138)


### Keeping in sync with changes upstream

*Note:* These instructions assume you're working from the head/root of the HydroSAR repository.

In the future, as dependencies are added or removed, you can update the conda environment with
```bash
conda env update -f environment.yml
```
and you should *not* need to restart OpenSARlab for the environment updates to take effect.

As `hydrosar` itself is developed, you may need to reinstall the editable version
if the package structure has significantly changed, or entrypoints have been added/removed.
```bash
python -m pip install -e .
```

In general, any time you pull changes it's a good idea to run both of the above commands.

## Run the example `water-extent-map.ipynb`

Once you've completed the above one-time setup, you can run through the example notebook,
which will be located in the `notebooks/` directory:
* `water-extent-map.ipynb` -- Run the water map code directly. This allows you to
  edit the code and see the changes reflected.
* `water-extent-map-on-demand.ipynb` -- Request water map products from the custom HyP3-watermap deployment
   using either ASF Search Vertex or the `asf_search` Python package, and the HyP3 SDK.

  ![prototype notebooks](https://github.com/HydroSAR/HydroSAR/assets/7882693/58ffb315-59b2-4dbf-b100-457b78a0a36e)


*Note: Make sure you change the notebook kernel for each of these notebooks to the `hydrosar` kernel!*
