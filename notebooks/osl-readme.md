# Creating an `asf_tools` prototype environment in OpenSARlab

<https://opensarlab.asf.alaska.edu/>

**NOTE:** OpenSARlab was [upgraded on June 10, 2021](https://github.com/ASFOpenSARlab/opensarlab-docs/blob/master/OpenSARlab_release_notes/OpenSARlab_Release_Notes_June_2021.ipynb)
with much better support for conda environments. This significantly simplifies setting
up development environments and these instructions have changed accordingly.

## Initial (one-time) setup of an `asf_tools` development environment

```bash
git clone https://github.com/ASFHyP3/asf-tools.git
cd asf-tools

conda env create -f prototype/osl-env.yml
conda activate asf-tools

# install an editable/develop version of the `asf_tools` python package
python -m pip install -e .

# test `asf_tools` is installed:
python -c 'import asf_tools; print(asf_tools.__version__)'
```

Refresh your browser and the new environment should be available!

Now you can:
* Create a new notebook using the `asf-tools` kernel
  
  ![new_notebook_kernel](https://user-images.githubusercontent.com/7882693/121728495-ce9af180-ca99-11eb-84eb-8114b7ce1183.png)

* Change the kernel of an existing notebook to the `asf-tools` kernel
  
  ![change_kernel](https://user-images.githubusercontent.com/7882693/121728676-0efa6f80-ca9a-11eb-959d-656c9376a8d7.png)

### Keeping in sync with changes upstream

*Note:* These instructions assume you're working from the head/root of the asf-tools repository.

In the future, as dependencies are added or removed, you can update the conda environment with
```bash
conda env update -f prototype/osl-env.yml
```
and you should *not* need to restart OpenSARlab for the environment updates to take effect.

As `asf_tools` itself is developed, you may need to reinstall the editable version
if the package structure has significantly changed, or entrypoints have been added/removed.
```bash
python -m pip install -e .
```

In general, any time you pull changes it's a good idea to run both of the above commands.

## Run the example `water-extent-map.ipynb`

Once you've completed the above one-time setup, you can run through the example notebook,
which will be located in the `prototype/` directory:
* `water-extent-map.ipynb` -- Run the water map code directly. This allows you to
  edit the code and see the changes reflected.
* `water-extent-map-on-demand.ipynb` -- Request water map products from the custom HyP3-watermap deployment
   using either ASF Search Vertex or the `asf_search` Python package, and the HyP3 SDK.

  ![prototype notebooks](https://user-images.githubusercontent.com/7882693/122486645-f7cded00-cf85-11eb-8c94-2ddd63961059.png)


*Note: Make sure you change the notebook kernel for each of these notebooks to the `asf-tools` kernel!*
