# Creating an `asf_tools` prototype environment in OpenSARlab

<https://opensarlab.asf.alaska.edu/>

## Initial (one-time) setup

Once you're logged into OpenSARlab, you'll need to do a little bit of setup to make
OpenSARlab ready for development.

*Note: The steps only need to be done **once** and many of these steps will be
incorporated into the next OpenSARlab deployment.*


###  Set up conda on OpenSARlab

First, open a new terminal:

![new_terminal](https://user-images.githubusercontent.com/7882693/108315967-ab55da80-7168-11eb-9fab-d66e01b52611.png)

Then, setup conda on OpenSARlab by creating a `${HOME}/.condarc` file with these contents:

```
channels:
  - conda-forge
  - defaults

channel_priority: strict

create_default_packages:
  - jupyter
  - kernda

envs_dirs:
  - /home/jovyan/.local/envs
  - /opt/conda/envs
```

and create a `${HOME}/.bash_profile` with these contents:
```
if [ -s ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc;
fi
```

and initialize conda in your shell
```bash
conda init
source ${HOME}/.bashrc
```

### Get `asf_tools` and setup a development environment

```bash
git clone https://github.com/ASFHyP3/asf-tools.git
cd asf-tools

conda env create -f prototype/osl-env.yml
conda activate asf-tools

# install the kernel so it'll be available in notebooks
python -m ipykernel install --user --name asf-tools
kernda ${HOME}/.local/share/jupyter/kernels/asf-tools/kernel.json --env-dir ${CONDA_PREFIX} -o

# install an editable/develop version of the `asf_tools` python package
python -m pip install -e .

# test `asf_tools` is installed:
python -c 'import asf_tools; print(asf_tools.__version__)'
```

### Finalize setup

[Restart your server to make the `asf-tools` kernel available in notebooks](https://github.com/asfadmin/asf-jupyter-docs/blob/master/user_docs/guides/restarting_server_and_kernel.md)

Now you can:
* Create a new notebook using the `asf-tools` kernel
  ![new_notebook_kernel](https://user-images.githubusercontent.com/7882693/116321835-538aaf80-a767-11eb-8d09-26b06ca96202.png)

* Change the kernel of an existing notebook to the `asf-tools` kernel
  ![change_kernel](https://user-images.githubusercontent.com/7882693/116321985-9c426880-a767-11eb-91f1-a36d2b39a678.png)

### Keeping in sync with changes upstream

In the future, as dependencies are added/removed, you can update the conda environment with
```bash
conda env update -f prototype/osl-env.yml
```
but you should *not* need to reinstall the kernel or restart OpenSARlab for the
environment updates to take effect.

As `asf_tools` itself is developed, you may need to reinstall the editable version
if the package structure has significantly changed, or entrypoints have been added/removed.
```bash
python -m pip install -e .
```

In general, any time you pull changes it's a good idea to run both of the above commands.

## Run the example `water-extent-map.ipynb`

Once you've completed the above one-time setup and restarted your OSL server, you
can run through the example `water-extent-map.ipynb` notebook, which will be located
at `${home}/asf-tools/prototype/`

![water-extent-map.ipynb](https://user-images.githubusercontent.com/7882693/116322104-d3b11500-a767-11eb-81cd-f7083ca0f42c.png)

*Note: Make sure you change this notebook's kernel to the `asf-tools` kernel!*
