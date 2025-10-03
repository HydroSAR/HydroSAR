FROM pangeo/base-notebook:latest AS release

USER root
WORKDIR /

RUN set -ve

RUN python -m pip install --no-cache-dir \
    jupyterlab-jupyterbook-navigation==1.0.2 \
    opensarlab-frontend==1.5.1 && \
wget https://raw.githubusercontent.com/HydroSAR/HydroSAR/feature/reorg_and_jupyter-book/notebooks/jupyter_book/jupyter_book_environment/hydrosar.yml -P /etc/env/ && \
mamba env create -f /etc/env/hydrosar.yml && \
mamba install -c conda-forge -n notebook nb_conda_kernels && \
mamba clean --yes --all && \
pip cache purge

RUN mkdir -p /opt/jupyter-settings/@jupyterlab/docmanager-extension && \
    printf '{ "defaultViewers": { "markdown": "Markdown Preview" } }' \
    > /opt/jupyter-settings/@jupyterlab/docmanager-extension/plugin.jupyterlab-settings && \
    chmod a+rx /opt/jupyter-settings

WORKDIR /home/jovyan
USER jovyan

CMD [ \
    "sh", "-c", \
    "mkdir -p /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/docmanager-extension && \
     cp -r /opt/jupyter-settings/@jupyterlab/docmanager-extension/* /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/docmanager-extension/ && \
     exec jupyter lab --ip=0.0.0.0 --port=8888" \
]
