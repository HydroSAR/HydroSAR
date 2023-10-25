FROM condaforge/mambaforge:latest

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="HydroSAR"
LABEL org.opencontainers.image.description="HyP3 Plugin for monitoring of hydrological hazards"
LABEL org.opencontainers.image.vendor="Alaska Satellite Facility"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://github.com/fjmeyer/HydroSAR"
LABEL org.opencontainers.image.source="https://github.com/fjmeyer/HydroSAR"
LABEL org.opencontainers.image.documentation="https://github.com/fjmeyer/HydroSAR"

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG CONDA_UID=1000
ARG CONDA_GID=1000

RUN groupadd -g "${CONDA_GID}" --system conda && \
    useradd -l -u "${CONDA_UID}" -g "${CONDA_GID}" --system -d /home/conda -m  -s /bin/bash conda && \
    chown -R conda:conda /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/conda/.profile && \
    echo "conda activate base" >> /home/conda/.profile

USER ${CONDA_UID}
SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /home/conda/

COPY --chown=${CONDA_UID}:${CONDA_GID} . /hydrosar/

RUN mamba env create -f /hydrosar/environment.yml && \
    conda clean -afy && \
    conda activate hydrosar && \
    sed -i 's/conda activate base/conda activate hydrosar/g' /home/conda/.profile && \
    python -m pip install --no-cache-dir /hydrosar

# attempt to speed up initial hydrosar imports by caching *something*...
RUN python -m hydrosar --help

ENTRYPOINT ["/hydrosar/src/hydrosar/etc/entrypoint.sh"]
CMD ["-h"]
