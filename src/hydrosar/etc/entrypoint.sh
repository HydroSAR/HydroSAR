#!/bin/bash --login
set -e
conda activate asf-tools
exec python -um asf_tools "$@"
