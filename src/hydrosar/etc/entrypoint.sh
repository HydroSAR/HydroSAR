#!/bin/bash --login
set -e
conda activate hydrosar
exec python -um hydrosar "$@"
