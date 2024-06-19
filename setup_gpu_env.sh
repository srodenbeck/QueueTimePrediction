#!/bin/bash
module purge
module load modtree/gpu
module load learning/conda-2021.05-py38-gpu
module load ml-toolkit-gpu/pytorch
module use /anvil/projects/x-cda090008/etc/modules
module load conda-env/gpupkg-py3.8.8

pip install -U "neptune[optuna]"
pip install absl-py
pip install imbalanced-learn
pip install psycopg2-binary
