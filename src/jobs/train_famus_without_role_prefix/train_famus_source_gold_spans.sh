#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --time=24:0:0
#SBATCH --gpus=1

eval "$(conda shell.bash hook)"
conda activate iterx

cd /brtx/601-nvme1/svashis3/iterx

PYTHONPATH=./src allennlp train \
  --include-package iterx \
  -s famus_model_source_data_gold_spans \
  /brtx/601-nvme1/svashis3/iterx/resources/training_configs/famus_source_config_gold_spans.jsonnet