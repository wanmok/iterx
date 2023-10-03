#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --time=24:0:0
#SBATCH --gpus=1
#SBATCH --output=/brtx/601-nvme1/svashis3/iterx/src/jobs/evaluate_famus.log

eval "$(conda shell.bash hook)"
conda activate iterx

cd /brtx/601-nvme1/svashis3/iterx

PYTHONPATH=./src allennlp predict \
            /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_debugging_slot_names \
             resources/data/famus/iterx_format/report_data/train.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.muc.doc_path": {"resources/data/famus/iterx_format/report_data/train.jsonl": "resources/data/famus/iterx_format/report_data/train.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_debugging_slot_names/train_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader