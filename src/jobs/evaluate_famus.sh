#!/bin/bash
#SBATCH --partition=brtx6
#SBATCH --time=24:0:0
#SBATCH --gpus=1
#SBATCH --output=/brtx/601-nvme1/svashis3/iterx/src/jobs/evaluate_famus_dev.log

eval "$(conda shell.bash hook)"
conda activate iterx

cd /brtx/601-nvme1/svashis3/iterx
############################
#### DEV Predictions #######
############################
# Report
PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_gold_spans \
             resources/data/famus/report_data/gold_spans/dev.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/report_data/gold_spans/dev.jsonl": "resources/data/famus/report_data/gold_spans/dev.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_gold_spans/dev_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader

PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_mixed_spans \
             resources/data/famus/report_data/mixed_spans/dev.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/report_data/mixed_spans/dev.jsonl": "resources/data/famus/report_data/mixed_spans/dev.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_mixed_spans/dev_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader
# Source
PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_gold_spans \
             resources/data/famus/source_data/gold_spans/dev.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/source_data/gold_spans/dev.jsonl": "resources/data/famus/source_data/gold_spans/dev.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_gold_spans/dev_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader

PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_mixed_spans \
             resources/data/famus/source_data/mixed_spans/dev.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/source_data/mixed_spans/dev.jsonl": "resources/data/famus/source_data/mixed_spans/dev.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_mixed_spans/dev_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader

############################
#### Test Predictions #######
############################
# Report Data
PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_gold_spans \
             resources/data/famus/report_data/gold_spans/test.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/report_data/gold_spans/test.jsonl": "resources/data/famus/report_data/gold_spans/test.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_gold_spans/test_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader


PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_mixed_spans \
             resources/data/famus/report_data/mixed_spans/test.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/report_data/mixed_spans/test.jsonl": "resources/data/famus/report_data/mixed_spans/test.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_report_data_mixed_spans/test_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader

# Source
PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_gold_spans \
             resources/data/famus/source_data/gold_spans/test.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/source_data/gold_spans/test.jsonl": "resources/data/famus/source_data/gold_spans/test.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_gold_spans/test_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader


PYTHONPATH=./src allennlp predict \
           /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_mixed_spans \
             resources/data/famus/source_data/mixed_spans/test.jsonl \
              --include-package iterx \
              --overrides '{"model.metrics.famus.doc_path": {"resources/data/famus/source_data/mixed_spans/test.jsonl": "resources/data/famus/source_data/mixed_spans/test.jsonl"}}' \
              --output-file /brtx/601-nvme1/svashis3/iterx/famus_model_source_data_mixed_spans/test_predictions.jsonl \
              --cuda-device 0 \
            --use-dataset-reader
