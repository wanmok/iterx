python preprocess.py --frames_list_path "/data/sid/rams2/pilots/event_state_process_Spanfinder.txt" \
                        --role_annotation_path "/data/sid/rams2/pilots/role_annotation_task/data/role_filler_bulk_task_cumulative_annotations.jsonl" \
                        --base_output_path "/data/sid/iterx/resources/data/famus/"

python prune_preprocessed_data.py \
            --input_data_path "/data/sid/iterx/resources/data/famus/preprocessed/tokenized/"
