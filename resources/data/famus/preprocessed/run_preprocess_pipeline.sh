FRAMES_LIST_PATH="/data/sid/rams2/pilots/event_state_process_Spanfinder.txt" 
ROLE_ANNOTATION_PATH="/data/sid/rams2/pilots/role_annotation_task/data/role_filler_bulk_task_cumulative_annotations.jsonl"

# python preprocess_famus_to_iterx_format.py \
#         --frames_list_path $FRAMES_LIST_PATH \
#         --role_annotation_path $ROLE_ANNOTATION_PATH \
#         --base_output_path "/data/sid/iterx/resources/data/famus/"

# python prune_preprocessed_data.py \
#         --input_data_path "/data/sid/iterx/resources/data/famus/preprocessed/tokenized/"

python preprocess_famus_to_squad_format.py \
        --frames_list_path $FRAMES_LIST_PATH \
        --role_annotation_path $ROLE_ANNOTATION_PATH
