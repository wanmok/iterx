FRAMES_LIST_PATH="/data/sid/rams2/pilots/event_state_process_Spanfinder.txt" 
ROLE_ANNOTATION_PATH="/data/sid/rams2/pilots/role_annotation_task/data/role_filler_bulk_task_cumulative_annotations.jsonl"
FAMUS_RELEASE_DIR="/data/sid/iterx/resources/data/famus/release_format"

echo "Running the release format preprocessing pipeline"
python preprocess_raw_famus_to_release_version.py \
        --frames_list_path $FRAMES_LIST_PATH \
        --role_annotation_path $ROLE_ANNOTATION_PATH \
        --output_path $FAMUS_RELEASE_DIR

echo "Running the QA format preprocessing pipeline"
python preprocess_release_format_to_qa_format.py \
        --input_dir $FAMUS_RELEASE_DIR \
        --output_dir "/data/sid/iterx/resources/data/famus/qa_format"

echo "Running the IterX format preprocessing pipeline"
python preprocess_release_format_to_iterx_format.py \
        --input_dir $FAMUS_RELEASE_DIR \
        --output_dir "/data/sid/iterx/resources/data/famus/iterx_format/"
        