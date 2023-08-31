import sys
sys.path.append("/data/sid/rams2/src/")

import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import argparse

from preprocess import role_df_row_to_iterx_instance
from preprocess import (convertCorefClusterSpans2AllSpanClusterDict, 
                        filter_spans_by_coref_clusters,
                        convertInstanceBasedonCorefClusters)

from preprocess import export_to_jsonl
from fastcoref import FCoref

def iterx_instances_to_filtered_instances(instances, 
                                          coref_model,
                                          filter_approach = 'first'):
    """
    Args:
        instances: list of iterx instances
        coref_model: fastcoref model


    Returns:
        filtered_instances: list of iterx instances with filtered spans
                            (based on coref clusters)
                            (i.e. choose one span from each coref cluster)
    """
    coref_preds = coref_model.predict(
                texts=[inst['doctext'] for inst in instances]
                    )

    docid_to_source_coref_clusters_as_spans = {}
    for docid, coref_pred in zip([inst['docid'] for inst in instances], coref_preds):
        docid_to_source_coref_clusters_as_spans[docid] = coref_pred.get_clusters(as_strings=False)

    filtered_instances = []
    for instance in tqdm(instances):
        instance_modif = convertInstanceBasedonCorefClusters(instance, 
                                                        docid_to_source_coref_clusters_as_spans, 
                                                        filter_approach = filter_approach)
        filtered_instances.append(instance_modif)

    return filtered_instances



def sample_smaller_iterx_instances(instances,
                                   all_spans_threshold=256,
                                   doctext_tok_threshold=400,):
    """Samples instances with smaller number of spans and smaller document length
    """
    sample_instances = []
    for inst in instances:
        if len(inst['all-spans']) < all_spans_threshold and (
            len(inst['doctext-tok'])< doctext_tok_threshold) :
            sample_instances.append(inst)

    return sample_instances

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_path",
                        type=str, default="/data/sid/iterx/resources/data/famus/preprocessed/tokenized/",
                        help="Path to the output directory")
    
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    coref_model = FCoref(device='cuda:0')
    # Load the data
    with open(os.path.join(args.input_data_path, 'train.jsonl')) as f:
        train = [json.loads(line) for line in f]

    with open(os.path.join(args.input_data_path, 'dev.jsonl')) as f:
        dev = [json.loads(line) for line in f]

    with open(os.path.join(args.input_data_path, 'test.jsonl')) as f:
        test = [json.loads(line) for line in f]

    # Run coref chains
    # Get the coref chains
    train_filtered_instances = iterx_instances_to_filtered_instances(train, coref_model)
    dev_filtered_instances = iterx_instances_to_filtered_instances(dev, coref_model)
    test_filtered_instances = iterx_instances_to_filtered_instances(test, coref_model)

    export_to_jsonl(train_filtered_instances, os.path.join(args.input_data_path,
                                                            "train_filtered_spans.jsonl"))
    export_to_jsonl(dev_filtered_instances, os.path.join(args.input_data_path, 
                                                         "dev_filtered_spans.jsonl"))
    export_to_jsonl(test_filtered_instances, os.path.join(args.input_data_path, 
                                                          "test_filtered_spans.jsonl"))
    
    ## Export a small sample of the data
    train_sample = sample_smaller_iterx_instances(train_filtered_instances)
    dev_sample = sample_smaller_iterx_instances(dev_filtered_instances)
    test_sample = sample_smaller_iterx_instances(test_filtered_instances)

    export_to_jsonl(train_sample, os.path.join(args.input_data_path,
                                                "train_filtered_spans_sample.jsonl"))
    export_to_jsonl(dev_sample, os.path.join(args.input_data_path,
                                                "dev_filtered_spans_sample.jsonl"))
    export_to_jsonl(test_sample, os.path.join(args.input_data_path,
                                                "test_filtered_spans_sample.jsonl"))

    

if __name__ == "__main__":
    main()
    
