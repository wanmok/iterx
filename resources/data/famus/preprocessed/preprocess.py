# Preprocess the FAMUS data to convert it to the iterx format.
# Running instructions
# python preprocess.py --frames_list_path /data/sid/rams2/pilots/event_state_process_Spanfinder.txt \
#                         --role_annotation_path /data/sid/rams2/pilots/role_annotation_task/data/role_filler_bulk_task_cumulative_annotations.jsonl \
#                         --base_output_path "/data/sid/iterx/resources/data/famus/" 


from typing import List, Dict, Tuple
import pandas as pd
import json
import spacy_alignments as tokenizations
import argparse
from nltk.corpus import framenet as fn
from tqdm import tqdm
import os
import numpy as np
from fastcoref import FCoref


def frame_to_core_roles(frame: str):
    """
    extract a list of all core roles
    """
    ## Added extra roles that are generally frequent
    extra_roles = ['Time', 'Place']

    extra_valid_roles = [role for role in extra_roles if role in fn.frame(frame).FE]

    core_roles = [ role for role, dcts in fn.frame(frame).FE.items() 
                                    if dcts['coreType']=="Core" or dcts['coreType']=="Core-Unexpressed"]

    extra_valid_roles = [role for role in extra_valid_roles if role not in core_roles]

    return core_roles + extra_valid_roles


def frame_to_definition_format(frame):
    '''Given an input frame, return a dictionary with the format expected 
    by the definition format of the MUC definitions.
    '''
    core_roles = frame_to_core_roles(frame)
    output_dict = {}

    output_dict['output_str'] = frame
    output_dict['slots'] = {}

    for role in core_roles:
        output_dict['slots'][role] = {'filler_types': ['spanset', 'event']}

    return output_dict


def sentence_token_span_to_doc_spans(sentence_token_spans, 
                                     document_tokens: List[List[str]]):
    """
    Given a candidate span in terms of sentence and token indices, return the
    corresponding span in terms of document character and document token indices.

    Parameters
    ----------
    sentence_token_spans : Dict[str, int]
        A dictionary mapping sentence indices to token indices.
        eg: {'sentenceIndex':2, 'startToken': 5, 'endToken': 6}

        The startToken is the first token in the span in that sentence and
        the endToken is the last token in the span in that sentence.

    note: endTokens are exclusive, so the span is [startToken, endToken)
    but the endToken is inclusive in the iterx format, so we subtract 1 to the endToken

    document_text : str

    Returns
    -------
    A list of 4 integers: [start_char_idx, end_char_idx, start_token_idx, end_token_idx]
    """
    document_text = " ".join([token for sent in document_tokens
                                        for token in sent])
    
    sentence_idx = sentence_token_spans['sentenceIndex']

    if sentence_idx == -1:
        return ['', -1, -1, -1, -1, '']

    # compute the doc level start and end token indices
    doc_level_start_token_idx = len([token for sent in document_tokens[:sentence_idx]
                                    for token in sent]) + sentence_token_spans['startToken']
    doc_level_end_token_idx = doc_level_start_token_idx + (sentence_token_spans['endToken'] - 
                                                           sentence_token_spans['startToken'])

    # store empty spaces before the start of the sentence and before the start of the token
    # based on the sentence and token indices
    if sentence_token_spans['startToken'] == 0:
        previous_token_space = 0
    else:
        previous_token_space = 1

    if sentence_idx == 0:
        previous_sentence_space = 0
    else:
        previous_sentence_space = 1

    doc_level_start_char_idx = previous_sentence_space  + previous_token_space + \
                len(" ".join([token for sent in document_tokens[:sentence_idx] 
                                                for token in sent])) + \
                len(" ".join(document_tokens[sentence_idx][:sentence_token_spans['startToken']]))

    doc_level_end_char_idx = doc_level_start_char_idx + len(" ".join(document_tokens[
                                    sentence_idx][sentence_token_spans['startToken']:
                                                  sentence_token_spans['endToken']]))
    
    char_text = document_text[doc_level_start_char_idx: doc_level_end_char_idx]

    # assert tokens and characters match
    assert char_text.strip() == " ".join(document_tokens[sentence_idx][
                                                sentence_token_spans['startToken']:
                                                sentence_token_spans['endToken']]).strip()

    # The last return item is the Frame type of the span (Producer, Product, etc.)
    # This is kept empty for now, but can be used later to see if modeling improves

    # we subtract 1 from end_char_idx because the end_char_idx is inclusive in the iterx format
    # but exclusive in the sentence_token_spans format. Similarly for end_token_idx.

    return [char_text, doc_level_start_char_idx, doc_level_end_char_idx-1, 
            doc_level_start_token_idx, doc_level_end_token_idx-1,
            '']
    
    

def annotated_spans_to_iterx_template_format(frame: str,
                                             annotated_spans: List[Dict],
                                             document_sentences: List[List[str]],
                                             iterx_span_to_idx_map: Dict):
    """
    Convert the annotated spans to the iterx template format.

    Parameters
    ----------
    frame : str
        The frame for which the annotated spans are being converted to the iterx template format.
    annotated_spans : List[Dict]
        Eg: [{'frameSpan': {}, 'role': 'Producer', 'notPresent': False, 
              'sentenceIndex': 2, 'startToken': 5,'endToken': 6,
              'status': 'ok', n'scrollTopValue': 0},
             {'frameSpan': {}, 'role': 'Product', 'notPresent': False,
              'sentenceIndex': 2, 'startToken': 17, 'endToken': 22,
              'status': 'ok', 'scrollTopValue': 0},
    document_sentences : List[List[str]]
        A list of sentences in the document.
    iterx_span_to_idx_map : Dict
        A dictionary mapping spans (in iterx format) to their all-spans indices.
            eg: {(0, 1, 2, 3): 0, (4, 5, 6, 7): 1)}
            here the keys are the spans which represent four numbers:
                start_char_idx, end_char_idx, start_token_idx, end_token_idx
            and the values are the indices of the corresponding spans in the all-spans list.

    Returns
    -------
    A list of dictionaries, each dictionary representing a template.
    """
    span_index_list = []

    template_dict = {}
    template_dict['incident_type'] = frame
    
    # sort the spans based on sentence index and start token index
    for span in  sorted(annotated_spans, key= lambda x: str(x['sentenceIndex']) + '_' + str(x['startToken'])):
        if span['sentenceIndex'] == -1:
            template_dict[span['role']] = []
        # this is to take care of a bug that was present in the annotation tool
        # where the startToken was greater than the endToken
        # We ignore such spans
        elif span['startToken'] >= span['endToken']:
            print("Annotated Span startToken exceeds endToken, so skipping this span: ", span)
            continue
        else:
            char_token_idxs = sentence_token_span_to_doc_spans(span, document_sentences)
            # the indices 1,2,3,4 represent the start_char_idx, end_char_idx,
            # start_token_idx, end_token_idx respectively
            span_index_list.append(iterx_span_to_idx_map[tuple(char_token_idxs[1:5])])
            char_token_idxs.append(span['role'])
            template_dict[span['role']] = [[char_token_idxs]]

    # find the indices of the spans in the all-spans list
    span_index_list_unique = sorted(list(set(span_index_list)))
    template_dict['template-spans'] = span_index_list_unique

    return [template_dict]


def role_df_row_to_iterx_instance(input_json: Dict,
                                  annotated_frame: str,
                                  annotated_sourceSpans: List[Dict]):
    """
    Convert a row of the role annotation dataframe to an iterx instance.

    Parameters
    ----------
    input_json : Dict
        The input.json field of the row.
    annotated_frame : str
        The frame for which the role annotation is being converted to the iterx instance.
    annotated_sourceSpans : List[Dict]
        The sourceSpans field of the row.

    Returns
    -------
    A dictionary representing an iterx instance.
    """
    # Fix the spans in the input_json
    # There are some spans where the startToken is greater than the endToken (upto iteration 4)
    # This is a bug in the annotation tool, so we ignore such spans
    fixed_sourceSpans = []
    for span in input_json['sourceCandidateSpans']:
        if span['startToken'] >= span['endToken']:
            # print("Source candidate Span startToken exceeds endToken, so skipping this span: ", span)
            continue
        else:
            fixed_sourceSpans.append(span)
    

    docid = input_json['passage_id'] + '-' + input_json['Spanfinder_frame_prediction']
    source_text = " ".join([token for sent in input_json['sourceSentences']
                    for token in sent])
    instance = {}
    instance['docid'] = docid
    instance['doctext'] = source_text

    # all spans from the input.json data field
    instance['all-spans'] = set()
    for span in fixed_sourceSpans:
        current_iterx_span = sentence_token_span_to_doc_spans(span, 
                                                              input_json['sourceSentences'])
        instance['all-spans'].add(tuple(current_iterx_span))

    # Add all GOLD spans to all-spans
    for span in annotated_sourceSpans:
        current_iterx_span = sentence_token_span_to_doc_spans(span, 
                                                              input_json['sourceSentences'])
        if current_iterx_span[0] != '':
            instance['all-spans'].add(tuple(current_iterx_span))

    # Convert the set to a list and sort it based on the start_char_idx
    instance['all-spans'] = sorted(list(instance['all-spans']), key=lambda x: x[1])

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    spans_to_idx_map = {}
    for span_idx, span in enumerate(instance['all-spans']):
        # the key is a tuple of the start char index, end char index, start token index, end token index
        spans_to_idx_map[tuple((span[1:5]))] = span_idx


    instance['templates'] = annotated_spans_to_iterx_template_format(annotated_frame,
                                         annotated_sourceSpans,
                                         input_json['sourceSentences'],
                                         spans_to_idx_map)
    
    instance['doctext-tok'] = [token for sent in input_json['sourceSentences']
                                for token in sent]
    
    tok2char, char2tok = tokenizations.get_alignments(instance['doctext-tok'], 
                                                      [char for char in instance['doctext']])
    
    instance['tok2char'] = tok2char
    instance['char2tok'] = char2tok
    instance['spans-to-spanset'] = []
    

    return instance


## Split role_df into train, dev, test
def split_df_to_train_dev_test_by_frame(df,
                                        train_frac=0.6,
                                        dev_frac=0.2,
                                        test_frac=0.2,
                                        random_state=42):
    """
    SPlit the role_df into train, dev, test by frame

    Args:
        df (pandas.DataFrame): role_df

    Returns:
        train_df, dev_df, test_df (pandas.DataFrame)
    """
    import math
    all_frames = sorted(df['final_frame'].unique())
    np.random.seed(random_state)

    # for each frame, create a subdf filtering only that frame
    # and then split that subdf into train, dev, test
    train_dfs = []
    dev_dfs = []
    test_dfs = []

    for frame in tqdm(all_frames):
        subdf = df[df['final_frame'] == frame]
        num_docs = len(subdf)
        num_train = int(num_docs * train_frac)
        num_dev = int(num_docs * dev_frac)
        num_test = num_docs - num_train - num_dev

        # shuffle the subdf
        subdf = subdf.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # split the subdf into train, dev, test
        train_dfs.append(subdf.iloc[:num_train])
        dev_dfs.append(subdf.iloc[num_train:num_train+num_dev])
        test_dfs.append(subdf.iloc[num_train+num_dev:])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    dev_df = pd.concat(dev_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)

    return train_df, dev_df, test_df

    
def convert_df_to_iterx_instances(df):
    """
    Convert the role_df to iterx instances

    Args:
        df (pandas.DataFrame): role_df

    Returns:
        iterx_instances (list): list of iterx instances
    """
    iterx_instances = []
    for row_idx, row in tqdm(df.iterrows()):
        input_json = json.loads(row['Input.json'])
        iterx_instances.append(role_df_row_to_iterx_instance(input_json,
                                                         row['full_role_dict']['frame'],
                                                         row['full_sourceSpans']))
    return iterx_instances


def export_to_jsonl(instances, output_path):
    with open(output_path, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_list_path', 
                        type=str, 
                        required=True,
                        default = "/data/sid/rams2/pilots/event_state_process_Spanfinder.txt",
                        help='Path to frames list to be used')
    
    parser.add_argument('--role_annotation_path', type=str, 
                        required=True,
                        default = "/data/sid/rams2/pilots/role_annotation_task/data/role_filler_bulk_task_cumulative_annotations.jsonl",
                        help='Path to the role annotation jsonl file')
    
    parser.add_argument('--base_output_path', type=str, 
                        required=True,
                        default = "/data/sid/iterx/resources/data/famus/",
                        help='Base directory path to the output files')
    
    return parser.parse_args()


def convertCorefClusterSpans2AllSpanClusterDict(cluster_spans):
    """
    Given a cluster span dict from FCoref model, convert it to
     All_span clster dict format.
      
    Input eg:
    [[(100, 167), (214, 225)],
    [(301, 311), (319, 321), (2051, 2069)],
    [(61, 87),
    (484, 492),
    (503, 506),
    (1004, 1012),
    (1440, 1443),
    (1479, 1487),
    (1497, 1498),
    (1921, 1929),
    (1962, 1964)],
    ...

    Output eg:
    {(100, 166): 1,
    (214, 224): 1,
    (301, 310): 2,
    (319, 320): 2,
    (2051, 2068): 2,
    (61, 86): 3,
    (484, 491): 3,
    (503, 505): 3,
    (1004, 1011): 3,
    (1440, 1442): 3,
    (1479, 1486): 3,
    (1497, 1497): 3,
    (1921, 1928): 3,
    (1962, 1963): 3, ...
    

    Note that the end_idx is inclusive in all-spans but exclusive in coref clusters.
    """
    from collections import defaultdict
    char_spans_to_coref_cluster_id = defaultdict(int)

    for cluster_id, (cluster_spans) in enumerate(cluster_spans):
        for char_start, char_end in cluster_spans:
            # we subtract 1 from char_end because the end_idx is inclusive in all-spans
            char_spans_to_coref_cluster_id[(char_start, char_end-1)] = cluster_id+1

    return char_spans_to_coref_cluster_id

def filter_spans_by_coref_clusters(all_spans, 
                                   char_spans_to_coref_cluster_id,
                                   filter_type = "first"):
    """
    Given 'all-spans' of a iterx format instance and the char_spans_to_coref_cluster_id dict,
    filter spans such that we have at most one span from each coref cluster.

    Parameters
    ----------
    all_spans : list
        List of all spans in the iterx format instance.
        eg: [['Sex education', 8, 20, 4, 5, ''],
             ['Sex', 8, 10, 4, 4, '']] ...
    char_spans_to_coref_cluster_id : dict
        Eg: {(100, 166): 1,
            (214, 224): 1,
            (301, 310): 2,
            (319, 320): 2, ...
    
    filter_type : str
        'first' or 'longest' or 'dense'
    """
    filtered_spans = []
    finished_clusters = set()
    finished_strings = set()

    for (string, start_char, end_char, start_tok, end_tok, type_str) in all_spans:
        # If the exact string is present multiple times, only extract the first one
        if string in finished_strings:
            continue

        cluster_num = char_spans_to_coref_cluster_id[(start_char, end_char)]
        # cluster is matched
        if cluster_num != 0 and cluster_num not in finished_clusters:
            finished_clusters.add(cluster_num)
            filtered_spans.append((string, start_char, end_char, start_tok, end_tok, type_str))
            finished_strings.add(string)

        elif cluster_num !=0 and cluster_num in finished_clusters:
            continue

        # cluster is not matched
        else:
            filtered_spans.append((string, start_char, end_char, start_tok, end_tok, type_str))
            finished_strings.add(string)

    return filtered_spans

def iterx_instances_to_instances_with_filtered_spans(instances, coref_model):
    """
    Given a list of iterx instances, return a list of instances with filtered spans
    based on coref clusters.
    """
    # Get the coref chains
    coref_preds = coref_model.predict(
                texts=[inst['doctext'] for inst in instances]
                    )
    
    docid_to_source_coref_clusters_as_spans = {}
    for docid, coref_pred in zip([inst['docid'] for inst in instances], coref_preds):
        docid_to_source_coref_clusters_as_spans[docid] = coref_pred.get_clusters(as_strings=False)


    for instance in instances:
        docid = instance['docid']
        char_spans_to_coref_cluster_id = convertCorefClusterSpans2AllSpanClusterDict(docid_to_source_coref_clusters_as_spans[docid])
        filtered_spans = filter_spans_by_coref_clusters(instance['all-spans'], char_spans_to_coref_cluster_id)
        instance['all-spans'] = filtered_spans

    return instances
        

def main():
    coref_model = FCoref(device='cuda:0')

    args = parse_arguments()

    # Select rows with relevant frames
    with open(args.frames_list_path) as f:
        v1_frames   = sorted([line.strip() for line in f if line.strip() != ''])

    ################################################################################################
    ########### Overwrite Definitions, Templates, and Slot files based on frames list ################
    ################################################################################################
    # definitions file
    famus_definition_dict = {}
    famus_definition_dict['@version'] = '1.0'
    famus_definition_dict['mappings'] = {}
    famus_definition_dict['definitions'] = {}

    for frame in v1_frames:
        famus_definition_dict['definitions'][frame] = frame_to_definition_format(frame)

    # save the dictionary as a json file
    with open(os.path.join(args.base_output_path, 'definitions.json'), 'w') as f:
        json.dump(famus_definition_dict, f, indent=4)


    # create template_labels.txt
    
    with open(os.path.join(args.base_output_path, 'vocabulary', 'template_labels.txt'), 'w') as f:
        for frame in v1_frames:
            f.write(frame + '\n')

    # create slot_types.txt
    initial_slots = ['@@UNKNOWN@@', 'none']
    slot_types_set = set()
    for frame in v1_frames:
        for role in frame_to_core_roles(frame):
            slot_types_set.add(role)
    slot_types_list = initial_slots + sorted(list(slot_types_set))
    with open(os.path.join(args.base_output_path, 'vocabulary', 'slot_types.txt'), 'w') as f:
        for slot_type in slot_types_list:
            f.write(slot_type + '\n')
    ################################################################################################
    ######## Load the role annotation data ##########
    ################################################################################################
    role_df = pd.read_json(args.role_annotation_path,
                                lines=True,
                                orient='records')
    # filter role_df to only include frames that are in v1_frames
    role_df['final_frame'] = role_df['full_role_dict'].map(lambda x: x['frame'])
    role_df = role_df[role_df['final_frame'].isin(v1_frames)]
    # Filter to frames with >= 5 docs
    frame2numDocs = role_df['final_frame'].value_counts().to_dict()
    frame_ge_5 = [frame for frame, numDocs in frame2numDocs.items() if numDocs >= 5]    
    role_df = role_df[role_df['final_frame'].isin(frame_ge_5)]

    ################################################################################################
    ####### Split role_df into train, dev, test by frame ########
    ################################################################################################
    train, dev, test = split_df_to_train_dev_test_by_frame(role_df)

    print(f"Total frames in train: {len(train['final_frame'].unique())}")
    print(f"Total frames in dev: {len(dev['final_frame'].unique())}")
    print(f"Total frames in test: {len(test['final_frame'].unique())}")

    train_iterx_instances = convert_df_to_iterx_instances(train)
    dev_iterx_instances = convert_df_to_iterx_instances(dev)
    test_iterx_instances = convert_df_to_iterx_instances(test)

    print(f"Total train instances: {len(train_iterx_instances)}")
    print(f"Total dev instances: {len(dev_iterx_instances)}")
    print(f"Total test instances: {len(test_iterx_instances)}")

    ################################################################################################
    ####### Export to jsonl
    ################################################################################################
    # make export directory
    export_dir = os.path.join(args.base_output_path, 'preprocessed', 'tokenized')
    
    os.makedirs(export_dir, exist_ok=True)

    export_to_jsonl(train_iterx_instances, os.path.join(export_dir, "train.jsonl"))
    export_to_jsonl(dev_iterx_instances, os.path.join(export_dir, "dev.jsonl"))
    export_to_jsonl(test_iterx_instances, os.path.join(export_dir, "test.jsonl"))


    #################################################################################################
    ####### Export to jsonl Filtered spans
    #################################################################################################
    


if __name__ == "__main__":
    main()
