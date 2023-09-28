import spacy_alignments as tokenizations
import argparse
import json
import os
from tqdm import tqdm

def _modify_template_spans(template,
                            charoffset,
                            tokenoffset,
                            spans_to_idx_map):
    import copy
    """Modify the template spans to account for the concatenated report and source text."""
    new_template_dict = {}
    span_index_list = []
    new_template_dict['incident_type'] = template['incident_type']
    # each role can have multiple spans
    for role_name, role_spans in template.items():
        if role_name == 'incident_type' or role_name == 'template-spans':
            continue
        copied_role_spans = copy.deepcopy(role_spans)
        # non-emty role spans
        if role_spans != []:
            # we modify the indices for each span in the role
            # print(copied_role_spans)
            # index 0 for all refers to the fact that there is a single span 
            # in each coref cluster of a role span
            for current_span in copied_role_spans:
                current_span[0][1] += charoffset
                current_span[0][2] += charoffset
                current_span[0][3] += tokenoffset
                current_span[0][4] += tokenoffset
                span_index_list.append(spans_to_idx_map[tuple(current_span[0][1:5])])
        
        new_template_dict[role_name] = copied_role_spans

    # find the indices of the spans in the all-spans list
    span_index_list_unique = sorted(list(set(span_index_list)))
    new_template_dict['template-spans'] = span_index_list_unique 

    return [new_template_dict]


def famusInstance2IterXSourceInstance(instance,
                                      trigger_tag='event'):
    """
    Given a FAMUS instance, return an IterX instance with the following fields:
    'doctext': report text with trigger + source text
    'all-spans': trigger span + all spans from the source text
    'templates': templates from the source text with correct span indices
    'doctext-tok': report text with trigger + source text tokenized
    'tok2char': 
    'char2tok', 
    'all-span-sets',
    'spans-to-spanset'
    """
    passage_with_trigger_dict = _famusInstance2ModifiedReportwithTrigger(instance,
                                                                trigger_tag=trigger_tag)
    source_data = instance["source"]
    # We concatenate the report text with the Source text
    reportAndSource_text = passage_with_trigger_dict['doctext'] + " " + source_data['doctext']
    reportAndSource_tokens = passage_with_trigger_dict['doctext-tok'] + source_data['doctext-tok']

    tok2char, char2tok = tokenizations.get_alignments(reportAndSource_tokens, 
                                                      [char for char in reportAndSource_text])
    
    ############################ Fix Indices for all-spans ##################################
    # Shift all-spans Chars and token indices by the length of the report text
    modified_spans = []
    # the 1 is added to account for the space between the report and source text
    charoffset = len(passage_with_trigger_dict['doctext']) + 1
    tokenoffset = len(passage_with_trigger_dict['doctext-tok'])

    # Add the trigger span as the first span
    modified_spans.append(passage_with_trigger_dict['trigger-span'])

    # Add the rest of the spans using the offsets
    for span_string, ch_start, ch_end, tok_start, tok_end, extra_string_val in source_data['all-spans']:
        modified_spans.append([span_string, 
                               ch_start + charoffset, 
                               ch_end + charoffset,
                               tok_start + tokenoffset, 
                               tok_end + tokenoffset, 
                               extra_string_val])
        
    #####################  Get Span-to-Idx Dictionary To be used by Templates ####################
    # Sort the list based on the start_char_idx and then the end_char_idx
    modified_spans = sorted(modified_spans, key=lambda x: x[1] +  x[2]/1000)

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    spans_to_idx_map = {}
    for span_idx, span in enumerate(modified_spans):
        # the key is a tuple of the start char index, end char index, start token index, end token index
        spans_to_idx_map[tuple((span[1:5]))] = span_idx
    
    IterxSourceInstance = {'docid': instance['instance_id'],
                           'doctext': reportAndSource_text,
                           'doctext-tok': reportAndSource_tokens,
                           'all-spans': modified_spans,
                           'templates': _modify_template_spans(source_data['templates'][0],
                                                                charoffset,
                                                                tokenoffset,
                                                                spans_to_idx_map),
                           'tok2char': tok2char,
                           'char2tok': char2tok,
                           'all-span-sets': [[span] for span in modified_spans],
                           'spans-to-spanset': [i for i in range(len(modified_spans))]
                          }
    
    return IterxSourceInstance


def famusInstance2IterXReportInstance(instance, trigger_tag='event'):
    """
    Given a FAMUS instance, return an IterX instance with the following fields:
    'doctext': report text with trigger + source text
    'all-spans': trigger span + all spans from the source text
    'templates': templates from the source text with correct span indices
    'doctext-tok': report text with trigger + source text tokenized
    'tok2char': 
    'char2tok', 
    'all-span-sets',
    'spans-to-spanset'
    """
    # For report, we fetch passage_with_trigger_dict only to get the trigger characters and tokens
    # we don't need the trigger spans as they would the the first token and chars in the report
    passage_with_trigger_dict = _famusInstance2ModifiedReportwithTrigger(instance,
                                                                trigger_tag=trigger_tag)
    
    report_data = instance['report']
    trigger_text_in_report = passage_with_trigger_dict['trigger-span'][0]
    trigger_token_start_idx = passage_with_trigger_dict['trigger-span'][3]
    trigger_token_end_idx = passage_with_trigger_dict['trigger-span'][4]
    trigger_tokens_in_report = passage_with_trigger_dict['doctext-tok'][trigger_token_start_idx:trigger_token_end_idx+1]

    # We concatenate the report text with the Source text
    triggerAndReport_text = trigger_text_in_report + " " + report_data['doctext']
    triggerAndReport_tokens = trigger_tokens_in_report + report_data['doctext-tok']

    tok2char, char2tok = tokenizations.get_alignments(triggerAndReport_tokens, 
                                                      [char for char in triggerAndReport_text])
    
    ############################ Fix Indices for all-spans ##################################
    # Shift all-spans Chars and token indices by the length of the report text
    modified_spans = []
    # the 1 is added to account for the space between the report and source text
    charoffset = len(trigger_text_in_report) + 1
    tokenoffset = len(trigger_tokens_in_report)

    # Add the trigger span as the first span
    # -1 is added because the end index is inclusive
    trigger_span = [trigger_text_in_report, 0 , 
                    len(trigger_text_in_report)-1, 
                    0, len(trigger_tokens_in_report)-1, 
                    '']
    
    modified_spans.append(trigger_span)

    # Add the rest of the spans using the offsets
    for span_string, ch_start, ch_end, tok_start, tok_end, extra_string_val in report_data['all-spans']:
        modified_spans.append([span_string, 
                               ch_start + charoffset, 
                               ch_end + charoffset,
                               tok_start + tokenoffset, 
                               tok_end + tokenoffset, 
                               extra_string_val])
        
    #####################  Get Span-to-Idx Dictionary To be used by Templates ####################
    # Sort the list based on the start_char_idx and then the end_char_idx
    modified_spans = sorted(modified_spans, key=lambda x: x[1] +  x[2]/1000)

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    spans_to_idx_map = {}
    for span_idx, span in enumerate(modified_spans):
        # the key is a tuple of the start char index, end char index, start token index, end token index
        spans_to_idx_map[tuple((span[1:5]))] = span_idx
    
    IterxReportInstance = {'docid': instance['instance_id'],
                           'doctext': triggerAndReport_text,
                           'doctext-tok': triggerAndReport_tokens,
                           'all-spans': modified_spans,
                           'templates': _modify_template_spans(report_data['templates'][0],
                                                                charoffset,
                                                                tokenoffset,
                                                                spans_to_idx_map),
                           'tok2char': tok2char,
                           'char2tok': char2tok,
                           'all-span-sets': [[span] for span in modified_spans],
                           'spans-to-spanset': [i for i in range(len(modified_spans))]
                          }
    
    return IterxReportInstance


def _famusInstance2ModifiedReportwithTrigger(instance, trigger_tag='event'):
    """
    Given famus release instance,
    output a string of passage sentences with the span highlighted in a trigger_tag
    """
    new_instance = {}

    trigger_span = instance['report']['trigger-span']

    trigger_token_text = trigger_span[0]
    trigger_char_start_idx = trigger_span[1]
    trigger_token_start_idx = trigger_span[3]
    trigger_token_end_idx = trigger_span[4] 
    passage_tokens = instance['report']['doctext-tok']

    # Updated fields
    # we add 1 to the end index because the end index is inclusive
    modified_trigger_tokens = [f"<{trigger_tag}>"] + \
                        passage_tokens[trigger_token_start_idx:trigger_token_end_idx+1] + \
                        [f"</{trigger_tag}>"] 
    modified_tokens = passage_tokens[:trigger_token_start_idx] + \
                        modified_trigger_tokens + \
                        passage_tokens[trigger_token_end_idx+1:]
    
    # the two +1's are for the spaces
    new_trigger_char_length = len(f"<{trigger_tag}>") + \
                                1 + \
                                len(trigger_token_text) + \
                                1 + \
                                len(f"</{trigger_tag}>") 

    # the +2 addition is for the two trigger <event> and </event>  
    # the -1 is because the end index is inclusive      
    modified_trigger_span = [" ".join(modified_trigger_tokens),
                        trigger_char_start_idx,
                        trigger_char_start_idx + new_trigger_char_length - 1,
                        trigger_token_start_idx,
                        trigger_token_end_idx+2,
                        '']
                        

    new_instance = {'doctext': " ".join(modified_tokens),
                    'doctext-tok': modified_tokens,
                    'trigger-span': modified_trigger_span}
    
    # sanity check: the trigger string should match the span constructed from char indices
    string_from_char_indices = new_instance['doctext'][new_instance['trigger-span'][1]:new_instance['trigger-span'][2]+1]
    assert string_from_char_indices == new_instance['trigger-span'][0]

    return new_instance


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
    # finished_strings = set()

    for (string, start_char, end_char, start_tok, end_tok, type_str) in all_spans:
        # If the exact string is present multiple times, only extract the first one
        # if string in finished_strings:
        #     continue

        cluster_num = char_spans_to_coref_cluster_id[(start_char, end_char)]
        # cluster is matched
        if cluster_num != 0 and cluster_num not in finished_clusters:
            finished_clusters.add(cluster_num)
            filtered_spans.append((string, start_char, end_char, start_tok, end_tok, type_str))
            # finished_strings.add(string)

        elif cluster_num !=0 and cluster_num in finished_clusters:
            continue

        # cluster is not matched
        else:
            filtered_spans.append((string, start_char, end_char, start_tok, end_tok, type_str))
            # finished_strings.add(string)

    filtered_spans = sorted(filtered_spans, key=lambda x: x[1] + x[2]/1000)
    
    return filtered_spans


def convertInstanceBasedonCorefClusters(instance,
                                        docid_to_source_coref_clusters_as_spans,
                                        filter_approach = 'first'):
    """
    Given a instance, convert the spans in each template based on the coref clusters.
    This function changes to keys in of the instance -- 'templates' and 'all-spans'
    
    Args:
        instance: an iterx instance 
        docid_to_source_coref_clusters_as_spans: dict mapping docid to coref clusters
        filter_approach: 'first', 'longest', 'dense'

    Returns:
        instance: an iterx instance with the 'templates' and 'all-spans' keys changed
    """
    # deep copy of the instance
    import copy
    instance = copy.deepcopy(instance)

    docid = instance['docid']

    charTupl2clusterid, clusterid2charTupl = convertCorefClusterSpans2AllSpanClusterDict(docid_to_source_coref_clusters_as_spans[docid])
    filtered_spans = filter_spans_by_coref_clusters(instance['all-spans'], charTupl2clusterid)    

    # add spans_to_idx_map which maps a span to its index in all-spans
    # this is used to fill template-spans field inside the templates field of each instance
    char_spans_to_all_span_idx_map = {}
    for span_idx, span in enumerate(filtered_spans):
        # the key is a tuple of the start char index, end char index
        char_spans_to_all_span_idx_map[tuple((span[1:3]))] = span_idx

    span_index_list = []
    # there's only one template in each instance
    template = instance['templates'][0]
    modified_template = {}
    for role, role_spans in template.items():
        if role == 'incident_type':
            modified_template[role] = role_spans

        elif role == "template-spans":
            continue

        # any other role than incident_type and template-spans with empty spans
        elif role_spans == []:
            modified_template[role] = []

        # any other role than incident_type and template-spans with non-empty spans
        else:
            span_str, char_start, char_end, token_start, token_end, span_frame_type, role_name = role_spans[0][0]
            cluster_id = charTupl2clusterid[(char_start, char_end)]
            # if the span tuple falls in a valid coref cluster, then edit the span
            # based on the filter approach
            if cluster_id:
                # get the correct char span based on approach
                if filter_approach == 'first':
                    cluster_item_idx = 0
                    char_start, char_end = clusterid2charTupl[cluster_id][cluster_item_idx]
                    # it is possible that the first span in the cluster is not in the filtered_spans list
                    # (possibly due to the fact that coref identifies this as an entity but the all-spans didn't)
                    # in such cases, we need to find the next span in the cluster that is in the filtered_spans list
                    while (char_start, char_end) not in char_spans_to_all_span_idx_map:
                        cluster_item_idx += 1
                        char_start, char_end = clusterid2charTupl[cluster_id][cluster_item_idx]

            # fetch the full span from the filtered_spans list
            span_idx = char_spans_to_all_span_idx_map[(char_start, char_end)]
            span_index_list.append(span_idx)
            
            char_token_idxs = list(filtered_spans[span_idx])
            char_token_idxs.append(role)
            modified_template[role] = [[char_token_idxs]]   

    # find the indices of the spans in the all-spans list
    span_index_list_unique = sorted(list(set(span_index_list)))
    modified_template['template-spans'] = span_index_list_unique  

    instance['all-spans'] = filtered_spans
    instance['templates'] = [modified_template]
    
    # 'all-span-sets' are coref clusters: for now we just assume each span is its own cluster
    instance['all-span-sets'] = [[span] for span in instance['all-spans']]
    instance['spans-to-spanset'] = [i for i in range(len(instance['all-spans']))]

    return instance


def convertCorefClusterSpans2AllSpanClusterDict(cluster_spans):
    """
    Given a cluster span dict from FCoref model, convert it to
     All_span clster dict format.
      
    Input eg:
    [[(100, 167), (214, 225)],
    [(301, 311), (319, 321), (2051, 2069)],
    [(61, 87), (484, 492), (503, 506),(1004, 1012),...],
    ...
    Output eg:
    charTupl2cluster = {(100, 166): 1, (214, 224): 1,
                    (301, 310): 2, (319, 320): 2, (2051, 2068): 2,
                    (61, 86): 3, (484, 491): 3, (503, 505): 3, (1004, 1011): 3,...}

    cluster2charTupl = {1: [(100, 166), (214, 224)],
                    2: [(301, 310), (319, 320), (2051, 2068)],
                    3: [(61, 86), (484, 491), (503, 505), (1004, 1011),...],}
    

    Note that the end_idx is inclusive in all-spans but exclusive in coref clusters.
    """
    from collections import defaultdict
    charTupl2cluster = defaultdict(int)
    cluster2charTupl = defaultdict(list)
    # Create both charTupl2cluster and cluster2charTupl
    for cluster_id, (cluster_spans) in enumerate(cluster_spans):
        for char_start, char_end in cluster_spans:
            # we subtract 1 from char_end because the end_idx is inclusive in all-spans
            current_cluster_id = cluster_id+1
            charTupl2cluster[(char_start, char_end-1)] = current_cluster_id
            cluster2charTupl[current_cluster_id].append((char_start, char_end-1))

    return charTupl2cluster, cluster2charTupl

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        type=str, 
                        required=True,
                        default = "/data/sid/iterx/resources/data/famus/release_format",
                        help='Path to the input release format files')
    
    parser.add_argument('--output_dir', type=str, 
                        required=True,
                        default = "/data/sid/iterx/resources/data/famus/iterx_format",
                        help='Path to the output iterx format files')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print("Reading the release format files...")
    with open(os.path.join(args.input_dir, "train.jsonl")) as f:
        train = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "dev.jsonl")) as f:
        dev = [json.loads(line) for line in f]

    with open(os.path.join(args.input_dir, "test.jsonl")) as f:
        test = [json.loads(line) for line in f]
    ####################################################################
    ########## Report Text Data ##########################################
    ####################################################################
    # convert the train, dev, test instances to iterx format
    print("Converting to iterx format for report...")
    train_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(train)]
    dev_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(dev)]
    test_iterx_report = [famusInstance2IterXReportInstance(instance) for instance in tqdm(test)]

    report_export_path = os.path.join(args.output_dir, "report_data")
    # Export the iterx format instances
    with open(os.path.join(report_export_path, "train.jsonl"), "w") as f:
        for instance in train_iterx_report:
            f.write(json.dumps(instance) + "\n")

    with open(os.path.join(report_export_path, "dev.jsonl"), "w") as f:
        for instance in dev_iterx_report:
            f.write(json.dumps(instance) + "\n")

    with open(os.path.join(report_export_path, "test.jsonl"), "w") as f:
        for instance in test_iterx_report:
            f.write(json.dumps(instance) + "\n")

    print(f"Report Data Files exported to: {report_export_path}")

    ####################################################################
    ########## Source Text Data ########################################
    ####################################################################
    # convert the train, dev, test instances to iterx format
    print("Converting to iterx format for source...")
    train_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(train)]
    dev_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(dev)]
    test_iterx_source = [famusInstance2IterXSourceInstance(instance) for instance in tqdm(test)]

    source_export_path = os.path.join(args.output_dir, "source_data")
    # Export the iterx format instances
    with open(os.path.join(source_export_path, "train.jsonl"), "w") as f:
        for instance in train_iterx_source:
            f.write(json.dumps(instance) + "\n")
    
    with open(os.path.join(source_export_path, "dev.jsonl"), "w") as f:
        for instance in dev_iterx_source:
            f.write(json.dumps(instance) + "\n")

    with open(os.path.join(source_export_path, "test.jsonl"), "w") as f:
        for instance in test_iterx_source:
            f.write(json.dumps(instance) + "\n")

    print(f"Source Data Files exported to: {source_export_path}")

if __name__ == "__main__":
    main()
    