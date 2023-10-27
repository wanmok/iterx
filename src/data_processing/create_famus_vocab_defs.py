
import os
import json
import argparse
from nltk.corpus import framenet as fn

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


def arg_parse():

    # input_dir
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir",
                        type=str,
                        default="/brtx/601-nvme1/svashis3/iterx/resources/data/famus"
                        )
    
    args  = parser.parse_args()

    return args

def main():
    args = arg_parse()
    print(f"Loading train files for fetching all the frames...")
    with open(os.path.join(args.output_dir, "report_data", "train.jsonl")) as f:
        train_instances = [json.loads(line) for line in f.readlines()]
    
    frames = sorted(list(set([instance['templates'][0]['incident_type'] for instance in train_instances])))
    print(f"Total number of frames: {len(frames)}")
    ########################################
    # Definitions
    ########################################
    famus_definition_dict = {}
    famus_definition_dict['@version'] = '1.0'
    famus_definition_dict['mappings'] = {}
    famus_definition_dict['definitions'] = {}
    for frame in frames:
        famus_definition_dict['definitions'][frame] = frame_to_definition_format(frame)

    # save the dictionary as a json file in output_dir
    with open(os.path.join(args.output_dir, 'definitions.json'), 'w') as f:
        json.dump(famus_definition_dict, f, indent=4)

    ########################################
    # Vocab
    ########################################
    
    ##########################
    #### slot_types.txt ####
    ##########################
    slot_set = set()
    for frame in frames:
        slots = frame_to_core_roles(frame)
        slot_set.update(slots)
    vocab_path = os.path.join(args.output_dir, 'vocabulary')
    os.makedirs(vocab_path, exist_ok=True)
    with open(os.path.join(vocab_path, 'slot_types.txt'), 'w') as f:
        # the first two lines should contain
        #@@UNKNOWN@@
        #none
        f.write('@@UNKNOWN@@\n')
        f.write('none\n')
        for slot in sorted(list(slot_set)):
            f.write(slot + '\n')

    ##########################
    #### template_labels.txt ####
    ##########################
    with open(os.path.join(vocab_path, 'template_labels.txt'), 'w') as f:
        for frame in frames:
            f.write(frame + '\n')

    ##################################
    #### non_span_slot_labels.txt ####
    ##################################
    # Only one line with 'none'
    with open(os.path.join(vocab_path, 'non_span_slot_labels.txt'), 'w') as f:
            f.write('none\n')
    
    ##################################
    #### non_padded_namespaces.txt ####
    ##################################
    with open(os.path.join(vocab_path, 'non_padded_namespaces.txt'), 'w') as f:
        f.write('*labels\n')
        f.write('*tags\n')

    ##################################
    #### event_arg_labels.txt ####
    ##################################
    # Only one line with 'none'
    with open(os.path.join(vocab_path, 'event_arg_labels.txt'), 'w') as f:
            f.write('none\n')

    ##################################
    #### .lock file ####
    ##################################
    # Create a file with an empty line
    with open(os.path.join(vocab_path, '.lock'), 'w') as f:
        f.write('\n')
         
         
if __name__ == "__main__":
    main()