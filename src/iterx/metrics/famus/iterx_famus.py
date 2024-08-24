import logging
import os
import json
import pandas as pd
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from allennlp.training.metrics import Metric
from overrides import overrides

from iterx.metrics.famus.gtt_eval_utils import read_gold_templates, add_normalized_templates
from iterx.metrics.famus.ceaf_rme import generate_scoring_structures, IterXTemplate, SCORER_CONSTRUCTOR

logger = logging.getLogger('iterx_famus')


@Metric.register('iterx_famus')
class IterXFAMuSMetric(Metric):
    def __init__(self,
                 doc_path: Optional[Dict[str, str]] = None,
                 convert_doc_id: bool = False,
                 ignore_no_template_doc: bool = False,
                 sanitize_special_chars: bool = False,
                 scorer_type: str = 'phi-3'):
        self.scorer_constructor = SCORER_CONSTRUCTOR[scorer_type]

        self.ignore_no_template_doc = ignore_no_template_doc
        self.ceafe = self.scorer_constructor(self.scorer_constructor.ceafe)
        self.references = {}
        self.predictions = {}
        if doc_path is not None:
            if len(doc_path) > 1:
                logger.warning('The current implementation might give problematic result given more than one doc.')

            for src_file, ref_file in doc_path.items():
                assert os.path.isfile(src_file), f"Could not locate source file {src_file}!"
                assert os.path.isfile(ref_file), f"Could not locate reference file {ref_file}!"
                self.references[src_file] = read_gold_templates(ref_file,
                                                                convert_doc_id=convert_doc_id,
                                                                sanitize_special_chars=sanitize_special_chars)
                self.predictions[src_file] = OrderedDict()

    def reset(self) -> None:
        self.ceafe = self.scorer_constructor(self.scorer_constructor.ceafe)
        self.predictions = {src_file: OrderedDict() for src_file in self.predictions}

    @overrides
    def __call__(self,
                 predictions: Dict[str, Any],
                 pred_src_file: str,
                 dedup: bool = True,
                 cluster_substr: bool = True,
                 normalize_role: bool = True) -> None:
        for doc_id, templates in predictions.items():
            if doc_id not in self.predictions[pred_src_file]:
                self.predictions[pred_src_file][doc_id] = []
            add_normalized_templates(templates,
                                     self.predictions[pred_src_file][doc_id],
                                     dedup=dedup,
                                     cluster_substr=cluster_substr,
                                     normalize_role=normalize_role)

    @staticmethod
    def prepare_scoring_clusters(
            clusters: List[IterXTemplate]
    ) -> Tuple[List[Tuple], Dict[Tuple, List]]:
        scoring_clusters = [tuple(tuple(m) for m in e) for c in clusters for e in c]
        mention_to_cluster = {m: c for c in scoring_clusters for e in c for m in e}
        return scoring_clusters, mention_to_cluster

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        if not reset or len(self.predictions) == 0:
            return dict()

        scoring_preds: OrderedDict[str, List[IterXTemplate]] = OrderedDict()
        scoring_golds: OrderedDict[str, List[IterXTemplate]] = OrderedDict()
        for doc in self.predictions:
            assert doc in self.references, f"source document {doc} not found in reference documents!"
            scoring_preds |= generate_scoring_structures(self.predictions[doc])
            scoring_golds |= generate_scoring_structures(self.references[doc])

        all_doc_ids: List[str] = list(set(list(scoring_preds.keys()) + list(scoring_golds.keys())))
        for doc_id in all_doc_ids:
            pred = scoring_preds.get(doc_id, [])
            gold = scoring_golds.get(doc_id, [])

            if self.ignore_no_template_doc and len(gold) == 0:
                continue

            self.ceafe.update(predicted=pred,
                              gold=gold,
                              mention_to_predicted=None,
                              mention_to_gold=None)

        precision = self.ceafe.get_precision()
        recall = self.ceafe.get_recall()
        f1 = self.ceafe.get_f1()

        if reset:
            self.reset()

        return {
            "iterx_famus_slot_p": precision,
            "iterx_famus_slot_r": recall,
            "iterx_famus_slot_f1": f1
        }

# Utility functions for FAMuS metric
def compute_ceafe_rme_scores(gold_file, predictions,
                             ignore_no_template_doc=False ,
                             sanitize_special_chars= False,
                             scorer_type = 'phi-3-levenshtein'):    
    # Soft Match
    iterx_famus = IterXFAMuSMetric({gold_file: gold_file},
                                   scorer_type = scorer_type,
                                   ignore_no_template_doc = ignore_no_template_doc,
                                   sanitize_special_chars = sanitize_special_chars)
    iterx_famus(predictions,
                gold_file,
                normalize_role = False)
    return iterx_famus.get_metric(reset=True)['iterx_famus_slot_f1']

def convert_gold_iterx_dict_to_jsonl_file(gold_data_dict, 
                                          output_file="temp_gold_data_can_be_deleted.jsonl"):
    """
    This is a helper function to convert the gold data dict to a jsonl file
    This is needed since the FAMuS metric expects the gold data in a jsonl file format
    """
    with open(output_file, 'w') as f:
        for instance_id, template_list in gold_data_dict.items():
            current_dict = {}
            current_dict['docid'] = instance_id
            current_dict['templates'] = template_list
            f.write(json.dumps(current_dict) + "\n")


def out_compute_ceafe_rme_scores(gold_predictions: Union[str, Dict],
                                 predictions,
                                ignore_no_template_doc =False ,
                                sanitize_special_chars= False,
                                metrics = ('CEAF_RME_phi-3', 'CEAF_RME_phi-a')):
                                
    # If gold_predictions is a iterx formatted jsonl file itself, process it as it is
    if isinstance(gold_predictions, str):
        temp_gold_file = gold_predictions
    # Else convert the gold_predictions to a jsonl file
    else:
        temp_gold_file = 'gold.jsonl'
        convert_gold_iterx_dict_to_jsonl_file(gold_predictions, temp_gold_file)
    # Exact Match
    iterx_famus = IterXFAMuSMetric({temp_gold_file: temp_gold_file},
                               scorer_type = 'phi-3',
                               ignore_no_template_doc = ignore_no_template_doc,
                               sanitize_special_chars = sanitize_special_chars)
    iterx_famus(predictions, 
            temp_gold_file,
            normalize_role = False)
    
    exact_match_dict = iterx_famus.get_metric(reset=True)

    # Soft Match
    iterx_famus = IterXFAMuSMetric({temp_gold_file: temp_gold_file},
                                   scorer_type = 'phi-3-levenshtein',
                                   ignore_no_template_doc = ignore_no_template_doc,
                                   sanitize_special_chars = sanitize_special_chars)
    iterx_famus(predictions,
                temp_gold_file,
                normalize_role = False)
    soft_match_dict = iterx_famus.get_metric(reset=True)


    # Get a dataframe
    metric_rows = []

    if 'CEAF_RME_phi-3' in metrics:
        current_row = {'metric': 'CEAF_RME_phi-3',
         'P': round(100*exact_match_dict['iterx_famus_slot_p'],2),
         'R': round(100*exact_match_dict['iterx_famus_slot_r'],2),
         'F1': round(100*exact_match_dict['iterx_famus_slot_f1'],2)
         }
        metric_rows.append(current_row)

    if 'CEAF_RME_phi-a' in metrics:
        current_row = {'metric': 'CEAF_RME_phi-a',
         'P': round(100*soft_match_dict['iterx_famus_slot_p'],2),
         'R': round(100*soft_match_dict['iterx_famus_slot_r'],2),
         'F1': round(100*soft_match_dict['iterx_famus_slot_f1'],2)
         }
        metric_rows.append(current_row)
        
    
    return pd.DataFrame(metric_rows)