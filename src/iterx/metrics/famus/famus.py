import logging
import os
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List

from allennlp.training.metrics import Metric
from overrides import overrides

from iterx.metrics.famus.gtt_eval_utils import add_normalized_templates, convert_docid, eval_tf, read_gold_templates

logger = logging.getLogger('FAMuS')


@Metric.register('famus')
class FAMuSMetric(Metric):
    def __init__(self, doc_path: Dict[str, str]):
        self.predicted_templates = defaultdict(OrderedDict)
        self.gold_templates = defaultdict(OrderedDict)
        for pred_path, ref_path in doc_path.items():
            if not os.path.exists(ref_path):
                logger.error(f"FAMuS reference file {ref_path} doesn't exist")
                exit(1)
            else:
                self.gold_templates[pred_path] = read_gold_templates(ref_path)
            self.predicted_templates[pred_path] = OrderedDict({k: [] for k in self.gold_templates[pred_path]})

    def reset(self) -> None:
        self.predicted_templates: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            pred_path: OrderedDict({doc_id: [] for doc_id in gold_templates})
            for pred_path, gold_templates in self.gold_templates.items()
        }

    @overrides
    def __call__(self, predictions: Dict[str, Dict[str, List[Dict[str, Any]]]]):
        data_path, predicted_templates = predictions
        for doc_id, preds in predicted_templates.items():
            converted_doc_id = convert_docid(doc_id)
            if len(preds) > 0:
                add_normalized_templates(preds, self.predicted_templates[data_path][converted_doc_id])

    def get_metric(self, reset: bool):
        if not reset or len(self.predicted_templates) == 0:
            return dict()
        all_scores = dict()
        for pred_file, gold_file in zip(self.predicted_templates, self.gold_templates):
            pred_templates: OrderedDict = self.predicted_templates[pred_file]
            gold_templates: OrderedDict = self.gold_templates[gold_file]
            all_scores[pred_file] = eval_tf(pred_templates, gold_templates, docids=[])
        self.reset()
        if len(all_scores) == 0:
            return dict()
        elif len(all_scores) == 1:
            famus_score = list(all_scores.values())[0]
            output_score = {
                'famus_template_type_p': famus_score['incident_type']['p'],
                'famus_template_type_r': famus_score['incident_type']['r'],
                'famus_template_type_f1': famus_score['incident_type']['f1'],
                'famus_slot_micro_p': famus_score['micro_avg']['p'],
                'famus_slot_micro_r': famus_score['micro_avg']['r'],
                'famus_slot_micro_f1': famus_score['micro_avg']['f1']
            }
            return output_score
        else:
            # TODO(@Will): may implement this if we ever actually need to
            # validate against something other than the dev split.
            raise NotImplementedError("More than one file detected for FAMuS validation"
                                      "This is currently unsupported.")
