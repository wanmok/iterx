import copy
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

from allennlp.training.metrics import Metric
from overrides import overrides

from better.postprocessing.utils import replace_template_span_with_spanset, annotate_bpdoc_using_decoded_templates
from lib.bp import BPDocument
from score import score_granular

logger = logging.getLogger('BETTER')


@Metric.register('better')
class BETTERMetric(Metric):
    def __init__(self,
                 template_definitions: Union[Dict[str, Any], str],
                 doc_path: Dict[str, Dict[str, str]],
                 prediction_format: str = 'iter',
                 doc_path_short_names: Dict[str, str] = {},
                 return_per_file_scores: bool = False):
        self.prediction_format = 'iter'
        self.predictions: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

        if isinstance(template_definitions, str):
            with open(template_definitions, 'r') as f:
                self.template_definitions = json.load(f)
        else:
            self.template_definitions = template_definitions

        self.ori_docs: Dict[str, Dict[str, dict]] = dict()
        for data_path in doc_path:
            for p in doc_path[data_path].values():
                if not os.path.exists(p):
                    logger.error(f"BETTER doc path {p} doesn't exist")
                    exit(1)

            # The key `data_path` points to the data file that dataset reader reads.
            self.ori_docs[data_path] = {
                # The key below is either "empty" or "ref", and the values should be pointed
                # to the BETTER empty or ref documents.
                key: json.load(open(path)) for key, path in doc_path[data_path].items()
            }
        self.doc_path_short_names = doc_path_short_names
        self.return_per_file_scores = return_per_file_scores

    def reset(self) -> None:
        self.predictions: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    @overrides
    def __call__(self, predictions: Tuple[str, Dict[str, List[Dict[str, Any]]]], **kwargs):
        self.predictions: Dict[str, Dict[str, List[Dict[str, Any]]]]
        data_path, predicted_templates = predictions
        for entry_k, preds in predicted_templates.items():
            whole_key = f'{data_path}:{entry_k}'
            if whole_key not in self.predictions[data_path]:
                self.predictions[data_path][whole_key] = []
            self.predictions[data_path][whole_key].extend(preds)

    @staticmethod
    def create_empty_json_dict(ref_doc, to_remove=('granular-templates',), in_place=False):
        empty_doc = copy.deepcopy(ref_doc) if not in_place else ref_doc
        for entry_id in empty_doc['entries'].keys():
            for k in to_remove:
                empty_doc['entries'][entry_id]['annotation-sets']['basic-events'][k] = {}
        return empty_doc

    def get_metric(
            self, reset: bool
    ) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        if not reset or len(self.predictions) == 0:
            return dict()

        all_scores = dict()
        for data_path, predictions in self.predictions.items():
            empty_doc_dict = self.create_empty_json_dict(self.ori_docs[data_path]['empty'], in_place=True)
            ref_doc_dict = self.ori_docs[data_path]['ref']

            predictions: Dict[str, List[Dict[str, Any]]]
            decoded_templates: Dict[str, List[Dict[str, Any]]] = predictions

            mapped_decoded_templates: Dict[str, List[Dict[str, List[Dict[str, str]]]]] = {
                k: replace_template_span_with_spanset(v)
                for k, v in decoded_templates.items()
            }

            sys_doc: BPDocument = BPDocument.from_dict(empty_doc_dict, no_validation=True)
            annotate_bpdoc_using_decoded_templates(bp_doc=sys_doc,
                                                   decoded_templates=mapped_decoded_templates,
                                                   template_definitions=self.template_definitions)

            scores = score_granular(sys_doc,
                                    BPDocument.from_dict(ref_doc_dict, no_validation=True),
                                    no_validation=True)[0]
            all_scores[data_path] = {
                'template': scores.template_measures.f1,
                'slot': scores.slotmatch_measures.f1,
                'irrealis': scores.irrealis_measures.f1,
                'time-attachment': scores.time_attachment_measures.f1,
                'better': scores.combined_score,
                'score_struct': scores
            }

        self.reset()

        if len(all_scores) == 0:
            return dict()
        elif len(all_scores) == 1:
            better_score = list(all_scores.values())[0]
            logger.info(f'template: {better_score["template"]}, '
                        f'slot: {better_score["slot"]}, '
                        f'irrealis: {better_score["irrealis"]}, '
                        f'time-attachment: {better_score["time-attachment"]}, '
                        f'better: {better_score["better"]}')
            return {
                'template': better_score["template"],
                'slot': better_score["slot"],
                'irrealis': better_score["irrealis"],
                'time-attachment': better_score["time-attachment"],
                'better': better_score["better"],
            }
        else:
            aggr_score = None
            scores_dict = dict()
            for data_path, score in all_scores.items():
                file_score = score['score_struct']
                if self.return_per_file_scores:
                    file_name = self.doc_path_short_names.get(data_path) or data_path
                    per_file_scores = {
                        f'{file_name}-template': file_score.template_measures.f1,
                        f'{file_name}-slot': file_score.slotmatch_measures.f1,
                        f'{file_name}-irrealis': file_score.irrealis_measures.f1,
                        f'{file_name}-time-attachment': file_score.time_attachment_measures.f1,
                        f'{file_name}-better': file_score.combined_score
                    }
                    scores_dict.update(per_file_scores)
                    logger.info(f'{file_name}-template: {file_score.template_measures.f1}, '
                                f'{file_name}-slot: {file_score.slotmatch_measures.f1}, '
                                f'{file_name}-irrealis: {file_score.irrealis_measures.f1}, '
                                f'{file_name}-time-attachment: {file_score.time_attachment_measures.f1}, '
                                f'{file_name}-better: {file_score.combined_score}')
                if aggr_score is None:
                    aggr_score = copy.deepcopy(file_score)
                    continue
                aggr_score.update(file_score)
            logger.info(f'template: {aggr_score.template_measures.f1}, '
                        f'slot: {aggr_score.slotmatch_measures.f1}, '
                        f'irrealis: {aggr_score.irrealis_measures.f1}, '
                        f'time-attachment: {aggr_score.time_attachment_measures.f1}, '
                        f'better: {aggr_score.combined_score}')
            scores_dict.update({
                'template': aggr_score.template_measures.f1,
                'slot': aggr_score.slotmatch_measures.f1,
                'irrealis': aggr_score.irrealis_measures.f1,
                'time-attachment': aggr_score.time_attachment_measures.f1,
                'better': aggr_score.combined_score,
            })
            return scores_dict
