import logging
from collections import OrderedDict
from typing import Dict, Optional

from allennlp.training.metrics import Metric

from iterx.data.dataset.scirex.utils import read_scirex_relations_as_gtt_template
from iterx.metrics.muc.iterx_muc import IterXMUCMetric

logger = logging.getLogger('iterx_scirex')


def sanitize_dataset_path(dataset_path: str) -> str:
    if ';' in dataset_path:
        return dataset_path.split(';')[0]
    return dataset_path


@Metric.register("iterx_scirex")
class IterXSciREXMetric(IterXMUCMetric):
    def __init__(self,
                 doc_path: Optional[Dict[str, str]] = None,
                 ignore_no_template_doc: bool = False,
                 scorer_type: str = 'phi-3'):
        super(IterXSciREXMetric, self).__init__(doc_path=None,  # avoids using MUC to read docs
                                                convert_doc_id=False,
                                                ignore_no_template_doc=ignore_no_template_doc,
                                                sanitize_special_chars=False,
                                                scorer_type=scorer_type)
        if doc_path is not None:
            for src_file, ref_file in doc_path.items():
                src_file = sanitize_dataset_path(src_file)
                ref_file = sanitize_dataset_path(ref_file)
                self.references[src_file] = read_scirex_relations_as_gtt_template(ref_file)
                self.predictions[src_file] = OrderedDict()

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        muc_scores = super(IterXSciREXMetric, self).get_metric(reset=reset)
        return {
            (
                'iterx_scirex_p' if '_p' in k else (
                    'iterx_scirex_r' if '_r' in k else (
                        'iterx_scirex_f' if '_f' in k else k
                    )
                )
            ): v
            for k, v in muc_scores.items()
        }
