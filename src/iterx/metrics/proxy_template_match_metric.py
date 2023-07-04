import collections
from typing import List, Tuple

from allennlp.training.metrics import Metric
from overrides import overrides

from iterx.metrics.utils import scores_to_metric


@Metric.register('proxy_template_match')
class ProxyTemplateMatchMetric(Metric):
    def __init__(self):
        self.gold_template_counter = collections.Counter()
        self.predicted_template_counter = collections.Counter()

    @overrides
    def __call__(self, gold_templates: List[Tuple[str, str]], predicted_templates: List[Tuple[str, str]]):
        self.gold_template_counter.update(gold_templates)
        self.predicted_template_counter.update(predicted_templates)

    def get_metric(self, reset: bool):
        true_positive = sum((self.gold_template_counter & self.predicted_template_counter).values())
        false_negative = sum(self.gold_template_counter.values()) - true_positive
        false_positive = sum(self.predicted_template_counter.values()) - true_positive

        scores = collections.Counter({
            'true_positive': true_positive,
            'false_negative': false_negative,
            'false_positive': false_positive,
        })

        metric = scores_to_metric(scores)

        if reset:
            self.reset()

        return {
            'tmplt-prec': metric['precision'],
            'tmplt-recall': metric['recall'],
            'tmplt-f1': metric['fscore'],
        }

    @overrides
    def reset(self) -> None:
        self.gold_template_counter = collections.Counter()
        self.predicted_template_counter = collections.Counter()
