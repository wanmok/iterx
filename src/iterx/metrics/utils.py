import collections
from typing import Union, Dict


def scores_to_metric(scores: collections.Counter) -> Dict[str, Union[float, int]]:
    if (scores['true_positive'] + scores['false_positive']) == 0 and (
            scores['true_positive'] + scores['false_negative']) == 0:
        precision: float = 1.
        recall: float = 1.
        f1: float = 1.
    elif scores['true_positive'] == 0:
        precision: float = 0
        recall: float = 0
        f1: float = 0
    else:
        precision: float = scores['true_positive'] / (scores['true_positive'] + scores['false_positive'])
        recall: float = scores['true_positive'] / (scores['true_positive'] + scores['false_negative'])
        f1: float = 2 * precision * recall / (precision + recall)
    return {
        'precision': precision,
        'recall': recall,
        'fscore': f1
    }
