from typing import List, Set, Tuple, Dict, Union

from allennlp.training.metrics import Metric
from overrides import overrides

from iterx.metrics.conll_coref_scores_mod import CountSingletonsScorer


@Metric.register('proxy_slot_match')
class ProxySlotMatchMetric(Metric):
    supports_distributed = True

    def __init__(self):
        self.ceafe = CountSingletonsScorer(CountSingletonsScorer.ceafe)

    def reset(self) -> None:
        self.ceafe = CountSingletonsScorer(CountSingletonsScorer.ceafe)

    @overrides
    def __call__(self,
                 gold_slot_sets: List[Set[Tuple[Union[str, int], str]]],
                 predicted_slot_sets: List[Set[Tuple[Union[str, int], str]]]) -> None:
        # The function doesn't consider batch
        predicted_clusters, mention_to_predicted_cluster = self.prepare_scoring_clusters(predicted_slot_sets)
        gold_clusters, mention_to_gold_cluster = self.prepare_scoring_clusters(gold_slot_sets)
        self.ceafe.update(
            predicted=predicted_clusters,
            gold=gold_clusters,
            mention_to_predicted=mention_to_predicted_cluster,
            mention_to_gold=mention_to_gold_cluster
        )

    @staticmethod
    def prepare_scoring_clusters(
            clusters: List[Set[Tuple[Union[str, int], str]]]
    ) -> Tuple[List[Tuple], Dict[Tuple, List]]:
        scoring_clusters = [tuple(tuple(m) for m in c) for c in clusters]
        mention_to_cluster = {m: c for c in scoring_clusters for m in c}
        return scoring_clusters, mention_to_cluster

    def get_metric(self, reset: bool = False) -> Tuple[float, float, float]:
        precision = self.ceafe.get_precision()
        recall = self.ceafe.get_recall()
        f1 = self.ceafe.get_f1()

        if reset:
            self.reset()

        return precision, recall, f1
