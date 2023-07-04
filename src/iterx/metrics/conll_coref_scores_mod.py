import numpy as np
from overrides import overrides
from scipy.optimize import linear_sum_assignment

from iterx.metrics.conll_coref_scores import Scorer


class CountSingletonsScorer(Scorer):
    """
    Subclassed from the AllenNLP `Scorer` class for coreference scoring,
    for the sole purpose of being able to count singleton clusters when
    computing the CEAF-E metric
    """

    @staticmethod
    @overrides
    def ceafe(clusters, gold_clusters, scoring_subroutine=Scorer.phi4, output_raw=False):
        # The following line has len(cluster) != 1 in the parent
        clusters = [cluster for cluster in clusters if len(cluster) >= 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j] = scoring_subroutine(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)
        similarity = sum(scores[row, col])

        if output_raw:
            return similarity, len(clusters), similarity, len(gold_clusters), scores, clusters, (row, col)
        return similarity, len(clusters), similarity, len(gold_clusters)
