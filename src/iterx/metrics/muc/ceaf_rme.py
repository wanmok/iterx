import collections
import copy
import enum
from typing import OrderedDict, List, Union, Tuple, Optional, Callable

import numpy as np
from overrides import overrides
from scipy.optimize import linear_sum_assignment

# GTT
from iterx.metrics.conll_coref_scores_mod import CountSingletonsScorer
from iterx.metrics.muc.gtt_eval_utils import normalize_string

GTTEntity = List[str]
GTTSlot = List[GTTEntity]
GTTTemplate = OrderedDict[str, Union[str, GTTSlot]]
# IterX
IterXEntity = List[Tuple[str, str, str]]  # (template_type, slot_type, mention_str)
IterXTemplate = List[IterXEntity]


def generate_scoring_structures(
        doc_tmplt_list: OrderedDict[str, OrderedDict[int, GTTTemplate]]
) -> OrderedDict[str, List[IterXTemplate]]:
    new_doc_tmplt_list = collections.OrderedDict()
    for doc_id in doc_tmplt_list:
        new_doc_tmplt_list[doc_id] = []
        for idx_temp in range(len(doc_tmplt_list[doc_id])):
            tmplt_type: Optional[str] = None
            for role in doc_tmplt_list[doc_id][idx_temp]:
                if role == "incident_type":
                    tmplt_type = doc_tmplt_list[doc_id][idx_temp][role]
                    break
            assert tmplt_type is not None, 'Template type is None'
            template: IterXTemplate = []
            for role in doc_tmplt_list[doc_id][idx_temp]:
                if role == "incident_type":
                    continue
                entities: List[IterXEntity] = []
                for idx in range(len(doc_tmplt_list[doc_id][idx_temp][role])):
                    entity: IterXEntity = [
                        (tmplt_type, role, normalize_string(str(mention)))
                        for mention in doc_tmplt_list[doc_id][idx_temp][role][idx]
                    ]
                    entities.append(entity)
                template.extend(entities)

            new_doc_tmplt_list[doc_id].append(template)

    return new_doc_tmplt_list


def phi_for_clusters_rme(
        gold_clustering: List[IterXEntity],
        predicted_clustering: List[IterXEntity],
        one_to_one: bool = False
) -> Tuple[int, int]:
    def count_cluster_match(a, b):
        count_match = 0
        matched_cluster = set()
        has_seen = set()
        for c_i, c in enumerate(a):
            found = False
            for m in c:
                if found:
                    break
                for c2_i, c2 in enumerate(b):
                    if one_to_one and c2_i in has_seen:
                        continue

                    if m in c2:
                        found = True
                        matched_cluster.add(c2_i)
                        has_seen.add(c2_i)
                        break
            if found:
                count_match += 1
        return count_match, len(matched_cluster)

    return count_cluster_match(predicted_clustering, gold_clustering)


def phi_subset_for_clusters_rme(
        gold_clustering: List[IterXEntity],
        predicted_clustering: List[IterXEntity]
) -> Tuple[int, int]:
    return phi_for_clusters_rme(gold_clustering, predicted_clustering, one_to_one=False)


class CEAFRMEScorer(CountSingletonsScorer):
    scoring_subroutine: Callable[[List[IterXEntity], List[IterXEntity]], Tuple[int, int]]

    @classmethod
    def ordinary_ceafe(cls, clusters, gold_clusters, scoring_subroutine, output_raw=False):
        scoring_subroutine = scoring_subroutine or cls.scoring_subroutine
        assert scoring_subroutine is not None, "scoring_subroutine must be specified"

        # The following line has len(cluster) != 1 in the parent
        clusters = [cluster for cluster in clusters if len(cluster) >= 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        r_scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                scores[i, j], r_scores[i, j] = scoring_subroutine(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)

        # recall similarity is different w.r.t. CEAF-REE definition
        similarity = sum(scores[row, col])
        r_similarity = sum(r_scores[row, col])

        if output_raw:
            return similarity, len(clusters), r_similarity, len(gold_clusters), scores, r_scores, clusters, (row, col)
        return similarity, len(clusters), r_similarity, len(gold_clusters)

    @classmethod
    @overrides(check_signature=False)
    def ceafe(cls, clusters, gold_clusters, output_raw=False):
        multlabel_gold_cluster_ids = []
        multilabels = {}
        for gc_id, gc in enumerate(gold_clusters):
            for e in gc:
                if gc_id in multlabel_gold_cluster_ids:
                    break
                for m in e:
                    if '/' in m[0]:
                        multlabel_gold_cluster_ids.append(gc_id)
                        multilabels[gc_id] = m[0]
                        break

        if len(multlabel_gold_cluster_ids) > 0 and len(clusters) > 0:
            # Split multi-label clusters into single-label clusters
            split_gold_clusters = []
            orig_to_split_mappings = collections.defaultdict(list)
            for gc_id, gc in enumerate(gold_clusters):
                if gc_id not in multlabel_gold_cluster_ids:
                    split_gold_clusters.append(gc)
                    continue
                multilabel = multilabels[gc_id]
                labels = [s.strip() for s in multilabel.split('/')]
                new_splits = [
                    [
                        [
                            (label, m[1], m[2])
                            for m in e
                        ]
                        for e in copy.deepcopy(gc)
                    ]
                    for label in labels

                ]
                for new_split in new_splits:
                    split_gold_clusters.append(new_split)
                    orig_to_split_mappings[gc_id].append(len(split_gold_clusters) - 1)

            p_num, p_den, r_num, r_den, scores, r_scores, reduced_clusters, _ = cls.ordinary_ceafe(
                clusters, split_gold_clusters,
                scoring_subroutine=cls.scoring_subroutine,
                output_raw=True
            )
            # Only keep one of the split with the highest matching score
            for split_ids in orig_to_split_mappings.values():
                picked_id = split_ids[scores[split_ids].sum(axis=1).argmax().item()]
                for split_id in split_ids:
                    if split_id != picked_id:
                        scores[split_id] = 0
                        r_scores[split_id] = 0
            row, col = linear_sum_assignment(-scores)
            similarity = sum(scores[row, col])
            r_similarity = sum(r_scores[row, col])

            num_predicted_entities = sum([len(c) for c in reduced_clusters])
            num_gold_entities = sum([len(c) for c in gold_clusters])

            if output_raw:
                orig_row = []
                for i in row:
                    for k, v in orig_to_split_mappings.items():
                        if i in v:
                            orig_row.append(k)
                            break
                orig_row = np.array(orig_row)
                return (
                    similarity, num_predicted_entities, r_similarity, num_gold_entities,
                    scores, r_scores,
                    reduced_clusters, (orig_row, col)
                )
            return similarity, num_predicted_entities, r_similarity, num_gold_entities
        else:
            num_predicted_entities = sum([len(c) for c in clusters])
            num_gold_entities = sum([len(c) for c in gold_clusters])

            if output_raw:
                p_num, p_den, r_num, r_den, scores, r_scores, reduced_clusters, aligns = cls.ordinary_ceafe(
                    clusters, gold_clusters,
                    scoring_subroutine=cls.scoring_subroutine,
                    output_raw=output_raw
                )
                return (
                    p_num, num_predicted_entities, r_num, num_gold_entities, scores, r_scores, reduced_clusters, aligns
                )

            p_num, p_den, r_num, r_den = cls.ordinary_ceafe(clusters, gold_clusters,
                                                            scoring_subroutine=cls.scoring_subroutine,
                                                            output_raw=output_raw)
            return p_num, num_predicted_entities, r_num, num_gold_entities


class CEAFRMEPhiSubsetScorer(CEAFRMEScorer):
    scoring_subroutine = phi_subset_for_clusters_rme


def phi3_for_clusters_rme(
        gold_clustering: List[IterXEntity],
        predicted_clustering: List[IterXEntity]
) -> float:
    def _score_sub(a, b):
        return len([mention for mention in b if mention in a])

    def count_cluster_match(a, b):
        clusters = [cluster for cluster in a if len(a) >= 1]
        scores = np.zeros((len(b), len(clusters)))
        for i, gold_cluster in enumerate(b):
            for j, cluster in enumerate(a):
                scores[i, j] = _score_sub(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)

        similarity = sum(scores[row, col])
        # r_similarity = len([m for c in b for e in c for m in e])
        # p_similarity = len([m for c in a for e in c for m in e])

        return similarity  # , r_similarity, p_similarity

    return count_cluster_match(predicted_clustering, gold_clustering)


class CEAFRMEPhi3Scorer(CountSingletonsScorer):
    scoring_subroutine: Callable[[List[IterXEntity], List[IterXEntity]], float] = phi3_for_clusters_rme

    @classmethod
    def ordinary_ceafe(cls, clusters, gold_clusters, scoring_subroutine, output_raw=False):
        scoring_subroutine = scoring_subroutine or cls.scoring_subroutine
        assert scoring_subroutine is not None, "scoring_subroutine must be specified"

        # The following line has len(cluster) != 1 in the parent
        clusters = [cluster for cluster in clusters if len(cluster) >= 1]
        scores = np.zeros((len(gold_clusters), len(clusters)))
        # r_scores = np.zeros((len(gold_clusters), len(clusters)))
        # p_scores = np.zeros((len(gold_clusters), len(clusters)))
        for i, gold_cluster in enumerate(gold_clusters):
            for j, cluster in enumerate(clusters):
                # scores[i, j], r_scores[i, j], p_scores[i, j] = scoring_subroutine(gold_cluster, cluster)
                scores[i, j] = scoring_subroutine(gold_cluster, cluster)
        row, col = linear_sum_assignment(-scores)

        # recall similarity is different w.r.t. CEAF-REE definition
        similarity = sum(scores[row, col])
        r_similarity = len([m for c in gold_clusters for e in c for m in e])
        p_similarity = len([m for c in clusters for e in c for m in e])
        # p_similarity = len([m for c in clusters for e in clusters for m in e])

        if output_raw:
            return similarity, p_similarity, similarity, r_similarity, scores, None, clusters, (row, col)
        return similarity, p_similarity, similarity, r_similarity

    @classmethod
    @overrides(check_signature=False)
    def ceafe(cls, clusters, gold_clusters, output_raw=False):
        multlabel_gold_cluster_ids = []
        multilabels = {}
        for gc_id, gc in enumerate(gold_clusters):
            for e in gc:
                if gc_id in multlabel_gold_cluster_ids:
                    break
                for m in e:
                    if '/' in m[0]:
                        multlabel_gold_cluster_ids.append(gc_id)
                        multilabels[gc_id] = m[0]
                        break

        if len(multlabel_gold_cluster_ids) > 0 and len(clusters) > 0:
            # Split multi-label clusters into single-label clusters
            split_gold_clusters = []
            orig_to_split_mappings = collections.defaultdict(list)
            for gc_id, gc in enumerate(gold_clusters):
                if gc_id not in multlabel_gold_cluster_ids:
                    split_gold_clusters.append(gc)
                    continue
                multilabel = multilabels[gc_id]
                labels = [s.strip() for s in multilabel.split('/')]
                new_splits = [
                    [
                        [
                            (label, m[1], m[2])
                            for m in e
                        ]
                        for e in copy.deepcopy(gc)
                    ]
                    for label in labels

                ]
                for new_split in new_splits:
                    split_gold_clusters.append(new_split)
                    orig_to_split_mappings[gc_id].append(len(split_gold_clusters) - 1)

            p_num, p_den, r_num, r_den, scores, _, reduced_clusters, _ = cls.ordinary_ceafe(
                clusters, split_gold_clusters,
                scoring_subroutine=cls.scoring_subroutine,
                output_raw=True
            )
            # Only keep one of the split with the highest matching score
            for split_ids in orig_to_split_mappings.values():
                picked_id = split_ids[scores[split_ids].sum(axis=1).argmax().item()]
                for split_id in split_ids:
                    if split_id != picked_id:
                        scores[split_id] = 0
                        # r_scores[split_id] = 0
            row, col = linear_sum_assignment(-scores)
            similarity = sum(scores[row, col])

            num_predicted_entities = p_den
            num_gold_entities = r_den

            if output_raw:
                orig_row = []
                for i in row:
                    for k, v in orig_to_split_mappings.items():
                        if i in v:
                            orig_row.append(k)
                            break
                orig_row = np.array(orig_row)
                return (
                    similarity, num_predicted_entities, similarity, num_gold_entities,
                    scores, None,
                    reduced_clusters, (orig_row, col)
                )
            return similarity, num_predicted_entities, similarity, num_gold_entities
        else:
            # num_predicted_entities = sum([len(c) for c in clusters])
            # num_gold_entities = sum([len(c) for c in gold_clusters])

            if output_raw:
                p_num, p_den, r_num, r_den, scores, r_scores, reduced_clusters, aligns = cls.ordinary_ceafe(
                    clusters, gold_clusters,
                    scoring_subroutine=cls.scoring_subroutine,
                    output_raw=output_raw
                )
                return (
                    p_num, p_den, r_num, r_den, scores, r_scores, reduced_clusters, aligns
                )

            p_num, p_den, r_num, r_den = cls.ordinary_ceafe(clusters, gold_clusters,
                                                            scoring_subroutine=cls.scoring_subroutine,
                                                            output_raw=output_raw)
            return p_num, p_den, r_num, r_den


class ScoreFunction(str, enum.Enum):
    PhiSubset = 'phi-subset'
    Phi3 = 'phi-3'


SCORER_CONSTRUCTOR = {
    ScoreFunction.PhiSubset: CEAFRMEPhiSubsetScorer,
    ScoreFunction.Phi3: CEAFRMEPhi3Scorer,
}
