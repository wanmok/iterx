"""
Adopted from https://github.com/allenai/SciREX/blob/master/scirex/evaluation_scripts/scirex_relation_evaluate.py
"""
import json
import os
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Any

import pandas as pd
from allennlp.training.metrics import Metric
from overrides import overrides

BASEPATH = os.getenv("RESULT_EXTRACTION_BASEPATH", "..")

available_entity_types_sciERC = ["Material", "Metric", "Task", "Generic", "OtherScientificTerm", "Method"]
map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
map_true_entity_to_available = {v: k for k, v in map_available_entity_to_true.items()}

used_entities = list(map_available_entity_to_true.keys())
true_entities = list(map_available_entity_to_true.values())


def has_all_mentions(doc, relation):
    has_mentions = all(len(doc["coref"][x[1]]) > 0 for x in relation)
    return has_mentions


def compute_mapping(predicted_relations: List[Dict[str, str]],
                    gold_entities: Dict[str, List],
                    doc_tokens: List[str]):
    """
    Each relation in predicted_relations is a dict with two elements (for binary relation). e.g.
    {
        'Metric': 'accuracy',
        'Task': 'Natural language inference',
    }
    """
    # make a copy, so we don't alter the original data
    gold_entities = deepcopy(gold_entities)
    predicted_mentions = set([mention for relation in predicted_relations for mention in relation.values()])

    # Assign each mention to one gold entity.
    predicted_mention2gold_entity_name: Dict[str, str] = {}
    for predicted_mention in predicted_mentions:
        gold_entity_name_to_pop = None
        for gold_entity_name, gold_mention_spans in gold_entities.items():
            gold_mentions = {' '.join(doc_tokens[start_tok:end_tok]) for (start_tok, end_tok) in gold_mention_spans}
            if predicted_mention in gold_mentions:
                gold_entity_name_to_pop = gold_entity_name
                predicted_mention2gold_entity_name[predicted_mention] = gold_entity_name
                break
        # Make sure each gold entity is only assigned once.
        if gold_entity_name_to_pop is not None:
            gold_entities.pop(gold_entity_name_to_pop)
        else:
            print(f"Cannot find span for {predicted_mention}")

    return predicted_mention2gold_entity_name


def scirex_eval(predicted_relations, gold_data, cardinality: int):
    all_metrics = []

    for types in combinations(used_entities, cardinality):
        for doc in gold_data:
            relations = predicted_relations[doc["doc_id"]]

            mapping = compute_mapping(relations, doc['coref'], doc["words"])

            for relation in relations:
                for entity_type, entity_name in relation.items():
                    relation[entity_type] = mapping.get(entity_name, entity_name)

            # each iteration only evaluate those of corresponding types
            relations = set([tuple((t, x[t]) for t in types) for x in relations if all(t in x.keys() for t in types)])

            gold_relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]
            gold_relations = set([x for x in gold_relations if has_all_mentions(doc, x)])

            matched = relations & gold_relations

            metrics = {
                "p": len(matched) / (len(relations) + 1e-7),
                "r": len(matched) / (len(gold_relations) + 1e-7),
            }
            metrics["f1"] = 2 * metrics["p"] * metrics["r"] / (metrics["p"] + metrics["r"] + 1e-7)

            if len(gold_relations) > 0:
                all_metrics.append(metrics)

    all_metrics = pd.DataFrame(all_metrics)
    # print("Relation Metrics n=2")
    # print(all_metrics.describe().loc['mean'][['p', 'r', 'f1']])

    # take the mean value
    return all_metrics.describe().loc['mean'][['p', 'r', 'f1']].to_dict()


@Metric.register('tempgen_scirex')
class TempGenSciREXMetric(Metric):
    def __init__(self,
                 doc_path: Dict[str, str],
                 cardinality: int = 2, ):
        super(TempGenSciREXMetric, self).__init__()

        self.cardinality = cardinality

        self.references: Dict[str, Dict[str, Any]] = {}
        for input_data_path, ref_data_path in doc_path.items():
            with open(ref_data_path, 'r') as f:
                self.references[input_data_path] = json.load(f)
        self.predictions: Dict[str, Dict[str, List[Dict[str, str]]]] = {
            input_data_path: {} for input_data_path in doc_path.keys()
        }

    @overrides
    def __call__(self, pred_data_path: str, predictions: Dict[str, List[Dict[str, str]]]):
        for doc_id, preds in predictions.items():
            self.references[pred_data_path][doc_id].extend(preds)

    def get_metric(self, reset: bool):
        results: Dict[str, Dict[str, float]] = {}
        for input_data_path, ref_docs in self.references.items():
            predictions = self.predictions[input_data_path]
            results[input_data_path] = scirex_eval(predictions, ref_docs, self.cardinality)

        if reset:
            self.reset()

        return {
            (f'{os.path.basename(input_data_path)}_{metric}' if len(results) > 1 else metric): value
            for input_data_path, results in results.items()
            for metric, value in results.items()
        }

    def reset(self) -> None:
        self.predictions = {
            input_data_path: {} for input_data_path in self.references.keys()
        }
