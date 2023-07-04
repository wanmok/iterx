import json
import re
import string
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Union

import itertools

"""
The eval.py script from this repo: https://github.com/xinyadu/gtt

Slightly modified to be able to process IL model outputs.
"""

tag2role = OrderedDict(
    {'incident_type': 'incident_type', 'perp_individual_id': "PerpInd", 'perp_organization_id': "PerpOrg",
     'phys_tgt_id': "Target", 'hum_tgt_name': "Victim", 'incident_instrument_id': "Weapon"})
role2uppercase = {
    'perpind': 'PerpInd',
    'perporg': 'PerpOrg',
    'target': 'Target',
    'victim': 'Victim',
    'weapon': 'Weapon'
}

DOCIDS_BY_NUMBER_OF_TEMPLATES = "resources/data/muc/docids_event_n.json"

GTTEntity = List[str]
GTTSlot = List[GTTEntity]
GTTTemplate = OrderedDict[str, Union[str, GTTSlot]]


def cluster_substrings(spans: Set[str]) -> List[List[str]]:
    """brain-dead coref method for clustering substrings together"""
    length_order_spans = sorted(spans, key=lambda x: len(x))
    clusters = {s: [s] for s in spans}
    for i, s1 in enumerate(length_order_spans):
        for j in range(i + 1, len(spans)):
            s2 = length_order_spans[j]
            if s1 in s2:
                clusters[s2].extend(clusters.pop(s1))
                break
    return list(clusters.values())


def add_normalized_templates(curr_pred_templates: List[Dict[str, Any]],
                             existing_pred_templates: OrderedDict[str, List[GTTTemplate]],
                             dedup: bool = True,
                             cluster_substr: bool = True,
                             normalize_role: bool = True):
    """Function for converting IL-style templates to GTT's format

    Our outputs are in Jsonlines format, with one line per (document, template type) pair.
    We therefore need to aggregate over template types to obtain all templates for a document.
    This function adds *one* line from a jsonlines file to a dictionary of aggregated templates.

    :param: curr_pred_templates: the templates from the current Jsonlines file
    :param: existing_pred_templates: all aggregated templates for all docs
    :param: dedup: whether to deduplicate multiple identical spans for the same role
    :param: cluster_substr: whether to apply a dumb span clustering heuristic based on subspans
    :param: normalize_role: whether to normalize the role names to uppercase

    :return: None (adds new templates directly to existing_pred_templates)
    """
    for pred_template in curr_pred_templates:
        template_norm = {'incident_type': pred_template['incident_type']}
        if normalize_role:
            to_iterate = role2uppercase.items()
        else:
            to_iterate = pred_template.keys()

        for iter_key in to_iterate:
            if normalize_role:
                role_raw, role_norm = iter_key
                # handle lowercased or normalized versions of role names
                role = role_norm if role_norm in pred_template else role_raw
            else:
                role = iter_key
                role_norm = iter_key

            if role in pred_template:
                if dedup:
                    # This assumes all fillers are singleton entities
                    pred_fillers = {normalize_string(filler[0]) for filler in pred_template[role]}
                    if cluster_substr:
                        pred_fillers = cluster_substrings(pred_fillers)
                    else:
                        pred_fillers = [[f] for f in pred_fillers]
                else:
                    # pred_fillers = [[filler[0]] for filler in pred_template[role]]
                    pred_fillers = pred_template[role]
            else:
                pred_fillers = []
            template_norm[role_norm] = pred_fillers
        existing_pred_templates.append(template_norm)


def jsonlines_to_gtt_templates(
        file_path: str,
        dedup: Optional[bool] = False,
        cluster_substr: Optional[bool] = False,
        normalize_role: Optional[bool] = False,
        iteration_breakdown: Optional[int] = None
) -> OrderedDict[str, List[GTTTemplate]]:
    """Converts Jsonlines-formatted file to a dictionary of GTT-style templates

    :param: file_path: the path to the jsonlines file

    :return: a dictionary of GTT-style templates, keyed on document ID
    """
    all_templates = OrderedDict()
    with open(file_path) as f:
        for line in f:
            line = json.loads(line)
            docid, templates = list(line.items())[0]
            if docid not in all_templates:
                all_templates[docid] = []
            if iteration_breakdown is None:
                templates_to_add = templates
            else:
                if iteration_breakdown >= len(templates):
                    templates_to_add = []
                else:
                    templates_to_add = [templates[iteration_breakdown]]
            add_normalized_templates(templates_to_add, all_templates[docid], dedup, cluster_substr,
                                     normalize_role=normalize_role)
    return all_templates


def normalize_string(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


def matching(c1, c2):
    # similarity: if c2 (pred) is subset of c1 (gold) return 1
    for m in c2:
        if m not in c1:
            return 0
    return 1


def is_valid_mapping(mapping):
    reverse_mapping = {}
    for k in mapping:
        v = mapping[k]
        if v not in reverse_mapping:
            reverse_mapping[v] = [k]
        else:
            reverse_mapping[v].append(k)

    for v in reverse_mapping:
        if v == -1:
            continue
        if len(reverse_mapping[v]) > 1:
            return False

    return True


def score(mapping, pred, gold):
    ex_result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    for key in all_keys:
        ex_result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}
    # if invalid mapping, return 0
    # if not is_valid_mapping(mapping):
    #     return ex_result

    mapped_temp_pred = []
    mapped_temp_gold = []
    for pred_temp_idx in mapping:
        gold_temp_idx = mapping[pred_temp_idx]
        if not isinstance(pred[pred_temp_idx]["incident_type"], str):
            pred[pred_temp_idx]["incident_type"] = "attack"
        if gold_temp_idx != -1 and pred[pred_temp_idx]["incident_type"] in gold[gold_temp_idx][
            "incident_type"]:  # attach vs attach / bombing
            mapped_temp_pred.append(pred_temp_idx)
            mapped_temp_gold.append(gold_temp_idx)
            pred_temp, gold_temp = pred[pred_temp_idx], gold[gold_temp_idx]

            # prec
            for role in pred_temp.keys():
                if role == "incident_type":
                    ex_result[role]["p_den"] += 1
                    ex_result[role]["p_num"] += 1
                    continue
                for entity_pred in pred_temp[role]:
                    ex_result[role]["p_den"] += 1
                    correct = False
                    for entity_gold in gold_temp[role]:
                        # import ipdb; ipdb.set_trace()
                        if matching(entity_gold, entity_pred):
                            correct = True
                    if correct:
                        ex_result[role]["p_num"] += 1

            # recall
            for role in gold_temp.keys():
                if role == "incident_type":
                    ex_result[role]["r_den"] += 1
                    ex_result[role]["r_num"] += 1
                    continue
                for entity_gold in gold_temp[role]:
                    ex_result[role]["r_den"] += 1
                    correct = False
                    for entity_pred in pred_temp[role]:
                        if matching(entity_gold, entity_pred):
                            correct = True
                    if correct:
                        ex_result[role]["r_num"] += 1

    # spurious
    for pred_temp_idx in range(len(pred)):
        pred_temp = pred[pred_temp_idx]
        if pred_temp_idx not in mapped_temp_pred:
            for role in pred_temp:
                if role == "incident_type":
                    ex_result[role]["p_den"] += 1
                    continue
                for entity_pred in pred_temp[role]:
                    ex_result[role]["p_den"] += 1

    # missing
    for gold_temp_idx in range(len(gold)):
        gold_temp = gold[gold_temp_idx]
        if gold_temp_idx not in mapped_temp_gold:
            for role in gold_temp:
                if role == "incident_type":
                    ex_result[role]["r_den"] += 1
                    continue
                for entity_gold in gold_temp[role]:
                    ex_result[role]["r_den"] += 1

    ex_result["micro_avg"]["p_num"] = sum(ex_result[role]["p_num"] for _, role in tag2role.items())
    ex_result["micro_avg"]["p_den"] = sum(ex_result[role]["p_den"] for _, role in tag2role.items())
    ex_result["micro_avg"]["r_num"] = sum(ex_result[role]["r_num"] for _, role in tag2role.items())
    ex_result["micro_avg"]["r_den"] = sum(ex_result[role]["r_den"] for _, role in tag2role.items())

    for key in all_keys:
        ex_result[key]["p"] = 0 if ex_result[key]["p_num"] == 0 else ex_result[key]["p_num"] / float(
            ex_result[key]["p_den"])
        ex_result[key]["r"] = 0 if ex_result[key]["r_num"] == 0 else ex_result[key]["r_num"] / float(
            ex_result[key]["r_den"])
        ex_result[key]["f1"] = f1(ex_result[key]["p_num"], ex_result[key]["p_den"], ex_result[key]["r_num"],
                                  ex_result[key]["r_den"])

    return ex_result


def eval_tf(preds, golds, docids=[]):
    # normalization mention strings
    for docid in preds:
        for idx_temp in range(len(preds[docid])):
            for role in preds[docid][idx_temp]:
                if role == "incident_type":
                    continue
                for idx in range(len(preds[docid][idx_temp][role])):
                    for idy in range(len(preds[docid][idx_temp][role][idx])):
                        preds[docid][idx_temp][role][idx][idy] = normalize_string(
                            preds[docid][idx_temp][role][idx][idy])
    for docid in golds:
        for idx_temp in range(len(golds[docid])):
            for role in golds[docid][idx_temp]:
                if role == "incident_type":
                    continue
                for idx in range(len(golds[docid][idx_temp][role])):
                    for idy in range(len(golds[docid][idx_temp][role][idx])):
                        golds[docid][idx_temp][role][idx][idy] = normalize_string(
                            golds[docid][idx_temp][role][idx][idy])

    # get eval results
    result = OrderedDict()
    all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
    for key in all_keys:
        result[key] = {"p_num": 0, "p_den": 0, "r_num": 0, "r_den": 0, "p": 0, "r": 0, "f1": 0}

    if not docids:
        for docid in golds:
            docids.append(docid)

    for docid in docids:
        pred = preds[docid]
        gold = golds[docid]
        K, V = list(range(len(pred))), list(range(len(gold)))
        V = V + [-1]
        init_maps = [dict(zip(K, p)) for p in itertools.product(V, repeat=len(K))]

        ex_best = None
        # map_best = None
        for mapping in init_maps:
            if not is_valid_mapping(mapping):
                continue
            ex_result = score(mapping, pred, gold)
            if ex_best is None:
                ex_best = ex_result
                # map_best = mapping
            elif ex_result["micro_avg"]["f1"] > ex_best["micro_avg"]["f1"]:
                ex_best = ex_result
                # map_best = mapping

        # sum for one docid
        for role in all_keys:
            if role == "micro_avg":
                continue
            result[role]["p_num"] += ex_best[role]["p_num"]
            result[role]["p_den"] += ex_best[role]["p_den"]
            result[role]["r_num"] += ex_best[role]["r_num"]
            result[role]["r_den"] += ex_best[role]["r_den"]

    # micro average
    result["micro_avg"]["p_num"] = sum(result[role]["p_num"] for _, role in tag2role.items())
    result["micro_avg"]["p_den"] = sum(result[role]["p_den"] for _, role in tag2role.items())
    result["micro_avg"]["r_num"] = sum(result[role]["r_num"] for _, role in tag2role.items())
    result["micro_avg"]["r_den"] = sum(result[role]["r_den"] for _, role in tag2role.items())

    for key in all_keys:
        result[key]["p"] = 0 if result[key]["p_num"] == 0 else result[key]["p_num"] / float(result[key]["p_den"])
        result[key]["r"] = 0 if result[key]["r_num"] == 0 else result[key]["r_num"] / float(result[key]["r_den"])
        result[key]["f1"] = f1(result[key]["p_num"], result[key]["p_den"], result[key]["r_num"], result[key]["r_den"])

    return result


def convert_docid(docid: str) -> str:
    return str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[-1]))


def read_gold_templates(file_path: str,
                        convert_doc_id: bool = True,
                        sanitize_special_chars: bool = False) -> OrderedDict:
    def _sanitize_special_char(raw_str: str) -> str:
        return (
            raw_str
            .replace('\'', '')
            .replace('"', '')
            .replace('\\', '')
            .replace('-', '')
        )

    process_mention_str = _sanitize_special_char if sanitize_special_chars else lambda x: x
    golds = OrderedDict()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            if convert_doc_id:
                docid = convert_docid(line["docid"])
            else:
                docid = line["docid"]
            templates_raw = line["templates"]
            templates = []
            for template_raw in templates_raw:
                template = OrderedDict()
                for role, value in template_raw.items():
                    if role == "incident_type":
                        template[role] = value
                    elif role == "template-spans":
                        continue
                    else:
                        template[role] = []
                        for entity_raw in value:
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(process_mention_str(mention_offset_pair[0]))
                            if entity:
                                template[role].append(entity)
                if template not in templates:
                    templates.append(template)
            golds[docid] = templates
    return golds


def load_pred_file(file_path: str) -> OrderedDict:
    preds = OrderedDict()

    with open(file_path, encoding="utf-8") as f:
        out_dict = json.load(f)
        for docid in out_dict:
            preds[docid] = out_dict[docid]["pred_templates"]

    return preds


def load_gold_file(file_path: str) -> OrderedDict:
    golds = OrderedDict()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            docid = str(int(line["docid"].split("-")[0][-1]) * 10000 + int(line["docid"].split("-")[-1]))
            templates_raw = line["templates"]
            templates = []
            for template_raw in templates_raw:
                template = OrderedDict()
                for role, value in template_raw.items():
                    if role == "incident_type":
                        template[role] = value
                    else:
                        template[role] = []
                        for entity_raw in value:
                            entity = []
                            for mention_offset_pair in entity_raw:
                                entity.append(mention_offset_pair[0])
                            if entity:
                                template[role].append(entity)
                if template not in templates:
                    templates.append(template)
            golds[docid] = templates

    return golds
