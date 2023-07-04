import enum
from collections import OrderedDict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Union, Any, List

from iterx.data.dataset.scirex.utils import read_scirex_relations_as_gtt_template
from iterx.metrics.muc.ceaf_rme import ScoreFunction, SCORER_CONSTRUCTOR, IterXTemplate, generate_scoring_structures
from iterx.metrics.muc.gtt_eval_utils import jsonlines_to_gtt_templates, read_gold_templates, load_pred_file
from iterx.metrics.muc.iterx_muc import IterXMUCMetric
from iterx.metrics.scirex.iterx_scirex import IterXSciREXMetric


class DatasetKind(str, enum.Enum):
    MUC = 'MUC'
    SciREX = 'SciREX'


class PredictionFileType(str, enum.Enum):
    IterX = 'IterX'
    GTT = 'GTT'


def _load_predictions(
        pred_file: Path,
        file_type: PredictionFileType = PredictionFileType.IterX,
        normalize_role: bool = False,
) -> OrderedDict:
    if file_type == PredictionFileType.IterX:
        return jsonlines_to_gtt_templates(str(pred_file), False, False,
                                          normalize_role=normalize_role)
    elif file_type == PredictionFileType.GTT:
        return load_pred_file(str(pred_file))
    else:
        raise NotImplementedError


def load_ref_file(
        ref_file: Path,
        dataset_kind: DatasetKind,
        sanitize_special_chars: bool = False,
) -> OrderedDict:
    if dataset_kind == DatasetKind.MUC:
        return read_gold_templates(str(ref_file), convert_doc_id=False,
                                   sanitize_special_chars=sanitize_special_chars)
    elif dataset_kind == DatasetKind.SciREX:
        return read_scirex_relations_as_gtt_template(str(ref_file))
    else:
        raise ValueError(f'Unknown dataset kind: {dataset_kind}')


def scirex_postprocessing_merge_mentions(
        predictions: OrderedDict,
) -> Dict:
    return {
        k: [
            {
                s: sv if isinstance(sv, str) else [list(set([m for c in sv for m in c]))]
                for s, sv in t.items()
            }
            for t in v
        ]
        for k, v in predictions.items()
    }


def load_metric(
        dataset_kind: DatasetKind,
        doc_path: Dict[str, str],
        ignore_no_template_doc: bool,
        sanitize_special_chars: bool,
        scorer_type: str
) -> Union[IterXSciREXMetric, IterXMUCMetric]:
    if dataset_kind == DatasetKind.MUC:
        return IterXMUCMetric(doc_path=doc_path,
                              # convert_doc_id=args.use_cornell_docids,
                              ignore_no_template_doc=ignore_no_template_doc,
                              sanitize_special_chars=sanitize_special_chars,
                              scorer_type=scorer_type)
    elif dataset_kind == DatasetKind.SciREX:
        return IterXSciREXMetric(doc_path=doc_path,
                                 ignore_no_template_doc=ignore_no_template_doc,
                                 scorer_type=scorer_type)
    else:
        raise ValueError(f'Unknown dataset kind: {dataset_kind}')


def load_predictions(
        pred_file: Path,
        dataset: DatasetKind,
        file_type: PredictionFileType = PredictionFileType.IterX,
        normalize_role: bool = False,
        scirex_merge_mentions: bool = True,
) -> Dict:
    raw_predictions = _load_predictions(pred_file=pred_file,
                                        file_type=file_type,
                                        normalize_role=normalize_role)
    if dataset == DatasetKind.SciREX:
        if scirex_merge_mentions:
            return scirex_postprocessing_merge_mentions(raw_predictions)
        else:
            return raw_predictions
    elif dataset == DatasetKind.MUC:
        return raw_predictions
    else:
        raise ValueError(f'Unknown dataset kind: {dataset}')


def load_references(
        ref_file: Path,
        dataset: DatasetKind,
        loader_configs: Dict[str, Any]
) -> OrderedDict:
    if dataset == DatasetKind.MUC:
        return read_gold_templates(file_path=str(ref_file),
                                   **loader_configs)
    elif dataset == DatasetKind.SciREX:
        return read_scirex_relations_as_gtt_template(str(ref_file))
    else:
        raise NotImplementedError


def print_prediction_comparison(
        scorer: ScoreFunction,
        preds: Dict,
        golds: Dict,
        ignore_no_template_doc: bool = False,
):
    scorer_constructor = SCORER_CONSTRUCTOR[scorer.value]
    scorer = scorer_constructor(scorer_constructor.ceafe)

    scoring_preds: OrderedDict[str, List[IterXTemplate]] = OrderedDict()
    scoring_golds: OrderedDict[str, List[IterXTemplate]] = OrderedDict()
    scoring_preds |= generate_scoring_structures(preds)
    scoring_golds |= generate_scoring_structures(golds)

    # for doc_id in sorted((set(golds.keys()) | set(preds.keys())), key=lambda x: int(x.split('-')[-1])):
    for doc_id in sorted((set(golds.keys()) | set(preds.keys()))):

        pred = [t for t in scoring_preds.get(doc_id, []) if len(t) > 0]
        gold = [t for t in scoring_golds.get(doc_id, []) if len(t) > 0]

        if ignore_no_template_doc and len(gold) == 0:
            continue

        p_num, p_den, r_num, r_den, scores, _, _, (row, col) = scorer.metric(pred, gold, output_raw=True)

        prec = p_num / p_den if p_den != 0 else 0
        recall = r_num / r_den if r_den != 0 else 0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0

        print(f'doc_id={doc_id}\t#pred={len(pred)}\t#gold={len(gold)}\tprec={prec:.4f}\trec={recall:.4f}\tf1={f1:.4f}')

        row = list(row)
        col = list(col)
        not_present_row = [i for i in range(len(gold)) if i not in row]
        not_present_col = [i for i in range(len(pred)) if i not in col]
        row = row + not_present_row + [-1 for _ in not_present_col]
        col = col + [-1 for _ in not_present_row] + not_present_col
        for row_idx, col_idx in zip_longest(row, col, fillvalue=-1):
            if row_idx == -1:
                print('\tPredicted but not matched:')
                print(f'\t\t{preds[doc_id][col_idx]}')
            elif col_idx == -1:
                print('\tNot predicted:')
                print(f'\t\t{golds[doc_id][row_idx]}')
            else:
                common_keys = set(preds[doc_id][col_idx].keys()) & set(golds[doc_id][row_idx].keys())
                common_keys.remove('incident_type')
                print(f'\tincident_type=\t{preds[doc_id][col_idx]["incident_type"]}'
                      f'\t{golds[doc_id][row_idx]["incident_type"]}')
                for key in common_keys:
                    print(f'\t{key}=')
                    for pred_val, gold_val in zip_longest(
                            sorted(preds[doc_id][col_idx][key], key=lambda x: -max([len(y) for y in x])),
                            sorted(golds[doc_id][row_idx][key], key=lambda x: -max([len(y) for y in x]))
                    ):
                        print(f'\t\t{pred_val}\t{gold_val}')
            print()
