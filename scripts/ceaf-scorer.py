from pathlib import Path
from typing import Annotated

import typer

from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from iterx.metrics.muc.ceaf_rme import ScoreFunction

app = typer.Typer(help='CEAF-RME scorer.')


@app.command()
def score(
        pred_file: Path,
        ref_file: Path,
        dataset: Annotated[DatasetKind, typer.Option()],
        ignore_no_template_doc: Annotated[bool, typer.Option(
            help='Whether to ignore documents without any templates during scoring.'
        )] = False,
        sanitize_special_chars: Annotated[bool, typer.Option(
            help='Whether to sanitize special characters in the predictions and references.'
        )] = True,
        scirex_merge_mentions: Annotated[bool, typer.Option(
            help='Whether to merge mentions in the SciREX predictions to form entities.'
        )] = True,
        scorer: Annotated[ScoreFunction, typer.Option(
            help='The scoring function to use.'
        )] = ScoreFunction.Phi3,
        file_type: Annotated[PredictionFileType, typer.Option(
            help='The type of the prediction file.'
        )] = PredictionFileType.IterX,
):
    """Score a prediction file against a reference file."""
    normalize_role = True if dataset == DatasetKind.MUC else False
    # SciREX postprocessing should have been moved out of the scorer.
    predictions = load_predictions(pred_file=pred_file,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   scirex_merge_mentions=scirex_merge_mentions)
    metric = load_metric(dataset_kind=dataset,
                         doc_path={str(pred_file): str(ref_file)},
                         ignore_no_template_doc=ignore_no_template_doc,
                         sanitize_special_chars=sanitize_special_chars,
                         scorer_type=scorer)

    metric(
        predictions=predictions,
        pred_src_file=str(pred_file),
        dedup=False,
        cluster_substr=False,
        normalize_role=normalize_role
    )
    results = metric.get_metric(reset=True)
    for k, v in results.items():
        print(f"{k}: {round(v, 4)}")


@app.command()
def compare(
        pred_file: Path,
        ref_file: Path,
        dataset: Annotated[DatasetKind, typer.Option()],
        ignore_no_template_doc: Annotated[bool, typer.Option(
            help='Whether to ignore documents without any templates during scoring.'
        )] = False,
        sanitize_special_chars: Annotated[bool, typer.Option(
            help='Whether to sanitize special characters in the predictions and references.'
        )] = True,
        scirex_merge_mentions: Annotated[bool, typer.Option(
            help='Whether to merge mentions in the SciREX predictions to form entities.'
        )] = True,
        scorer: Annotated[ScoreFunction, typer.Option(
            help='The scoring function to use.'
        )] = ScoreFunction.Phi3,
        file_type: Annotated[PredictionFileType, typer.Option(
            help='The type of the prediction file.'
        )] = PredictionFileType.IterX,
):
    """Compare a prediction file against a reference file."""
    normalize_role = True if dataset == DatasetKind.MUC else False
    # SciREX postprocessing should have been moved out of the scorer.
    predictions = load_predictions(pred_file=pred_file,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   scirex_merge_mentions=scirex_merge_mentions)
    loader_configs = {}
    if dataset == DatasetKind.MUC:
        loader_configs['convert_doc_id'] = False
        loader_configs['sanitize_special_chars'] = sanitize_special_chars
    references = load_references(ref_file=ref_file,
                                 dataset=dataset,
                                 loader_configs=loader_configs)
    print_prediction_comparison(scorer=scorer,
                                preds=predictions,
                                golds=references,
                                ignore_no_template_doc=ignore_no_template_doc)


if __name__ == '__main__':
    app()
