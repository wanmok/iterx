import json
import logging
from itertools import groupby
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import jsonlines
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer, Vocabulary
from allennlp.data.fields import FlagField, TextField, LabelField, ListField, MetadataField, SequenceLabelField, \
    SpanField
from allennlp.data.token_indexers import SingleIdTokenIndexer

TEMPLATE = Dict[str, Dict[str, Any]]

logger = logging.getLogger(__name__)


@DatasetReader.register("famus")
class FAMUSDataset(DatasetReader):
    def __init__(self,
                 definition_file: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 is_training: bool = True,
                 skip_docs_without_templates: bool = False,
                 skip_docs_without_spans: bool = True,
                 max_instances: int = None,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        with open(definition_file) as f:
            self.definitions = json.load(f)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # Since gold templates are always included (for scoring purposes),
        # the `is_training` field is currently unused, though we may decide
        # we want it later.
        self.is_training = is_training
        self.skip_docs_without_templates = skip_docs_without_templates
        self.skip_docs_without_spans = skip_docs_without_spans
        self.max_instances = max_instances
        self.verbose = verbose

    def apply_token_indexers(self, instance: Instance) -> None:
        if 'text' in instance:
            instance['text'].token_indexers = self.token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        total_entries = 0
        docs_without_spans = 0
        docs_without_templates = 0
        docs_with_spans_and_templates = 0
        with jsonlines.open(file_path) as f:
            for entry in f:
                has_spans = True
                has_templates = True
                total_entries += 1
                # Are we using predicted or gold spans?
                if "all-pred-spans" in entry:
                    spans_key = "all-pred-spans"
                else:
                    spans_key = "all-spans"

                if not entry[spans_key]:
                    has_spans = False
                    docs_without_spans += 1
                if not entry["templates"]:
                    has_templates = False
                    docs_without_templates += 1
                if has_spans and has_templates:
                    docs_with_spans_and_templates += 1
                else:
                    if self.verbose:
                        warn_msg = f"Document {entry['docid']} {'DOES' if has_spans else 'DOES NOT'} have spans " \
                                   f"and {'DOES' if has_templates else 'DOES NOT'} have templates"
                        logger.warn(warn_msg)
                if (not has_spans and self.skip_docs_without_spans) or \
                        (not has_templates and self.skip_docs_without_templates):
                    if self.verbose:
                        logger.warn(f"Skipping document {entry['docid']}.")
                    continue

                # TODO(@Will): add template validator?
                # TODO(@Will): handle "attack / bombing" and "bombing / attack" template types
                templates = entry["templates"]
                templates_by_type: Dict[str, TEMPLATE] = {
                    template_type: [template for template in grouped_templates]
                    for template_type, grouped_templates in groupby(
                        sorted(templates,
                               key=lambda x: x["incident_type"]),
                        key=lambda x: x["incident_type"]
                    )
                }
                # Uncomment the below for loop if you want to iterate for all templates for each entry
                # for template_type in self.definitions["definitions"].keys():
                #     yield self.text_to_instance(entry, template_type, file_path, templates_by_type)
                
                # For cross-doc famus, we have exactly one template for each entry
                gold_template_type = entry["templates"][0]['incident_type']
                yield self.text_to_instance(entry, gold_template_type, file_path, templates_by_type)
                
        warn_msg = f"Read {total_entries} documents. " \
                   f"Of these, {docs_with_spans_and_templates} had both templates and spans. " \
                   f"{docs_without_templates} had no templates and {docs_without_spans} had no spans."
        logger.warn(warn_msg)

    def fetch_gold_slot_label_from_span_tuple(self,
                                              input_span_tuple,
                                              gold_template):
        """
        Given a span 5-tuple: (span_string, char_start_idx, char_end_idx, token_start_idx, token_end_idx)
            and the gold template, fetch the gold slot label for the given tuple
       
        """
        match_found = False
        for role_name, role_values in gold_template.items():
            # we skip the names that are not roles
            if role_name == 'incident_type' or role_name == 'template-spans':
                continue
            # for all roles, we loop over each span for each role
            for current_span in role_values:
                # assuming each there is exactly one coref for each span
                current_single_coref_mention = current_span[0]
                if current_single_coref_mention[:5] == input_span_tuple:
                    match_found = True
                    return current_single_coref_mention[-1]
        
        # return error if match wasn't found because we use this function only for tuples inside the gold template
        if match_found == False:
            raise ValueError("This span was not found in the gold template spans")
        


    def text_to_instance(self,
                         entry: Dict[str, Any],
                         template_type: str,
                         data_path: str,
                         templates_by_type: Optional[Dict[str, TEMPLATE]] = None
                         ) -> Instance:
        doc_id = entry["docid"]
        spans: List[SpanField] = []
        span_types: List[int] = []
        event_types: List[str] = []
        text_field = TextField([Token(t) for t in entry["doctext-tok"]])
        gold_templates = templates_by_type.get(template_type, {})

        # Are we using gold gold spans or predicted spans?
        # If the latter, the gold templates + spans will still
        # be present in the file for ease of scoring purposes
        # (e.g. we may want to compute mention F1 w.r.t the gold mentions,
        # even though the actual mentions used by the model *aren't* gold)
        if "all-pred-spans" in entry:
            pred_span_key = "all-pred-spans"
            assert "all-pred-span-sets" in entry
            assert "pred-spans-to-spanset" in entry
        else:
            pred_span_key = "all-spans"

        for span in entry[pred_span_key]:
            # span[0]: span text
            # span[1], span[2]: character start and end indices
            # span[3], span[4]: token start and end indices
            # span[5]: slot label
            spans.append(SpanField(span[3], span[4], text_field))
            # 0 denotes an entity-valued (rather than event-valued) span;
            # all spans in MUC are entities
            span_types.append(0)
            # since all spans are entities, they have no event type, so we use a dummy value;
            # ideally, event types would be made optional in the model
            event_types.append("none")

        template_type_field = LabelField(template_type, label_namespace="template_labels")
        is_training_field = FlagField(self.is_training)
        metadata = {
            "raw_doc_ref": entry,
            "data_path": data_path,
            "instance_id": f"{doc_id}:{template_type}",
            "span_ssid": entry["spans-to-spanset"]
        }
        if "pred-spans-to-spanset" in entry:
            metadata["pred_span_ssid"] = entry["pred-spans-to-spanset"]

        fields = {
            "metadata": MetadataField(metadata),
            "template_type": template_type_field,
            "is_training": is_training_field,
            "text": text_field
        }

        if len(spans) > 0:
            spans_field = ListField(spans)
            fields["spans"] = spans_field
            fields["span_types"] = SequenceLabelField(span_types, spans_field, label_namespace="span_type_labels")
            fields["event_types"] = SequenceLabelField(event_types, spans_field, label_namespace='event_types')

        # Add metadata for gold templates, if available
        # As written, this currently can *never* be None
        if gold_templates is not None:
            all_gold_template_span_labels: List[List[str]] = []
            all_gold_span_slot_sets: List[Set[Tuple[int, str]]] = []
            for template in gold_templates:
                gold_span_slot_set: Set[Tuple[int, str]] = set()
                gold_template_span_labels: List[str] = ["none"] * len(entry['all-spans'])
                for span_idx in template["template-spans"]:
                    # the 5th entry in all-spans is empty in FAMuS (it is only in templates field that it has values)
                    # the first five fields: span_string, char_start_idx, char_end_idx, token_start_idx, token_end_idx
                    string_char_token_idxs = entry["all-spans"][span_idx][:5]
                    span_label = self.fetch_gold_slot_label_from_span_tuple(string_char_token_idxs,
                                                               template)
                    
                    # span_label = entry["all-spans"][span_idx][5]

                    gold_span_slot_set.add((span_idx + 1, span_label))
                    gold_template_span_labels[span_idx] = span_label
                all_gold_template_span_labels.append(gold_template_span_labels)
                all_gold_span_slot_sets.append(gold_span_slot_set)
                # DEBUGG line:
                # print(f"gold_template_span_labels for this example in reader: {gold_template_span_labels}")

            gold_slot_spans: List[Dict[str, List[int]]] = [
                {
                    slot: [i - 1 for i, _ in spans]
                    for slot, spans in groupby(sorted(list(template), key=lambda x: x[1]), key=lambda x: x[1])
                }
                for template in all_gold_span_slot_sets
            ]

            if len(all_gold_template_span_labels) > 0 and len(entry["all-spans"]) > 0:
                gold_template_span_label_field = ListField([
                    ListField([
                        LabelField(label, label_namespace='slot_types')
                        for label in gold_template_span_labels
                    ])
                    for gold_template_span_labels in all_gold_template_span_labels
                ])
                fields["gold_template_span_labels"] = gold_template_span_label_field

            metadata['gold_span_slot_sets'] = all_gold_span_slot_sets
            metadata['gold_non_span_slot_sets'] = [set()] * len(all_gold_span_slot_sets)  # No non-span slots for MUC
            metadata['gold_templates'] = gold_templates
            metadata['gold_slot_spans'] = gold_slot_spans
            # DEBUGG line:
            # print(f"metadata['gold_templates'] for this example in reader: {gold_template_span_labels}")
        return Instance(fields)


if __name__ == "__main__":
    import tqdm

    vocab = Vocabulary.from_files("resources/data/famus/vocabulary")
    definition_file = "resources/data/famus/definitions.json"
    reader = FAMUSDataset(
        definition_file=definition_file,
        is_training=True
    )
    for entry in tqdm.tqdm(reader.read("resources/data/famus/preprocessed/tokenized/train.jsonl")):
        entry.index_fields(vocab)
        t = entry.as_tensor_dict()
