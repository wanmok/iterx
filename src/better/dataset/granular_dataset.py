import itertools
import json
import logging
from collections import defaultdict
from itertools import groupby
from typing import Dict, Any, Iterable, Tuple, List, Optional, Set, Union

import h5py
import numpy as np
import torch
from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.fields import TextField, FlagField, LabelField, ListField, SpanField, SequenceLabelField, \
    MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer

from better.utils.bpjson_utils import sentence_range, sent_char_to_token_range, build_spanset_mappings

logger = logging.getLogger(__name__)

TEMPLATE_NON_SLOT_KEYS = {'template-id', 'template-type', 'template-anchor'}

IRREALIS_OUTCOME_MAPPINGS = {
    'hypothetical': 'hypothetical',
    'counterfactual': 'averted',
}


@DatasetReader.register("iterative_filling_dataset")
class GranularDataset(DatasetReader):
    def __init__(self,
                 definition_file: str,
                 ignore_template_types: Optional[List[str]] = None,
                 add_self_loop: bool = False,
                 convert_outcome_slot: bool = True,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 is_training: bool = True,
                 max_instances: int = None,
                 gold_basic_file: Optional[str] = None,
                 **kwargs):
        super(GranularDataset, self).__init__(max_instances=max_instances, **kwargs)

        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.convert_outcome_slot = convert_outcome_slot
        self.add_self_loop = add_self_loop
        self.is_training = is_training

        # This is a hack to accommodate the case where we might have gold Basic
        # events for some (not not necessarily all) of the documents on which
        # we're running inference. We pre-load these annotations and then, when
        # reading, check whether we have gold Basic events for any of the documents
        # that we're reading in.
        self.gold_basic_file = gold_basic_file

        # Loads definition file
        with open(definition_file) as f:
            self.definitions = json.load(f)

        if self.gold_basic_file:
            data_desc, entry_stream = self._read_data_entry_stream(self.gold_basic_file)
            self.gold_basic_annotations = {k: v for k, v in entry_stream}
        else:
            self.gold_basic_annotations = {}

        self.ignore_template_types = ignore_template_types or []

        self.non_span_slots_by_template = {}
        self.allowed_values_for_non_span_slots = defaultdict(set)
        for temp_type, temp_def in self.definitions['definitions'].items():
            self.non_span_slots_by_template[temp_type] = []
            for slot_type, slot_def in temp_def['slots'].items():
                if ('spanset' not in slot_def['filler_types']) and ('event' not in slot_def['filler_types']):
                    self.non_span_slots_by_template[temp_type].append(slot_type)
                    self.allowed_values_for_non_span_slots[slot_type] = {'none'}
                if 'bool' in slot_def['filler_types']:
                    # an oddity of boolean slots is that they are either present
                    # with value 'true' or else not present at all ('none')
                    self.allowed_values_for_non_span_slots[slot_type] |= {'true'}
                for constraint in slot_def['constraints']:
                    if constraint['constraint_type'] == 'string_type':
                        self.allowed_values_for_non_span_slots[slot_type] |= set(constraint['allows'])

    @staticmethod
    def _access_cache_by_key(cache_handler, key: List[str]) -> Union[h5py.Group, np.ndarray]:
        t = cache_handler['/'.join(key)]
        return t if isinstance(t, h5py.Group) else np.array(t)

    @staticmethod
    def _read_data_entry_stream(file_path: str) -> Tuple[Dict[str, Any], Iterable[Tuple[str, Dict]]]:
        # Adopted from `slot_filling_dataset.py` - might exist duplicate code
        def _data_stream() -> Iterable[Tuple[str, Dict]]:
            for entry_k, entry in entries:
                raw_text = entry['segment-text']
                tokenized_text = entry['segment-text-tok']

                # Maps each token to start and end index in characters
                # NOTE: seemingly not actually needed anymore, omitting --Will
                # token_char_offsets = [(x[0], x[-1]) for x in entry['tok2char'] if x else ]

                # Reads annotation-sets from the document entry
                events = entry['annotation-sets']['basic-events'].get('events', {})
                includes_relations = entry['annotation-sets']['basic-events'].get('includes-relations', {})
                span_sets = entry['annotation-sets']['basic-events'].get('span-sets', {})
                templates = entry['annotation-sets']['basic-events'].get('granular-templates', {})

                # Computes section/sentence offsets
                section_sent_char_offsets = sentence_range(entry['segment-sections'])
                sent_token_range = sent_char_to_token_range(entry['char2tok'], section_sent_char_offsets)
                ssid_span_mappings, span_ssid_mappings, ssid_eventid_mappings = build_spanset_mappings(span_sets,
                                                                                                       events)

                entry_dict_to_yield = {
                    'raw_text': raw_text,
                    'tokenized_text': tokenized_text,
                    # 'token_char_offsets': token_char_offsets,
                    'events': events,
                    'span_sets': span_sets,
                    'templates': templates,
                    'ssid_span_mappings': ssid_span_mappings,
                    'span_ssid_mappings': span_ssid_mappings,
                    'ssid_eventid_mappings': ssid_eventid_mappings,
                    'sent_token_range': sent_token_range,
                    'includes_relations': includes_relations,
                    'raw_doc_ref': entry
                }

                yield entry_k, entry_dict_to_yield

        with open(file_path) as f:
            data = json.load(f)

        entries = sorted(data['entries'].items())
        desc = {
            'corpus-id': data['corpus-id']
        }
        return desc, _data_stream()

    def _read(self, file_path) -> Iterable[Instance]:
        def _validate_template(t) -> bool:
            for k, v in t.items():
                if k in TEMPLATE_NON_SLOT_KEYS:
                    continue
                if isinstance(v, list):
                    if len(v) > 0:
                        return True
            return False

        file_paths = file_path.split(';')
        data_path = file_paths[0]
        cache_handler: Optional[h5py.File] = None
        if len(file_paths) == 2:
            cache_path = file_paths[1]
            cache_handler = h5py.File(cache_path, 'r')

        data_desc, entry_stream = self._read_data_entry_stream(data_path)

        for entry_k, entry in entry_stream:
            if entry_k in self.gold_basic_annotations:
                # If we have pre-loaded the gold basic annotations for this document, use those
                entry = self.gold_basic_annotations[entry_k]
            raw_doc_ref = entry['raw_doc_ref']
            ssid_span_mappings = entry['ssid_span_mappings']
            span_ssid_mappings = entry['span_ssid_mappings']
            ssid_eventid_mappings = entry['ssid_eventid_mappings']
            templates = entry['templates']
            tokenized_text = entry['tokenized_text']
            events = entry.get('events')
            includes_relations = entry.get('includes_relations')

            if cache_handler is not None:
                # Shape: (num_layers, seq_len, embed_dim)
                sequence_tensor = self._access_cache_by_key(cache_handler, [entry_k, entry_k])
                assert sequence_tensor.shape[-2] == len(tokenized_text), (
                    f'Cache length mismatch: {sequence_tensor.shape[-2]} != {len(tokenized_text)}'
                )
            else:
                sequence_tensor = None

            templates_by_type: Dict[str, Dict[str, Dict[str, Any]]] = {
                template_type.lower(): {
                    template['template-id']: template
                    for template in grouped_templates
                    if _validate_template(template)
                }
                for template_type, grouped_templates in groupby(
                    sorted(templates.values(),
                           key=lambda x: x['template-type']),
                    key=lambda x: x['template-type']
                )
            }
            for template_type in self.definitions['definitions'].keys():
                # if template_type not in templates_by_type and self.is_training:
                #     continue
                if template_type in self.ignore_template_types:
                    continue

                instance = self.text_to_instance(
                    raw_doc_ref=raw_doc_ref,
                    data_path=data_path,
                    instance_id=f'{entry_k}:{template_type}',
                    text=tokenized_text,
                    template_type=template_type,
                    span_ssid_mappings=span_ssid_mappings,
                    ssid_span_mappings=ssid_span_mappings,
                    ssid_eventid_mappings=ssid_eventid_mappings,
                    events=events,
                    includes_relations=includes_relations,
                    sequence_tensor=sequence_tensor,
                    gold_templates=templates_by_type.get(template_type, {} if self.is_training else None),
                )
                if instance is not None:
                    yield instance

    def text_to_instance(self,
                         raw_doc_ref: Dict[str, Any],
                         data_path: str,
                         instance_id: str,
                         text: List[str],
                         template_type: str,
                         span_ssid_mappings: Dict[Tuple[int, int], Set[str]],
                         ssid_span_mappings: Dict[str, Set[Tuple[Tuple[int, int], str]]],
                         ssid_eventid_mappings: Dict[str, str],
                         events: Dict[str, Any],
                         includes_relations: Dict[str, Any],
                         sequence_tensor: Optional[np.ndarray] = None,
                         gold_templates: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Instance]:
        # GNN inputs
        entity_mention_node_list: List[int] = []
        event_mention_node_list: List[int] = []
        entity_ss_edges: List[Tuple[int, int]] = []
        event_ss_edges: List[Tuple[int, int]] = []
        event_ref_edges: List[Tuple[int, int]] = []
        entity_includes_edges: List[Tuple[int, int]] = []
        event_arg_edges: List[Tuple[int, int]] = []
        event_arg_attrs: List[str] = []

        span_idx_entity_mention_node_mapping: Dict[int, int] = {}
        entity_mention_node_span_idx_mapping: Dict[int, int] = {}

        span_idx_event_mention_node_mapping: Dict[int, int] = {}
        event_mention_node_span_idx_mapping: Dict[int, int] = {}

        # Build spans
        spans: List[Tuple[int, int]] = []
        span_ssid: List[str] = []
        span_types: List[int] = []
        event_types: List[str] = []
        ssid_span_idx: Dict[str, Set[int]] = defaultdict(set)
        span_idx_mapping: Dict[Tuple[str, Tuple[int, int]], int] = {}
        for span, ssids in span_ssid_mappings.items():
            for ssid in ssids:
                span_idx = len(spans)
                spans.append(span)
                span_ssid.append(ssid)
                ssid_span_idx[ssid].add(span_idx)
                if ssid.startswith('ss-'):
                    span_type = 0
                    event_type = '@@PADDING@@'
                    span_idx_mapping[('ss', span)] = span_idx
                    # GNN inputs
                    entity_mention_node_idx = len(entity_mention_node_list)
                    entity_mention_node_list.append(span_idx)
                    span_idx_entity_mention_node_mapping[span_idx] = entity_mention_node_idx
                    entity_mention_node_span_idx_mapping[entity_mention_node_idx] = span_idx
                elif ssid.startswith('event-'):
                    span_type = 1
                    event_type = events[ssid]['event-type'].lower()
                    span_idx_mapping[('event', span)] = span_idx
                    # GNN inputs
                    event_mention_node_idx = len(event_mention_node_list)
                    event_mention_node_list.append(span_idx)
                    span_idx_event_mention_node_mapping[span_idx] = event_mention_node_idx
                    event_mention_node_span_idx_mapping[event_mention_node_idx] = span_idx
                else:
                    raise ValueError(f'Unknown span type: {ssid}')
                span_types.append(span_type)
                event_types.append(event_type)

        # Spanset edges
        get_mention_node_idx = {
            'ss': lambda x: span_idx_entity_mention_node_mapping[x],
            'event': lambda x: span_idx_event_mention_node_mapping[x]
        }
        # Rebuild ssid ot spans mappings as `span_ssid_mappings` might contain duplicates for event-only spans
        sanitized_ssid_span_mappings: Dict[Tuple[str, str], Set[Tuple[int, int]]] = defaultdict(set)
        for span, ssids in span_ssid_mappings.items():
            for ssid in ssids:
                prefix = ssid.split('-')[0]
                sanitized_ssid_span_mappings[(prefix, ssid)].add(span)
        # Build edges for spanset graph - they're bidirectional as information could flow back and forth
        for (spanset_type, ssid), ss in sanitized_ssid_span_mappings.items():
            span_list = list(ss)
            for i in range(len(span_list)):
                src_span_idx = get_mention_node_idx[spanset_type](span_idx_mapping[(spanset_type, span_list[i])])
                if self.add_self_loop:
                    if spanset_type == 'ss':
                        entity_ss_edges.append((src_span_idx, src_span_idx))
                    elif spanset_type == 'event':
                        event_ss_edges.append((src_span_idx, src_span_idx))
                    else:
                        raise ValueError(f'Unknown span type: {spanset_type}')
                for j in range(i + 1, len(span_list)):
                    tgt_span_idx = get_mention_node_idx[spanset_type](span_idx_mapping[(spanset_type, span_list[j])])
                    if spanset_type == 'ss':
                        entity_ss_edges.append((src_span_idx, tgt_span_idx))
                        entity_ss_edges.append((tgt_span_idx, src_span_idx))
                    elif spanset_type == 'event':
                        event_ss_edges.append((src_span_idx, tgt_span_idx))
                        event_ss_edges.append((tgt_span_idx, src_span_idx))
                    else:
                        raise ValueError(f'Unknown prefix: {spanset_type}')
        # Build edges for event graph
        for event_id, event in events.items():
            event_mention_node_ids: List[int] = [
                get_mention_node_idx['event'](span_idx_mapping[('event', anchor_span)])
                for anchor_span in sanitized_ssid_span_mappings[('event', event_id)]
            ]
            agent_mention_node_ids: List[int] = [
                get_mention_node_idx['ss'](span_idx_mapping[('ss', mention_span)])
                for agent_ssid in event['agents']
                for mention_span in sanitized_ssid_span_mappings[('ss', agent_ssid)]
            ]
            patient_mention_node_ids: List[int] = [
                get_mention_node_idx['ss'](span_idx_mapping[('ss', mention_span)])
                for patient_ssid in event['patients']
                for mention_span in sanitized_ssid_span_mappings[('ss', patient_ssid)]
            ]
            for e in itertools.product(event_mention_node_ids, agent_mention_node_ids):
                event_arg_edges.append(e)
                event_arg_attrs.append('agent')
            for e in itertools.product(event_mention_node_ids, patient_mention_node_ids):
                event_arg_edges.append(e)
                event_arg_attrs.append('patient')
            # Build edges for ref-events graph
            for ref_event_id in event.get('ref-events', []):
                ref_event_mention_node_ids: List[int] = [
                    get_mention_node_idx['event'](span_idx_mapping[('event', anchor_span)])
                    for anchor_span in sanitized_ssid_span_mappings[('event', ref_event_id)]
                ]
                for e in itertools.product(event_mention_node_ids, ref_event_mention_node_ids):
                    event_ref_edges.append(e)
        # Build edges for includes-relations graph
        for tgt_ss, src_sss in includes_relations.items():
            tgt_mention_node_ids: List[int] = [
                get_mention_node_idx['ss'](span_idx_mapping[('ss', tgt_span)])
                for tgt_span in sanitized_ssid_span_mappings[('ss', tgt_ss)]
            ]
            src_mention_node_ids: List[int] = [
                get_mention_node_idx['ss'](span_idx_mapping[('ss', mention_span)])
                for src_ss in src_sss
                for mention_span in sanitized_ssid_span_mappings[('ss', src_ss)]
            ]
            for e in itertools.product(src_mention_node_ids, tgt_mention_node_ids):
                entity_includes_edges.append(e)

        # Compose a dict to store graph inputs
        # The graph inputs would be built on the fly within the model forward
        # Therefore, we don't store them in AllenNLP fields
        if not entity_mention_node_list:
            # Skip the instance when there is no entity
            return None

        text_field = TextField([Token(t) for t in text])
        span_field = ListField([SpanField(span[0], span[1], text_field) for span in spans])
        span_type_field = SequenceLabelField(span_types, span_field, label_namespace='span_type_labels')
        event_type_field = SequenceLabelField(event_types, span_field, label_namespace='event_types')
        template_type_field = LabelField(template_type, label_namespace='template_labels')
        is_training_field = FlagField(self.is_training)
        non_span_slots = self.non_span_slots_by_template[template_type]

        metadata = {
            'raw_doc_ref': raw_doc_ref,
            'data_path': data_path,
            'instance_id': instance_id,
            'span_ssid': span_ssid,
            'non_span_slots': non_span_slots
        }

        fields = {'metadata': MetadataField(metadata), 'spans': span_field, 'span_types': span_type_field,
                  'event_types': event_type_field, 'template_type': template_type_field,
                  'is_training': is_training_field, 'text': text_field}

        if len(non_span_slots) > 0:
            # All the non-spanset and non-event valued slots for this template
            non_span_slots_field = ListField([
                LabelField(slot_type, label_namespace='non_span_slot_types')
                for slot_type in non_span_slots
            ])
            # For each such slot, the set of labels allowed for that slot
            # (This enables us to handle boolean- and string-valued slots together)
            allowed_non_span_labels_field = ListField([
                ListField([
                    LabelField(non_span_label, label_namespace='non_span_slot_labels')
                    for non_span_label in self.allowed_values_for_non_span_slots[slot_type]
                ]) for slot_type in non_span_slots
            ])
            fields['non_span_slots'] = non_span_slots_field
            fields['allowed_non_span_labels'] = allowed_non_span_labels_field

        if gold_templates is not None:
            gold_template_span_labels: List[List[str]] = []
            gold_template_non_span_labels: List[List[str]] = []
            gold_span_slot_sets: List[Set[Tuple[int, str]]] = []
            gold_non_span_slot_sets: List[Set[Tuple[str, str]]] = []
            for template in gold_templates.values():
                if len(template) <= 3:
                    continue
                gold_template_span_label: List[str] = ['none'] * len(spans)
                gold_template_non_span_label: List[str] = ['none'] * len(non_span_slots)
                gold_span_slot_set: Set[Tuple[int, str]] = set()
                gold_non_span_slot_set: Set[Tuple[str, str]] = set()
                for slot_name, slot_values in template.items():
                    sanitized_slot_name = slot_name.lower()
                    if slot_name in TEMPLATE_NON_SLOT_KEYS:
                        continue
                    filler_types = self.definitions['definitions'][template_type]['slots'][
                        slot_name.lower()]['filler_types']
                    if 'spanset' not in filler_types and 'event' not in filler_types:
                        if 'bool' in filler_types or 'str' in filler_types:
                            sanitized_slot_value = str(slot_values).lower()
                            gold_template_non_span_label[
                                non_span_slots.index(sanitized_slot_name)] = sanitized_slot_value
                            gold_non_span_slot_set.add((sanitized_slot_name, sanitized_slot_value))
                        else:
                            raise ValueError(f'Unrecognized filler type in template definition file: {filler_types}')
                        continue

                    for slot_value in slot_values:
                        if template['template-type'].lower() not in {
                            'etiplate',
                            'cybercrimeplate'
                        } and self.convert_outcome_slot and slot_name == 'outcome':
                            slot_name_affix = IRREALIS_OUTCOME_MAPPINGS.get(slot_value.get('irrealis'), 'occurred')
                            slot_name_to_use = f'outcome-{slot_name_affix}'
                        else:
                            slot_name_to_use = sanitized_slot_name

                        if 'event-id' in slot_value:
                            candidate_ssid = slot_value['event-id']
                        elif 'ssid' in slot_value:
                            candidate_ssid = slot_value['ssid']
                        else:
                            raise ValueError(f'Unknown slot value: {slot_value}')
                        for span_idx in ssid_span_idx[candidate_ssid]:
                            gold_template_span_label[span_idx] = slot_name_to_use
                            # Offsets by 1 to account for the first token being template embedding
                            gold_span_slot_set.add((span_idx + 1, slot_name_to_use))
                gold_template_span_labels.append(gold_template_span_label)
                gold_template_non_span_labels.append(gold_template_non_span_label)
                gold_span_slot_sets.append(gold_span_slot_set)
                gold_non_span_slot_sets.append(gold_non_span_slot_set)

            # Stores as template-slot-spans for span sampling purpose
            gold_slot_spans: List[Dict[str, List[int]]] = [
                {
                    slot: [i - 1 for i, _ in spans]
                    for slot, spans in groupby(sorted(list(template), key=lambda x: x[1]), key=lambda x: x[1])
                }
                for template in gold_span_slot_sets
            ]

            if len(gold_template_span_labels) > 0:
                gold_template_span_label_field = ListField([
                    ListField([
                        LabelField(label, label_namespace='slot_types')
                        for label in gold_template_span_label
                    ])
                    for gold_template_span_label in gold_template_span_labels
                ])

                fields['gold_template_span_labels'] = gold_template_span_label_field

                # Add labels for non-span valued slots
                if len(non_span_slots) > 0:
                    gold_template_non_span_label_field = ListField([
                        ListField([
                            LabelField(label, label_namespace='non_span_slot_labels')
                            for label in gold_template_non_span_label
                        ])
                        for gold_template_non_span_label in gold_template_non_span_labels
                    ])
                    fields['gold_template_non_span_labels'] = gold_template_non_span_label_field

            metadata['non_span_slots'] = non_span_slots
            metadata['gold_span_slot_sets'] = gold_span_slot_sets
            metadata['gold_non_span_slot_sets'] = gold_non_span_slot_sets
            metadata['gold_templates'] = gold_templates
            metadata['gold_slot_spans'] = gold_slot_spans

        return Instance(fields)

    def _to_params(self) -> Dict[str, Any]:
        return super()._to_params()

    def apply_token_indexers(self, instance: Instance) -> None:
        if 'text' in instance:
            instance['text'].token_indexers = self.token_indexers


if __name__ == '__main__':
    from allennlp.data import Vocabulary
    import tqdm

    torch.manual_seed(0)
    vocab = Vocabulary.from_files('resources/data/granular/iter-template/vocabularies/vocabulary-p2')
    dataset_reader = GranularDataset(
        definition_file='resources/data/granular/template_definitions/phase2_simple_definitions.json',
        is_training=True
    )
    for x in tqdm.tqdm(dataset_reader.read(
            # 'resources/data/outputs_from_basic/granular.eng-provided-72.0pct.train-70.0pct.d.bp.json'
            'resources/data/granular/gold/clean/'
            'phase2_english_granular_20211201_v5-provided-72.0pct.analysis-15.0pct.ref.d.bp.json'
    )):
        x
        x.index_fields(vocab)
        t = x.as_tensor_dict()
        pass
