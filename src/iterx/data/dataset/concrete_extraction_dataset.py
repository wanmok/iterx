import collections
import json
import os
from typing import Iterable, List, Tuple, Optional, Dict, Set, Any

from allennlp.data import DatasetReader, Instance, Token, TokenIndexer
from allennlp.data.fields import TextField, SpanField, ListField, LabelField, FlagField, \
    MetadataField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from concrete import Situation
from itertools import groupby

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention

VALID_EXTRACTION_TASKS = {
    'scirex-re',
    'granular',
    'muc'
}


@DatasetReader.register("concrete_extraction")
class ConcreteExtractionDataset(DatasetReader):
    def __init__(self,
                 definition_file: str,
                 extraction_task: str,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 training_with_empty_docs: bool = False,
                 is_training: bool = True,
                 filter_incomplete_relations: bool = True,
                 max_instances: Optional[int] = None,
                 max_token_cutoff: Optional[int] = None,
                 **kwargs):
        super(ConcreteExtractionDataset, self).__init__(max_instances=max_instances, **kwargs)

        self.extraction_task = extraction_task
        assert self.extraction_task in VALID_EXTRACTION_TASKS, \
            'Invalid extraction task: {}'.format(self.extraction_task)

        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        self.training_with_empty_docs = training_with_empty_docs
        self.filter_incomplete_relations = filter_incomplete_relations
        self.is_training = is_training
        self.max_token_cutoff = max_token_cutoff

        # Loads definition file
        with open(definition_file) as f:
            self.definitions = json.load(f)

    def _read(self, file_path) -> Iterable[Instance]:
        split_paths = file_path.split(';')  # Use ; instead of : to avoid allennlp bug
        if len(split_paths) == 1:
            dataset_path = split_paths[0]
            upstream_anno_set = None
        elif len(split_paths) == 2:
            dataset_path = split_paths[0]
            upstream_anno_set = split_paths[1]
        else:
            raise ValueError('Invalid file path: {}'.format(file_path))

        file_names = os.listdir(dataset_path)

        for fn in file_names:
            if '.concrete' not in fn and '.comm' not in fn:
                continue
            comm_path = os.path.join(dataset_path, fn)
            cdoc = CementDocument.from_communication_file(comm_path)

            if self.max_token_cutoff is not None:
                text: List[str] = cdoc[:self.max_token_cutoff]
            else:
                text = cdoc[:]

            # Left, right inclusive
            sentence_ranges: List[Tuple[int, int]] = [
                (
                    s,
                    ((e - 1)
                     if self.max_token_cutoff is None
                     else (self.max_token_cutoff - 1 if e >= self.max_token_cutoff else e - 1))
                )
                for s, e in cdoc.iterate_sentence_ranges()
                if self.max_token_cutoff is None or s < self.max_token_cutoff
            ]

            # Load upstream spans if provided
            if upstream_anno_set is not None:
                upstream_anno_doc = CementDocument.from_communication(comm=cdoc.comm,
                                                                      annotation_set=upstream_anno_set)
                upstream_cems: Optional[List[CementEntityMention]] = []
                upstream_span_to_cem_mappings: Optional[Dict[Tuple[int, int], CementEntityMention]] = {}
                upstream_em_uuid_cem_mappings: Optional[Dict[str, CementEntityMention]] = {}
                for entity_mention in upstream_anno_doc.iterate_entity_mentions():
                    cem = CementEntityMention.from_entity_mention(entity_mention, document=cdoc)
                    if self.max_token_cutoff is None or (cem.start <= cem.end < self.max_token_cutoff):
                        upstream_cems.append(cem)
                        upstream_span_to_cem_mappings[cem.to_index_tuple()] = cem
                        upstream_em_uuid_cem_mappings[entity_mention.uuid.uuidString] = cem
            else:
                upstream_anno_doc = None
                upstream_cems = None
                upstream_span_to_cem_mappings = None
                upstream_em_uuid_cem_mappings = None

            cems: List[CementEntityMention] = []
            em_uuid_cem_mappings: Dict[str, CementEntityMention] = {}
            seen_spans: Set[Tuple[int, int]] = set()

            if self.is_training or upstream_anno_set is None:
                for entity_mention in cdoc.iterate_entity_mentions():
                    cem = CementEntityMention.from_entity_mention(entity_mention, document=cdoc)
                    em_indices = cem.to_index_tuple()
                    if self.max_token_cutoff is None or (cem.start <= cem.end < self.max_token_cutoff):
                        # Use upstream annotations to filter
                        if upstream_anno_set is not None and em_indices not in upstream_span_to_cem_mappings:
                            continue
                        cems.append(cem)
                        em_uuid_cem_mappings[entity_mention.uuid.uuidString] = cem
                        seen_spans.add(em_indices)
                if upstream_anno_set is not None:
                    for upstream_em_uuid, upstream_cem in upstream_em_uuid_cem_mappings.items():
                        upstream_span = upstream_cem.to_index_tuple()
                        if upstream_span not in seen_spans:
                            cems.append(upstream_cem)
                            em_uuid_cem_mappings[upstream_em_uuid] = upstream_cem
                            seen_spans.add(upstream_span)
            else:
                cems = upstream_cems
                em_uuid_cem_mappings = upstream_em_uuid_cem_mappings

            span_ssid_mappings: Dict[Tuple[int, int], Set[str]] = collections.defaultdict(set)
            ssid_span_mappings: Dict[str, Set[Tuple[Tuple[int, int], str]]] = collections.defaultdict(set)
            span_cem_mappings: Dict[Tuple[int, int], List[CementEntityMention]] = collections.defaultdict(list)

            for em_id, cem in enumerate(cems):
                em_indices: Tuple[int, int] = cem.to_index_tuple()  # left, right inclusive
                em_ssid = f'ss-{em_id}'
                span_ssid_mappings[em_indices].add(em_ssid)
                ssid_span_mappings[em_ssid].add((em_indices, cem.to_text()))
                span_cem_mappings[em_indices].append(cem)

            # Filter situations that have no mentions
            if self.is_training:
                if self.extraction_task == 'scirex-re':
                    filter = lambda x: x.situationType == 'RELATION'
                elif self.extraction_task == 'granular':
                    filter = lambda x: True
                else:
                    raise ValueError(f'Unknown extraction task {self.extraction_task}')
                all_situations: List[Situation] = list(cdoc.iterate_situations(filter))
                valid_situations: Optional[Dict[str, List[Situation]]] = collections.defaultdict(list)
                for situation in all_situations:
                    is_valid = True
                    num_valid_args = 0
                    for arg in situation.argumentList:
                        # Assume that the arguments can only be `Entity`
                        entity_uuid = arg.entityId
                        entity = cdoc.comm.entityForUUID[entity_uuid.uuidString]
                        if len(entity.mentionIdList) == 0:
                            if self.filter_incomplete_relations:
                                is_valid = False
                            break
                        else:
                            # Count valid mentions under the filtered doc
                            valid_mention_count = 0
                            for mention_id in entity.mentionIdList:
                                if mention_id.uuidString in em_uuid_cem_mappings:
                                    valid_mention_count += 1
                            if valid_mention_count == 0:
                                if self.filter_incomplete_relations:
                                    is_valid = False
                                break
                        num_valid_args += 1

                    if not self.filter_incomplete_relations and num_valid_args == 0:
                        is_valid = False

                    if is_valid:
                        valid_situations[situation.situationKind.lower()].append(situation)
            else:
                valid_situations = None

            # Iterate over all query to extract
            for template_type in self.definitions['definitions'].keys():
                built_instance: Optional[Instance] = self.text_to_instance(
                    raw_doc_ref=cdoc,
                    data_path=dataset_path,
                    instance_id=f'{cdoc.comm.id}:{template_type}',
                    template_type=template_type,
                    text=text,
                    sentence_ranges=sentence_ranges,
                    ssid_span_mappings=ssid_span_mappings,
                    span_ssid_mappings=span_ssid_mappings,
                    span_cem_mappings=span_cem_mappings,
                    em_uuid_cem_mappings=em_uuid_cem_mappings,
                    gold_situations=valid_situations.get(template_type, []) if self.is_training else None
                )
                if built_instance is not None:
                    yield built_instance

    def text_to_instance(
            self,
            raw_doc_ref: CementDocument,
            data_path: str,
            instance_id: str,
            template_type: str,
            text: List[str],
            sentence_ranges: List[Tuple[int, int]],
            span_ssid_mappings: Dict[Tuple[int, int], Set[str]],
            ssid_span_mappings: Dict[str, Set[Tuple[Tuple[int, int], str]]],
            span_cem_mappings: Dict[Tuple[int, int], List[CementEntityMention]],
            em_uuid_cem_mappings: Dict[str, CementEntityMention],
            gold_situations: List[Situation] = None,
    ) -> Optional[Instance]:
        # Build spans
        spans: List[Tuple[int, int, str]] = []
        span_ssid: List[str] = []
        span_types: List[int] = []
        ssid_span_idx: Dict[str, Set[int]] = collections.defaultdict(set)
        span_idx_mapping: Dict[Tuple[str, Tuple[int, int]], int] = {}
        for span, ssids in span_ssid_mappings.items():
            for ssid in ssids:
                span_idx = len(spans)
                # Assume that there is only one span
                span_cem = span_cem_mappings[span][0]
                spans.append(
                    (span[0], span[1],
                     span_cem.attrs.entity_type.lower() if span_cem.attrs.entity_type is not None else None)
                )
                span_ssid.append(ssid)
                ssid_span_idx[ssid].add(span_idx)
                if ssid.startswith('ss-'):
                    span_type = 0
                    # event_type = '@@PADDING@@'
                    span_idx_mapping[('ss', span)] = span_idx
                else:
                    raise ValueError(f'Unknown span type: {ssid}')
                span_types.append(span_type)

        text_field = TextField([Token(t) for t in text])
        span_field = ListField([SpanField(span[0], span[1], text_field) for span in spans])
        span_type_field = SequenceLabelField(span_types, span_field, label_namespace='span_type_labels')
        template_type_field = LabelField(template_type, label_namespace='template_labels')
        is_training_field = FlagField(self.is_training)
        sentence_range_field = ListField([SpanField(span[0], span[1], text_field) for span in sentence_ranges])

        metadata = {
            'raw_doc_ref': raw_doc_ref,
            'data_path': data_path,
            'instance_id': instance_id,
            'span_ssid': span_ssid,
        }

        fields = {
            'metadata': MetadataField(metadata),
            'text': text_field,
            'spans': span_field,
            'span_types': span_type_field,
            'sentence_ranges': sentence_range_field,
            'template_type': template_type_field,
            'is_training': is_training_field,
        }

        if gold_situations is not None and len(gold_situations) > 0:
            gold_template_span_labels: List[List[str]] = []
            gold_span_slot_sets: List[Set[Tuple[int, str]]] = []

            for situation in gold_situations:
                gold_template_span_label: List[str] = ['none'] * len(spans)
                gold_span_slot_set: Set[Tuple[int, str]] = set()
                for argument in situation.argumentList:
                    sanitized_slot_name = argument.role.lower()
                    # Assume that the arguments can only be `Entity`
                    entity_uuid = argument.entityId
                    entity = raw_doc_ref.comm.entityForUUID[entity_uuid.uuidString]

                    for mention_id in entity.mentionIdList:
                        # mention = CementEntityMention.from_entity_mention(
                        #     mention=raw_doc_ref.comm.entityMentionForUUID[mention_id.uuidString],
                        #     document=raw_doc_ref
                        # )
                        mention = em_uuid_cem_mappings.get(mention_id.uuidString)
                        if mention is None:
                            continue
                        mention_indices: Tuple[int, int] = mention.to_index_tuple()
                        mention_span_idx = span_idx_mapping[('ss', mention_indices)]

                        gold_template_span_label[mention_span_idx] = sanitized_slot_name
                        gold_span_slot_set.add((mention_span_idx + 1, sanitized_slot_name))

                gold_template_span_labels.append(gold_template_span_label)
                gold_span_slot_sets.append(gold_span_slot_set)

            # Stores as template-slot-spans for span sampling purpose
            gold_slot_spans: List[Dict[str, List[int]]] = [
                {
                    slot: [i - 1 for i, _ in spans]
                    for slot, spans in groupby(sorted(list(template), key=lambda x: x[1]), key=lambda x: x[1])
                }
                for template in gold_span_slot_sets
            ]

            # if len(gold_template_span_labels) > 0:
            gold_template_span_label_field = ListField([
                ListField([
                    LabelField(label, label_namespace='slot_types')
                    for label in gold_template_span_label
                ])
                for gold_template_span_label in gold_template_span_labels
            ])

            fields['gold_template_span_labels'] = gold_template_span_label_field

            metadata['gold_span_slot_sets'] = gold_span_slot_sets
            metadata['gold_non_span_slot_sets'] = [set()] * len(gold_span_slot_sets)  # No non-span slots for SciREX
            metadata['gold_situations'] = gold_situations
            metadata['gold_slot_spans'] = gold_slot_spans

        else:
            if self.is_training and not self.training_with_empty_docs:
                return None

        return Instance(fields)

    def _to_params(self) -> Dict[str, Any]:
        return super()._to_params()

    def apply_token_indexers(self, instance: Instance) -> None:
        if 'text' in instance:
            instance['text'].token_indexers = self.token_indexers


if __name__ == "__main__":
    from allennlp.data import Vocabulary
    import tqdm
    import torch

    torch.manual_seed(0)
    vocab = Vocabulary.from_files('resources/data/scirex/vocabulary')
    dataset_reader = ConcreteExtractionDataset(
        definition_file='resources/data/scirex/definitions.json',
        extraction_task='scirex-re',
        filter_incomplete_relations=False,
        is_training=True,
        max_token_cutoff=None
    )

    from collections import Counter

    num_templates = Counter()
    doc_lenghts = Counter()
    num_mentions = Counter()
    for x in tqdm.tqdm(dataset_reader.read(
            'resources/data/scirex/preprocessed/sf-outputs/train'
    )):
        x.index_fields(vocab)
        t = x.as_tensor_dict()
        pass
