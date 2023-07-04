import collections
import copy
import json
import logging
from typing import Tuple, Dict, Set, Any, List, Union, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

BPJsonSpan = NamedTuple('BPJsonSpan', [('range', Tuple[int, int]), ('type', str), ('ssid', str), ('str', str)])

# Constants
TEMPLATE_NON_SLOT_FIELD = ['template-anchor', 'template-id', 'template-type']


# Stream wrapper
def spanset_stream_wrapper(x): return x['annotation-sets']['basic-events']['span-sets'].items()


def events_stream_wrapper(x): return x['annotation-sets']['basic-events']['events'].items()


def granular_templates_stream_wrapper(
        x): return x['annotation-sets']['basic-events'].get('granular-templates', {}).items()


def includes_relations_stream_wrøapper(
        x): return x['annotation-sets']['basic-events'].get('includes-relations', {}).items()


def get_span_list(x, y): return x['annotation-sets']['basic-events']['span-sets'][y]['spans']


def get_event_anchor(x, y): return x['annotation-sets']['basic-events']['events'][y]['anchors']


def get_event_type(x, y): return x['annotation-sets']['basic-events']['events'][y]['event-type']


def read_as_data_dict(path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def induce_ontology_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    # Induce the ontology and count
    event_type_counter = collections.Counter()
    span_synclass_counter = collections.Counter()
    induced_template_ontology = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: {
                'filler_type': collections.Counter(),
                'filler_event_type': collections.Counter(),
                'filler_synclass': collections.Counter(),
                'irrealis': collections.Counter(),
                'time_attachments': 0,
            }
        )
    )
    # Mappings
    for split_name, data_dict in data.items():
        for entry_k, entry in data_dict['entries'].items():
            # Count events
            for evtid, event in events_stream_wrapper(entry):
                event_type_counter[event['event-type']] += 1
            # Count spansets
            for ssid, spanset in spanset_stream_wrapper(entry):
                for span in spanset['spans']:
                    if 'synclass' in span:
                        span_synclass_counter[span['synclass']] += 1
            # Count templates
            for tmplt_id, tmplt in granular_templates_stream_wrapper(entry):
                for k, v in tmplt.items():
                    if k in TEMPLATE_NON_SLOT_FIELD:
                        continue
                    tmplt_type = tmplt['template-type']
                    # Determines the slot filler type
                    if isinstance(v, str):
                        induced_template_ontology[tmplt_type][k]['filler_type']['str'] += 1
                    elif isinstance(v, bool):
                        induced_template_ontology[tmplt_type][k]['filler_type']['bool'] += 1
                    elif isinstance(v, list):
                        for filler in v:
                            if 'event-id' in filler:
                                induced_template_ontology[tmplt_type][k]['filler_type']['event'] += 1
                                induced_template_ontology[tmplt_type][k]['filler_event_type'][
                                    get_event_type(entry, filler['event-id'])] += 1
                            if 'ssid' in filler:
                                induced_template_ontology[tmplt_type][k]['filler_type']['spanset'] += 1
                                for span in get_span_list(entry, filler['ssid']):
                                    if 'synclass' in span:
                                        induced_template_ontology[tmplt_type][k]['filler_synclass'][
                                            span['synclass']] += 1
                            if 'time-attachments' in filler:
                                induced_template_ontology[tmplt_type][k]['time_attachments'] += 1
                            if 'irrealis' in filler:
                                induced_template_ontology[tmplt_type][k]['irrealis'][filler['irrealis']] += 1

    return {
        'event_types': event_type_counter,
        'span_synclass': span_synclass_counter,
        'induced_template_ontology': induced_template_ontology,
    }


def build_spanset_mappings(spansets, events) -> Tuple[
    Dict[Tuple[str, str], Set[Tuple[Tuple[int, int], str]]],
    Dict[Tuple[int, int], Set[str]],
    Dict[str, str]
]:
    """ Builds different mappings from span sets and events.
    Args:
        spansets: `Dict[str, Any]`, required.
        events: `Dict[str, Any]`, required.
    Returns:
        ssid_span_mappings: `Dict[Tuple[str, str], Set[Tuple[Tuple[int, int], str]]]`
        span_ssid_mappings: `Dict[Tuple[int, int], Set[str]]`
        ssid_eventid_mappings: `Dict[str, str]`
    """
    ssid_span_mappings: Dict[Tuple[str, str], Set[Tuple[Tuple[int, int], str]]] = collections.defaultdict(set)
    span_ssid_mappings: Dict[Tuple[int, int], Set[str]] = {}
    ssid_eventid_mappings: Dict[str, str] = {}

    for ssid in spansets:
        for span in spansets[ssid]['spans']:
            try:
                index = (span['start-token'], span['end-token'])
            except KeyError:
                logger.warn(f"Dropping span \"{span}\" because it was not properly tokenized.")
                continue
            ssid_span_mappings[('ssid', ssid)].add((index, span['string']))
            if index not in span_ssid_mappings:
                span_ssid_mappings[index] = {ssid}
            else:
                span_ssid_mappings[index].add(ssid)

    # TODO(@Yunmo): there is something wrong with building the event mappings.
    for eventid in events:
        anchor_id = events[eventid]['anchors']
        ssid_eventid_mappings[anchor_id] = eventid
        anchor_spans: Set[Tuple[Tuple[int, int], str]] = ssid_span_mappings[('ssid', anchor_id)]
        ssid_span_mappings[('event-id', eventid)] = copy.deepcopy(anchor_spans)
        # Makes sure `span_ssid_mappings` is updated to include the event spans.
        for anchor_span, _ in anchor_spans:
            if anchor_span in span_ssid_mappings:
                span_ssid_mappings[anchor_span] = {
                    ssid if ssid != anchor_id else eventid
                    for ssid in span_ssid_mappings[anchor_span]
                }

    return ssid_span_mappings, span_ssid_mappings, ssid_eventid_mappings


def get_template_range(template: Dict[str, Any],
                       ssid_span_mappings: Dict[Tuple[str, str], Set[Tuple[Tuple[int, int], str]]],
                       include_anchor: bool = True) -> Tuple[int, int]:
    """ Gets template range by sorting all spans appearing within the template.
    Args:
        template: `Dict[str, Any]`, required.
        ssid_span_mappings: `Dict[Tuple[str, str], Set[Tuple[Tuple[int, int], str]]]`, required.
        include_anchor: `bool`, optional.
    Returns:
        range: `Tuple[int, int]`
    """
    all_spans: List[Tuple[int, int]] = []
    for k, v in template.items():
        if isinstance(v, list):
            for ref in v:
                espans = ssid_span_mappings.get(('event-id', ref.get('event-id')))
                sspans = ssid_span_mappings.get(('ssid', ref.get('ssid')))
                if espans is not None:
                    all_spans.extend([s[0] for s in espans])
                if sspans is not None:
                    all_spans.extend([s[0] for s in sspans])
        elif k == 'template-anchor' and include_anchor:
            sspans = ssid_span_mappings.get(('ssid', v))
            all_spans.extend([s[0] for s in sspans])
    all_span_tensor = np.array(all_spans, dtype=np.long)
    start = all_span_tensor.min(0)[0].item()
    end = all_span_tensor.max(0)[-1].item()

    return start, end


def sent_char_to_token_range(char2tok: List[List[int]],
                             sent_char_range: List[List[int]]) -> List[Tuple[int, int]]:
    """ Converts sentence character range to token range using `char2tok` entry.
    Args:
        char2tok: `List[List[int]]`, required.
        sent_char_range: `List[List[int]]`, required.
    Returns:
        sent_token_range: `List[Tuple[int, int]]`.
    """

    def _resolve_tok_pos(char_pos: int, step=-1) -> int:
        while 0 <= char_pos < len(char2tok):
            if len(char2tok[char_pos]) != 0:
                return char2tok[char_pos][0]
            char_pos += step

        if char_pos >= len(char2tok):
            return char2tok[-1][-1]

        raise IndexError(f'Got char_pos={char_pos} < 0.')

    sent_token_range: List[Tuple[int, int]] = []
    for sc_start, sc_end in sent_char_range:
        # tok_start = char2tok[sc_start][0]
        tok_start = _resolve_tok_pos(sc_start, step=1)
        tok_end = _resolve_tok_pos(sc_end - 1)

        assert tok_start <= tok_end, f'Sentence start={tok_start} > end={tok_end}.'
        sent_token_range.append((tok_start, tok_end))
    return sent_token_range


def token_to_sent_pos(sent_token_range: List[Tuple[int, int]], token_ids: List[int]) -> List[Tuple[int, int]]:
    """ Converts token ids to its corresponding sentence start and end.
    TODO(@Yunmo): this method could be optimized.
    Args:
        sent_token_range: `List[Tuple[int, int]]`, required.
        token_ids: `List[int]`, required.
    Returns:
        sent_ids: `List[Tuple[int, int]]`
    """
    sent_ids: List[Tuple[int, int]] = []
    for tid in token_ids:
        for sent_start, sent_end in sent_token_range:
            if sent_start <= tid <= sent_end:  # TODO(@Yunmo): makes sure the boundary is right inclusive?
                sent_ids.append((sent_start, sent_end))
                break
    return sent_ids


def parse_ssid_type(ssid: str, return_stype: bool = False) -> str:
    ssid_prefix = ssid.split('-')[0]
    if ssid_prefix == 'ss':
        if return_stype:
            return 'spanset'
        else:
            return 'ssid'
    elif ssid_prefix == 'event':
        if return_stype:
            return 'event'
        else:
            return 'event-id'
    else:
        # raise NotImplementedError
        return ssid


def parse_ssid_no(ssid: str) -> str:
    return ssid.split('-')[-1]


def offset_range(start_offset: int, r: Tuple[int, int]) -> Tuple[int, int]:
    if start_offset == 0:
        return r
    else:
        return r[0] - start_offset, r[1] - start_offset


def offset_spans(s: Union[Dict, List, BPJsonSpan], offset: int):
    if isinstance(s, dict):
        return {
            k: offset_spans(v, offset)
            for k, v in s.items()
        }
    elif isinstance(s, list):
        return [
            offset_spans(i, offset)
            for i in s
        ]
    elif isinstance(s, set):
        return {
            offset_spans(i, offset)
            for i in s
        }
    elif isinstance(s, BPJsonSpan):
        return BPJsonSpan(range=(s.range[0] + offset, s.range[-1] + offset), type=s.type, ssid=s.ssid, str=s.str)
    else:
        raise NotImplementedError


def sentence_range(sections, sentence_only=True, deduplicate=True):
    # sections: "segment-sections" field in bp.json
    sent = []
    for sec in sections:
        if sentence_only:
            if sec["structural-element"] != "Sentence":
                continue
        sent.append([sec["start"], sec["end"]])
    if deduplicate:
        sent = [list(t) for t in list(set([tuple(t) for t in sent]))]
    return sorted(sent, key=lambda x: x[0])


def make_right_exclusive_ranges(sentence_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [
        (start, end + 1)
        for start, end in sentence_ranges
    ]
