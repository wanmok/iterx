from typing import List, Dict, Any, Optional, Union, Iterator

from better.utils.bp_annotators.utils import next_valid_id, parse_id
from better.utils.bpjson_utils import parse_ssid_type, BPJsonSpan
from lib.bp import BPDocument, BPDocumentEntry, BPDocumentGranularTemplate

OUTCOME_IRREALIS_MAPPINGS = {
    'outcome-averted': 'counterfactual',
    'outcome-hypothetical': 'hypothetical',
}


def replace_template_span_with_spanset(templates: List[Dict[str, Any]]) -> List[Dict[str, List[Dict[str, str]]]]:
    def _iterate_bpjson_spans(ss) -> Iterator[BPJsonSpan]:
        for s in ss:
            if isinstance(s, BPJsonSpan):
                yield s
            elif isinstance(s, tuple):
                yield s[0]
            else:
                raise ValueError(f'Unsupported type: {type(s)}')

    replaced_templates: List[Dict[str, List[Dict[str, str]]]] = []
    for template in templates:
        new_template = {}
        for k, v in template.items():
            if k == 'template_slots':
                new_template['template_slots']: Dict[str, List[Dict[str, str]]] = {}
                for stype, sval in v.items():
                    if isinstance(sval, bool) or isinstance(sval, str):
                        # just copy over values for any non-span-valued slots;
                        # no filtering necessary
                        new_template['template_slots'][stype] = sval
                    else:
                        new_template['template_slots'][stype] = [
                            {stype: ssid}
                            for stype, ssid in
                            set([
                                (parse_ssid_type(bpspan.ssid), bpspan.ssid)
                                for bpspan in _iterate_bpjson_spans(sval)
                            ])
                        ]
            else:
                new_template[k] = v
        replaced_templates.append(new_template)
    return replaced_templates


def map_template_type_to_output(template_type: str,
                                definitions: Dict[str, Any]) -> Optional[str]:
    template_def = definitions['definitions'].get(template_type)
    if template_def is not None:
        return template_def.get('output_str')
    return None


def annotate_bpdoc_using_decoded_templates(bp_doc: BPDocument,
                                           decoded_templates: Dict[str, List[Dict[str, Any]]],
                                           template_definitions: Dict[str, Any],
                                           convert_outcome_slot: bool = True):
    for whole_id, templates in decoded_templates.items():
        data_path, entry_id = whole_id.split(':')
        entry: BPDocumentEntry = bp_doc.entries.value[entry_id]
        if entry.annotation_sets.value.basic_events.value.granular_templates.value is None:
            entry.annotation_sets.value.basic_events.value.granular_templates.value = {}
        granular_template_node: Dict = entry.annotation_sets.value.basic_events.value.granular_templates.value
        # Decoded templates in some case might have both templates w/ ids and templates w/o ids
        # In such cases, we would add those templates w/ ids first to
        # avoid different template getting the same template_id
        sorted_templates = sorted(
            templates,
            key=lambda k: parse_id(k['template_id'])[1] if k.get('template_id') is not None else 9999
        )
        for template in sorted_templates:
            template_id: Optional[Union[str, int]] = template.get('template_id')
            if template_id is None:
                template_id = next_valid_id(granular_template_node, prefix='template')
            if isinstance(template_id, int):
                template_id = f'template-{template_id}'

            new_template_dict = {
                # Required fields
                'template-id': template_id,
                'template-type': map_template_type_to_output(template_type=template['template_type'],
                                                             definitions=template_definitions),
                # Determines which to use as anchors
                # TODO(@Yunmo): this anchor selection might cause troubles
                'template-anchor': template.get('template_anchor', ''),
            }

            for slot_type, slot_value in template['template_slots'].items():
                if isinstance(slot_value, bool) or isinstance(slot_value, str) or len(slot_value) > 0:
                    if convert_outcome_slot and slot_type.startswith('outcome'):
                        slot_value: List[Dict]
                        irrealis_attr: Optional[str] = OUTCOME_IRREALIS_MAPPINGS.get(slot_type)
                        slot_type_to_add: str = 'outcome'
                        irrealis_to_append = {'irrealis': irrealis_attr} if irrealis_attr is not None else {}
                        if slot_type_to_add not in new_template_dict:
                            new_template_dict[slot_type_to_add] = []
                        new_template_dict[slot_type_to_add].extend([
                            x | irrealis_to_append
                            for x in slot_value
                        ])
                    else:
                        slot_type_to_add: str = slot_type
                        new_template_dict[slot_type_to_add] = slot_value

            new_template = BPDocumentGranularTemplate.from_dict(new_template_dict)

            granular_template_node[template_id] = new_template
