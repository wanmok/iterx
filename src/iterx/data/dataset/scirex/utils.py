import os
from collections import OrderedDict
from typing import List

from cement.cement_document import CementDocument
from cement.cement_entity_mention import CementEntityMention
from iterx.metrics.muc.ceaf_rme import GTTTemplate


def read_scirex_relations_as_gtt_template(
        dataset_path: str,
        dedup_str: bool = True,
) -> OrderedDict[str, OrderedDict[int, GTTTemplate]]:
    file_names = [
        x for x in os.listdir(dataset_path)
        if '.concrete' in x or '.comm' in x
    ]

    all_templates: OrderedDict[str, OrderedDict[int, GTTTemplate]] = OrderedDict()
    for file_name in file_names:
        templates: OrderedDict[int, GTTTemplate] = OrderedDict()
        file_path = os.path.join(dataset_path, file_name)
        cdoc = CementDocument.from_communication_file(file_path)
        for situation in cdoc.iterate_situations():
            is_valid = True
            new_template = {
                'incident_type': situation.situationKind.lower()
            }
            for argument in situation.argumentList:
                sanitized_slot_name = argument.role.lower()
                # Assume that the arguments can only be `Entity`
                entity_uuid = argument.entityId
                entity = cdoc.comm.entityForUUID[entity_uuid.uuidString]
                if len(entity.mentionIdList) == 0:
                    is_valid = False
                    break

                entity_mentions: List[str] = []
                for mention_id in entity.mentionIdList:
                    mention = cdoc.comm.entityMentionForUUID[mention_id.uuidString]
                    if mention is None:
                        continue

                    cement_mention = CementEntityMention.from_entity_mention(mention=mention,
                                                                             document=cdoc)
                    entity_mentions.append(cement_mention.to_text())

                if len(entity_mentions) == 0:
                    is_valid = False
                    break

                if sanitized_slot_name not in new_template:
                    new_template[sanitized_slot_name] = []
                new_template[sanitized_slot_name].append(list(set(entity_mentions)) if dedup_str else entity_mentions)

            if not is_valid:
                break
            templates[len(templates)] = new_template

        all_templates[cdoc.comm.id] = templates

    return all_templates
