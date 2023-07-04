from typing import Tuple, Any, Dict


def parse_id(id_str: str) -> Tuple[str, int]:
    parsed_strs = id_str.split('-')
    assert len(parsed_strs) == 2, f'Parsing error, length does not match {len(parsed_strs)} != 2.'
    return parsed_strs[0], int(parsed_strs[1])


def next_valid_id(node_dict: Dict[str, Any], prefix: str) -> str:
    all_ids = sorted([parse_id(k) for k in node_dict.keys()], key=lambda x: x[1])
    if len(all_ids) == 0:
        return f'{prefix}-1'
    else:
        if len(all_ids) == all_ids[-1][1]:  # This list is complete and sorted
            return f'{prefix}-{len(all_ids) + 1}'
        else:  # Finds the missing id and insert at this pos
            for pos, i in enumerate(all_ids):
                if pos + 1 != i:
                    return f'{prefix}-{pos + 1}'
