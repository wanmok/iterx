def spans_to_bio(spans, token_num, right_inclusive=True):
    """ Convert entity mentions in a sentence to an entity label sequence with
    the length of token_num

    Args:
        spans: `List[Tuple[int, int, str]]`, required.
            A list of entity mentions.
        token_num: `int`, required.
            The number of tokens.
        right_inclusive: `bool`, optional.
            Whether spans are right inclusive.

    Returns:
        A sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for span in spans:
        start, end = span[0], span[1]
        if right_inclusive:
            end += 1
        entity_type = span[2]
        if any([labels[i] != 'O' for i in range(start, end)]):
            continue
        labels[start] = 'B-{}'.format(entity_type)
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format(entity_type)
    return labels
