from typing import Set, Dict, Any, List, Optional, Tuple, Union

import torch
from allennlp.data import Vocabulary
from allennlp.nn.util import min_value_of_dtype

SLOT_FILLER_TYPES = ['spanset', 'event']


class TemplateSetDecoder(object):
    def __init__(self,
                 vocab: Vocabulary,
                 definitions: Dict[str, Any],
                 span_slot_none_penalty: Union[float, Dict[str, float]] = 0.0,
                 non_span_slot_none_penalty: Union[float, Dict[str, float]] = 0.0,
                 use_sampling: bool = False):
        self.vocab = vocab
        self.definitions = definitions

        self.span_slot_none_penalty = span_slot_none_penalty
        self.non_span_slot_none_penalty = non_span_slot_none_penalty

        # Shape: (num_templates, num_slots, num_filler_types)
        self.slot_filler_type_constraints: torch.Tensor = self.build_slot_filler_type_constraint_mask(
            vocab=vocab,
            definitions=definitions
        )
        self.output_filtered_span_slot_idx: Set[int] = {
            self.vocab.get_token_index(token='none', namespace='slot_types'),
            self.vocab.get_token_index(token='@@PADDING@@', namespace='slot_types'),
            self.vocab.get_token_index(token='@@UNKNOWN@@', namespace='slot_types'),
        }

        # Ading this for debugging slot names (to check if they are lowercase)

        self.output_filtered_non_span_slot_idx: Set[int] = {
            self.vocab.get_token_index(token='none', namespace='non_span_slot_labels')
        }

        self.use_sampling = use_sampling

    @staticmethod
    def build_slot_filler_type_constraint_mask(
            vocab: Vocabulary,
            definitions: Dict[str, Any]
    ) -> torch.Tensor:
        template_idx_to_token = sorted(vocab.get_index_to_token_vocabulary(namespace='template_labels').items(),
                                       key=lambda x: x[0])
        slot_idx_to_token = sorted(vocab.get_index_to_token_vocabulary(namespace='slot_types').items(),
                                   key=lambda x: x[0])

        constraint_mask: List[List[List[bool]]] = []
        for template_idx, template_name in template_idx_to_token:
            template_constraint_mask: List[List[bool]] = []
            for slot_idx, slot_name in slot_idx_to_token:
                if slot_name in ['@@PADDING@@', '@@UNKNOWN@@']:
                    template_constraint_mask.append([False, False])
                    continue

                if slot_name == 'none':
                    template_constraint_mask.append([True, True])
                    continue

                if slot_name in definitions['definitions'][template_name]['slots']:
                    slot_filler_mask: List[bool] = []
                    slot_def = definitions['definitions'][template_name]['slots'][slot_name]
                    for filler_type in SLOT_FILLER_TYPES:
                        if filler_type in slot_def['filler_types']:
                            slot_filler_mask.append(True)
                        else:
                            slot_filler_mask.append(False)
                    template_constraint_mask.append(slot_filler_mask)
                else:
                    template_constraint_mask.append([False, False])
            constraint_mask.append(template_constraint_mask)
        return torch.tensor(constraint_mask, dtype=torch.bool)

    def decode(self,
               template_type: torch.Tensor,
               span_types: torch.Tensor,
               span_slot_logits: torch.Tensor,
               span_mask: torch.Tensor,
               non_span_slots: Optional[torch.Tensor] = None,
               non_span_slot_logits: Optional[torch.Tensor] = None,
               non_span_slot_logits_mask: Optional[torch.Tensor] = None,
               is_training: Optional[bool] = False) -> Tuple[Set[Tuple[int, str]], Set[Tuple[int, str]]]:
        # Shape: (batch_size)
        template_type = template_type.detach().cpu()
        # Shape: (batch_size, num_spans)
        span_types = span_types.detach().cpu()
        # Shape: (batch_size, num_spans, num_slot_types)
        span_slot_logits = span_slot_logits.detach().cpu()
        # Shape: (batch_size, num_spans)
        span_mask = span_mask.detach().cpu()

        if isinstance(self.span_slot_none_penalty, float):
            span_slot_none_penalty = self.span_slot_none_penalty
        elif isinstance(self.span_slot_none_penalty, dict):
            template_type_str = self.vocab.get_token_from_index(
                index=template_type.item(), namespace='template_labels')
            # There's potential for silent failure here if a penalty isn't explicitly specified for
            # a particular template when it ought to be. However, this preserves the existing default
            # of zero none penalty and requires explicit configuration only for templates for which
            # it should be non-zero
            span_slot_none_penalty = self.span_slot_none_penalty.get(template_type_str, 0.)

        span_slot_logits[:, :,
                         self.vocab.get_token_index(token='none', namespace='slot_types')] -= span_slot_none_penalty

        # Shape: (batch_size, num_spans, num_slot_types)
        filler_constraint_mask = self.slot_filler_type_constraints[template_type.unsqueeze(-1), :, span_types]
        masked_span_slot_logits = torch.masked_fill(span_slot_logits,
                                                    ~filler_constraint_mask,
                                                    min_value_of_dtype(span_slot_logits.dtype))

        # Shape: (batch_size, num_spans)
        if self.use_sampling and is_training:
            predicted_span_slot_types = torch.distributions.Categorical(logits=masked_span_slot_logits).sample()
        else:
            predicted_span_slot_types = masked_span_slot_logits.argmax(-1)

        predicted_span_slots = [
            {
                (span_id + 1, self.vocab.get_token_from_index(index=slot_type_idx, namespace='slot_types'))
                for span_id, (slot_type_idx, mask) in enumerate(zip(batch, batch_mask))
                if slot_type_idx not in self.output_filtered_span_slot_idx and mask
            }
            for batch, batch_mask in zip(predicted_span_slot_types.tolist(), span_mask.tolist())
        ]

        # If non-span-valued slots were predicted, decode them as well
        if non_span_slots is not None and len(non_span_slots) > 0:
            # Shape: (batch_size, num_non_spans, num_non_span_slot_labels)
            non_span_slot_logits = non_span_slot_logits.detach().cpu()
            non_span_slot_logits_mask = non_span_slot_logits_mask.detach().cpu()

            if isinstance(self.non_span_slot_none_penalty, float):
                non_span_slot_none_penalty = self.non_span_slot_none_penalty
            elif isinstance(self.non_span_slot_none_penalty, dict):
                template_type_str = self.vocab.get_token_from_index(index=template_type.item(),
                                                                    namespace='template_labels')
                # See warning from analogous code block above
                non_span_slot_none_penalty = self.non_span_slot_none_penalty.get(template_type_str, 0.)

            non_span_slot_logits[:, :, self.vocab.get_token_index(
                token='none', namespace='non_span_slot_labels')] -= non_span_slot_none_penalty

            # Shape: (batch_size, num_non_spans, num_non_span_slot_labels)
            masked_non_span_slot_logits = torch.masked_fill(non_span_slot_logits,
                                                            ~non_span_slot_logits_mask,
                                                            min_value_of_dtype(non_span_slot_logits.dtype))

            # Shape: (batch_size, num_non_spans)
            if self.use_sampling and is_training:
                predicted_non_span_slot_types = torch.distributions.Categorical(
                    logits=masked_non_span_slot_logits).sample()
            else:
                predicted_non_span_slot_types = masked_non_span_slot_logits.argmax(-1)

            predicted_non_span_slots = [
                {
                    (self.vocab.get_token_from_index(index=non_span_slot_type, namespace='non_span_slot_types'),
                     self.vocab.get_token_from_index(index=non_span_slot_value, namespace='non_span_slot_labels'))
                    for non_span_slot_type, non_span_slot_value in zip(batch_slot, batch_pred)
                    if non_span_slot_value not in self.output_filtered_non_span_slot_idx
                }
                for (batch_slot, batch_pred) in zip(non_span_slots.tolist(), predicted_non_span_slot_types.tolist())
            ]
        else:
            predicted_non_span_slots = [set()]

        return predicted_span_slots[0], predicted_non_span_slots[0]  # batch size: 1
