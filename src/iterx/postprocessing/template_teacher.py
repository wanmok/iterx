from typing import List, Set, Tuple, Optional, Any, Dict

import torch
from allennlp.nn.util import min_value_of_dtype

ALLOWED_SAMPLING_DISTRIBUTIONS = {'categorical', 'uniform', 'argmax', 'deterministic'}


class TemplateTeacher(object):
    def __init__(self,
                 span_slot_criterion_config: dict,
                 non_span_slot_criterion_config: dict,
                 gold_span_slot_sets: List[Set[Tuple[int, str]]],
                 gold_non_span_slot_sets: List[Set[Tuple[str, str]]],
                 teacher_state: Dict[str, Any],
                 gold_template_span_labels: Optional[torch.Tensor] = None,
                 gold_template_non_span_labels: Optional[torch.Tensor] = None,
                 sampling_distribution: str = 'categorical',
                 temp_start: float = 10.,
                 temp_end: float = 1.,
                 temp_decay: float = 0.999,
                 span_reuse_discount_factor: Optional[float] = None,
                 none_slot_type_index: Optional[int] = None):
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.temp_decay = temp_decay
        self.teacher_state = teacher_state
        self.sampling_distribution = sampling_distribution
        assert sampling_distribution in ALLOWED_SAMPLING_DISTRIBUTIONS, \
            f'Sampling distribution {sampling_distribution} not supported'
        self.span_reuse_discount_factor = span_reuse_discount_factor
        self.none_slot_type_index = none_slot_type_index

        self.gold_span_slot_sets = gold_span_slot_sets
        self.gold_non_span_slot_sets = gold_non_span_slot_sets
        # Shape: (batch_size, num_gold_templates, num_spans)
        self.gold_template_span_labels = (
            gold_template_span_labels.detach().cpu()
            if gold_template_span_labels is not None
            else None
        )
        # Shape: (batch_size, num_gold_templates, num_non_spans)
        self.gold_template_non_span_labels = gold_template_non_span_labels.detach(
        ).cpu() if gold_template_non_span_labels is not None else None

        self.used_template_mask = (
            torch.zeros(len(gold_span_slot_sets), dtype=torch.bool)
            if len(gold_span_slot_sets) > 0 else None
        )
        self.span_slot_match_criterion = torch.nn.CrossEntropyLoss(**span_slot_criterion_config)
        self.non_span_slot_match_criterion = torch.nn.CrossEntropyLoss(**non_span_slot_criterion_config)

        self.xentropy_span_slot_ignore_index = span_slot_criterion_config['ignore_index']

    def get_eos(self) -> bool:
        if self.used_template_mask is None:
            return True
        return self.used_template_mask.all().item()

    def get_reference_action(
            self,
            span_slot_scores: torch.Tensor,
            span_mask: torch.Tensor,
            non_span_slot_scores: Optional[torch.Tensor] = None,
            span_use_counts: Optional[torch.Tensor] = None
            # predicted_template_set: Set[Tuple[int, str]]
    ) -> Tuple[int, Set[Tuple[int, str]], Set]:
        assert self.used_template_mask is not None, "No gold templates"

        # TODO(@Yunmo): actually, we could implement a tensorized version of this function... nvm...
        scores: List[float] = []
        # Shape: (batch_size, num_spans, num_slot_types)
        span_slot_scores = span_slot_scores.detach().cpu()
        span_mask = span_mask.detach().cpu()
        batch_size, num_spans, num_slot_types = span_slot_scores.shape

        if span_use_counts is not None:
            span_use_counts = span_use_counts.detach().cpu()

        # For each gold template, the (positive) log-likelihood of its span-valued slots
        for i in range(self.gold_template_span_labels.shape[1]):
            # It should be loglikelihood instead of NLL
            reference = self.gold_template_span_labels[0, i, :].view(batch_size * num_spans).contiguous().clone()
            reference[~span_mask[0]] = self.xentropy_span_slot_ignore_index
            score = - self.span_slot_match_criterion(span_slot_scores.view(batch_size * num_spans, -1), reference)
            if self.span_reuse_discount_factor is not None:
                # Incorporate span use counts for current template in computing score
                discount_factor_exponent = span_use_counts + (reference != self.none_slot_type_index)
                score = torch.mean(score * (self.span_reuse_discount_factor ** discount_factor_exponent))
            scores.append(score.item())

        if non_span_slot_scores is not None:
            non_span_slot_scores = non_span_slot_scores.detach().cpu()
            batch_size, num_non_spans, num_non_span_labels = non_span_slot_scores.shape
            # For each gold template, the (positive) log-likelihood of its NON-span-valued slots
            for i in range(self.gold_template_non_span_labels.shape[1]):
                non_span_slot_match_score = - self.non_span_slot_match_criterion(
                    non_span_slot_scores.view(batch_size * num_non_spans, -1),
                    self.gold_template_non_span_labels[0, i, :].view(batch_size * num_non_spans)).item()
                # logsumexp with span-valued slot LL to obtain overall slot score per template
                scores[i] = torch.logsumexp(torch.tensor([scores[i], non_span_slot_match_score]), dim=0).item()

        # for i, gold_span_slot_set in enumerate(self.gold_span_slot_sets):
        #     intersected_set = gold_span_slot_set.intersection(predicted_template_set)
        #     num_intersected = len(intersected_set)
        #     num_gold = len(gold_span_slot_set)
        #     scores.append(num_intersected / num_gold)

        # Shape: (batch_size, num_gold_templates)
        tensorized_scores = torch.tensor(scores)

        current_temp = self.temp_end + self.temp_start * (self.temp_decay ** self.teacher_state['step_done'])
        logits = torch.exp(tensorized_scores / current_temp)
        masked_logits = torch.masked_fill(logits,
                                          self.used_template_mask,
                                          min_value_of_dtype(tensorized_scores.dtype))
        if self.sampling_distribution == 'categorical':
            best_score_idx: int = torch.distributions.Categorical(logits=masked_logits).sample().item()
        elif self.sampling_distribution == 'argmax':
            best_score_idx: int = masked_logits.argmax().item()
        elif self.sampling_distribution == 'uniform':
            logits = torch.zeros_like(tensorized_scores)
            masked_logits = torch.masked_fill(logits,
                                              self.used_template_mask,
                                              min_value_of_dtype(tensorized_scores.dtype))
            best_score_idx: int = torch.distributions.Categorical(logits=masked_logits).sample().item()
        elif self.sampling_distribution == 'deterministic':
            logits = torch.zeros_like(tensorized_scores)
            masked_logits = torch.masked_fill(logits,
                                              self.used_template_mask,
                                              min_value_of_dtype(tensorized_scores.dtype))
            # all unmasked logits are equal; argmax always returns the first
            best_score_idx: int = masked_logits.argmax().item()
        else:
            raise NotImplementedError(f'Unknown sampling distribution: {self.sampling_distribution}')

        # Update the used template mask
        self.used_template_mask[best_score_idx] = True

        self.teacher_state['step_done'] += 1

        best_non_span_slot_set = set()
        if self.gold_non_span_slot_sets is not None and len(self.gold_non_span_slot_sets) > 0:
            best_non_span_slot_set = self.gold_non_span_slot_sets[best_score_idx]
        else:
            best_non_span_slot_set = None

        return best_score_idx, self.gold_span_slot_sets[best_score_idx], best_non_span_slot_set
