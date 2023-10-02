import json
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import Embedding, TimeDistributed, FeedForward, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from allennlp.nn import InitializerApplicator, Activation
from allennlp.training.metrics import Metric
from overrides import overrides
from torch.utils.checkpoint import checkpoint

from iterx.metrics.muc.gtt_eval_utils import normalize_string, role2uppercase
from iterx.modules.positional_embeddings import generate_sinusoidal_features
from iterx.postprocessing.template_set_decoder import TemplateSetDecoder
from iterx.postprocessing.template_teacher import TemplateTeacher

StrSlot = Tuple[str, str]
IntSlot = Tuple[int, str]

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


@Model.register("iterative_template_extraction_famus")
class IterativeTemplateExtractionFAMuS(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 definition_file: str,
                 graph_encoder: Seq2SeqEncoder,
                 span_sampling_rate: float = 1.0,
                 use_stochastic_layer_selection: bool = False,
                 text_field_embedder: Optional[TextFieldEmbedder] = None,
                 text_field_embedder_output_dim: int = 1024,
                 expert_roll_out: Optional[float] = None,
                 pairwise_scoring: bool = False,
                 span_slot_none_penalty: Union[float, Dict[str, float]] = 0.0,
                 non_span_slot_none_penalty: Union[float, Dict[str, float]] = 0.0,
                 span_slot_none_weight: float = 0.1,
                 non_span_slot_none_weight: float = 0.1,
                 lexical_dropout: float = 0.2,
                 feature_dropout: float = 0.2,
                 metrics: Optional[Dict[str, Metric]] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 iteration_cutoff: int = 15,
                 grad_discount_factor: float = 0.95,
                 span_reuse_discount_factor: Optional[float] = None,
                 teacher_configs: Dict[str, Any] = None,
                 template_set_decoder_use_sampling: bool = False,
                 use_checkpoint: bool = False,
                 **kwargs):
        super(IterativeTemplateExtractionFAMuS, self).__init__(vocab=vocab, **kwargs)

        # Loads definition file
        with open(definition_file) as f:
            self.definitions = json.load(f)

        self.use_checkpoint = use_checkpoint

        # Some variables
        if span_sampling_rate < 1.:
            self.span_sampling_distribution = torch.distributions.bernoulli.Bernoulli(probs=span_sampling_rate)
            self.full_span_sampling_distribution = torch.distributions.bernoulli.Bernoulli(
                probs=1 - span_sampling_rate)
        else:
            self.span_sampling_distribution = None

        self.use_stochastic_layer_selection = use_stochastic_layer_selection

        self.grad_discount_factor = grad_discount_factor
        self.pairwise_scoring = pairwise_scoring

        # Defines the model
        # Text encoding
        self.text_field_embedder: Optional[TextFieldEmbedder] = text_field_embedder
        # Workaround to support T5-kind of models
        if self.text_field_embedder is not None:
            tf_model_name: str = (
                self.text_field_embedder._token_embedders['tokens']._matched_embedder.transformer_model.name_or_path
            )
            if ('t0' in tf_model_name.lower()) or ('t5' in tf_model_name.lower()) or ('bart' in tf_model_name.lower()):
                # hack to only keep the transformer encoder
                self.text_field_embedder._token_embedders['tokens']._matched_embedder.transformer_model.config.update(
                    {"gradient_checkpointing": self.use_checkpoint}
                )
                self.text_field_embedder._token_embedders['tokens']._matched_embedder.transformer_model \
                    = self.text_field_embedder._token_embedders['tokens']._matched_embedder.transformer_model.encoder

            text_field_embedder_output_dim = self.text_field_embedder.get_output_dim()

        if lexical_dropout > 0:
            self.lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self.lexical_dropout = lambda x: x

        # Span representations
        self.endpoint_span_extractor = EndpointSpanExtractor(
            input_dim=text_field_embedder_output_dim,
            combination='x,y',
        )
        self.attentive_span_extractor = SelfAttentiveSpanExtractor(
            input_dim=text_field_embedder_output_dim,
        )
        # Graph encoding
        self.graph_encoder = graph_encoder
        # Defines embeddings
        self.template_embeddings = Embedding(vocab=vocab,
                                             vocab_namespace='template_labels',
                                             # num_embeddings=vocab.get_vocab_size(namespace='template_labels'),
                                             embedding_dim=graph_encoder.get_input_dim())
        self.event_type_embeddings = Embedding(vocab=vocab,
                                               vocab_namespace='event_types',
                                               # num_embeddings=vocab.get_vocab_size(namespace='event_types'),
                                               embedding_dim=graph_encoder.get_input_dim(),
                                               # Avoids messing up with entities
                                               padding_index=vocab.get_token_index('@@PADDING@@',
                                                                                   namespace='event_types'))
        self.span_type_embeddings = Embedding(num_embeddings=2,  # EVENT ENTITY
                                              embedding_dim=graph_encoder.get_input_dim())
        self.slot_type_embeddings = Embedding(vocab=vocab,
                                              vocab_namespace='slot_types',
                                              # num_embeddings=vocab.get_vocab_size(namespace='slot_types'),
                                              embedding_dim=graph_encoder.get_output_dim())  # Final classifications
        # Embeddings for NSV slots. NOTE: slot_type_embeddings also includes embeddings for NSV slots, but
        # those are used only for slot memory updates.
        self.non_span_embeddings = Embedding(vocab=vocab,
                                             vocab_namespace='non_span_slot_types',
                                             # num_embeddings=vocab.get_vocab_size(namespace='non_span_slot_types'),
                                             embedding_dim=graph_encoder.get_output_dim())

        self.non_span_slot_label_embeddings = Embedding(
            vocab=vocab,
            vocab_namespace='non_span_slot_labels',
            # num_embeddings=vocab.get_vocab_size(namespace='non_span_slot_labels'),
            embedding_dim=graph_encoder.get_output_dim())

        # FFNNs
        self.compression_feedforward = TimeDistributed(FeedForward(
            input_dim=self.attentive_span_extractor.get_output_dim() + self.endpoint_span_extractor.get_output_dim(),
            num_layers=2,
            hidden_dims=self.graph_encoder.get_input_dim(),
            activations=Activation.by_name('gelu')(),
            dropout=feature_dropout
        ))
        self.eos_feedforward = torch.nn.Sequential(
            FeedForward(
                input_dim=self.graph_encoder.get_output_dim(),
                num_layers=2,
                hidden_dims=self.graph_encoder.get_output_dim(),
                activations=Activation.by_name('gelu')(),
                dropout=feature_dropout
            ),
            torch.nn.Linear(self.graph_encoder.get_output_dim(), 2)
        )
        self.slot_feedforward = TimeDistributed(FeedForward(
            input_dim=self.graph_encoder.get_output_dim() * (2 if pairwise_scoring else 1),
            num_layers=2,
            hidden_dims=self.graph_encoder.get_output_dim(),
            activations=Activation.by_name('gelu')(),
            dropout=feature_dropout
        ))
        self.slot_memory_update_feedforward = TimeDistributed(FeedForward(
            input_dim=2 * self.graph_encoder.get_output_dim(),
            num_layers=1,
            hidden_dims=self.graph_encoder.get_output_dim(),
            activations=Activation.by_name('gelu')()
        ))
        self.template_memory_update_feedforward = FeedForward(
            input_dim=self.graph_encoder.get_output_dim(),
            num_layers=1,
            hidden_dims=self.graph_encoder.get_output_dim(),
            activations=Activation.by_name('gelu')()
        )

        # Memory mechanism
        self.memory_updator = torch.nn.GRUCell(input_size=self.graph_encoder.get_output_dim(),
                                               hidden_size=self.graph_encoder.get_output_dim())

        # Loss weights on "none" label for span-valued slots
        # and non-span-valued slots
        self.span_slot_none_weight = span_slot_none_weight
        self.non_span_slot_none_weight = non_span_slot_none_weight

        # Variable loss for each span depending on the
        # number of templates in which it has already been used
        self.span_reuse_discount_factor = span_reuse_discount_factor

        # Loss functions
        eos_weights = torch.tensor([1.0, 1.0])
        self.eos_criterion = torch.nn.CrossEntropyLoss(weight=eos_weights)
        span_slot_weights = torch.ones(vocab.get_vocab_size(namespace='slot_types'))
        span_slot_weights[vocab.get_token_index('none', namespace='slot_types')] = self.span_slot_none_weight
        xentropy_span_slot_ignore_index = vocab.get_token_index('@@PADDING@@', namespace='slot_types')
        self.xentropy_span_slot_config = {
            'weight': span_slot_weights,
            'ignore_index': xentropy_span_slot_ignore_index,
            # No mean-reducing if we need to do per-span loss weighting
            'reduction': 'mean' if self.span_reuse_discount_factor is None else 'none'
        }
        self.span_slot_criterion = torch.nn.CrossEntropyLoss(**self.xentropy_span_slot_config)
        self.xentropy_span_slot_ignore_index = xentropy_span_slot_ignore_index

        non_span_slot_weights = torch.ones(vocab.get_vocab_size(namespace='non_span_slot_labels'))
        non_span_slot_weights[
            vocab.get_token_index('none', namespace='non_span_slot_labels')] = self.non_span_slot_none_weight
        self.xentropy_non_span_slot_config = {
            'weight': non_span_slot_weights
        }
        self.non_span_slot_criterion = torch.nn.CrossEntropyLoss(**self.xentropy_non_span_slot_config)

        # Decoders
        self.template_set_decoder = TemplateSetDecoder(vocab=vocab,
                                                       definitions=self.definitions,
                                                       span_slot_none_penalty=span_slot_none_penalty,
                                                       non_span_slot_none_penalty=non_span_slot_none_penalty,
                                                       use_sampling=template_set_decoder_use_sampling)

        self.iteration_cutoff = iteration_cutoff

        # Metrics
        self.metrics = {
            # Comment out the following metrics to make them only configurable
            # 'proxy_template_match_metric': ProxyTemplateMatchMetric(),
            # 'proxy_slot_mention_match_metric': ProxySlotMatchMetric(),
            # 'proxy_slot_cluster_match_metric': ProxySlotMatchMetric(),
            # 'proxy_non_span_slot_match_metric': ProxySlotMatchMetric()
        }
        if metrics is not None:
            for metric_name, metric in metrics.items():
                assert metric_name not in self.metrics, f'Metric name {metric_name} already exists.'
                self.metrics[metric_name] = metric

        self.expert_roll_out = expert_roll_out
        self.expert_roll_out_distribution = torch.distributions.Uniform(0, 1)
        self.teacher_state = {
            'step_done': 0
        }

        # Teacher configs
        self.teacher_configs = {
            'temp_start': 10.,
            'temp_end': 1.0,
            'temp_decay': 0.999,
            'sampling_distribution': 'categorical'
        } if teacher_configs is None else teacher_configs

        initializer(self)

    @overrides
    def _post_load_state_dict(
            self, missing_keys: List[str], unexpected_keys: List[str]
    ) -> Tuple[List[str], List[str]]:
        unexpected_keys = [
            key for key in unexpected_keys
            if not key.startswith('text_field_embedder') and self.text_field_embedder is None
        ]
        return missing_keys, unexpected_keys

    def create_sampled_span_mask(self,
                                 num_spans: int,
                                 gold_slot_spans: List[Dict[str, int]]) -> torch.BoolTensor:
        reserved_span_mask = torch.zeros([1, num_spans], dtype=torch.bool)
        for gold_template in gold_slot_spans:
            for span_ids in gold_template.values():
                chosen_span_idx = np.random.choice(span_ids)
                reserved_span_mask[0, chosen_span_idx] = True
        sampled_span_mask = self.span_sampling_distribution.sample([1, num_spans]).bool()

        full_sampled_span_mask = sampled_span_mask | reserved_span_mask
        # Make sure that there is at least 1 valid span
        if not full_sampled_span_mask.all().item():
            full_sampled_span_mask[0, np.random.choice(list(range(num_spans)))] = True

        return full_sampled_span_mask

    def forward(self,
                metadata: List[Dict[str, Any]],
                template_type: torch.IntTensor,
                spans: Optional[torch.LongTensor] = None,
                span_types: Optional[torch.IntTensor] = None,
                event_types: Optional[torch.IntTensor] = None,
                text: Optional[TextFieldTensors] = None,
                sequence: Optional[Dict[str, torch.Tensor]] = None,
                non_span_slots: Optional[torch.IntTensor] = None,
                allowed_non_span_labels: Optional[torch.IntTensor] = None,
                gold_template_span_labels: Optional[torch.IntTensor] = None,
                gold_template_non_span_labels: Optional[torch.IntTensor] = None,
                gold_span_labels: Optional[torch.Tensor] = None,
                candidate_spans: Optional[torch.Tensor] = None,
                candidate_span_mask: Optional[torch.Tensor] = None,
                sentence_ranges: Optional[torch.LongTensor] = None,
                is_training: bool = True,
                **kwargs) -> Dict[str, Any]:
        assert len(metadata) == 1, "Batch size must be 1 for this model."
        # DEBUG LINES BELOW
        print(f"Print gold_template_span_labels for this example: {gold_template_span_labels}")
        # Constants
        template_type_str = self.vocab.get_token_from_index(index=template_type.item(), namespace='template_labels')
        none_slot_type_index = self.vocab.get_token_index('none', namespace='slot_types')
        none_non_span_label_index = self.vocab.get_token_index('none', namespace='non_span_slot_labels')

        # It's possible no spans are provided as input. Currently, this is
        # only supported for inference, and this will occur when we have no
        # upstream spans to work with. If this happens, we still need to
        # predict *something* (and may still want to validate against gold spans)
        if spans is None and not span_finding:
            assert not is_training, "Model is in training mode, but no spans were provided to the forward method!"
            if 'gold_span_slot_sets' in metadata[0]:
                entry_k, _ = metadata[0]['instance_id'].split(':')
                gold_slot_sets = metadata[0]['gold_span_slot_sets']
                gold_cluster_slot_sets: List[Set[StrSlot]] = self.replace_span_slot_sets_mention_with_cluster(
                    span_slot_sets=metadata[0]['gold_span_slot_sets'],
                    span_ssids=metadata[0]['span_ssid']
                )
                if 'proxy_slot_mention_match_metric' in self.metrics:
                    self.metrics['proxy_slot_mention_match_metric'](gold_slot_sets=gold_slot_sets,
                                                                    predicted_slot_sets=[])
                if 'proxy_non_span_slot_match_metric' in self.metrics:
                    self.metrics['proxy_non_span_slot_match_metric'](
                        gold_slot_sets=metadata[0]['gold_non_span_slot_sets'],
                        predicted_slot_sets=[])
                if 'proxy_slot_cluster_match_metric' in self.metrics:
                    self.metrics['proxy_slot_cluster_match_metric'](gold_slot_sets=gold_cluster_slot_sets,
                                                                    predicted_slot_sets=[])
                if 'proxy_template_match_metric' in self.metrics:
                    self.metrics['proxy_template_match_metric'](
                        gold_templates=[(entry_k, template_type_str) for _ in
                                        range(len(metadata[0]['gold_span_slot_sets']))],
                        predicted_templates=[]
                    )

            doc_key, _ = metadata[0]['instance_id'].split(':')
            return {
                'predicted_span_slot_sets': [[]],
                'predicted_non_span_slot_sets': [[]],
                'predicted_cluster_span_slot_sets': [[]],
                # Assumes we're working with MUC; not an issue on Granular
                'predicted_muc_template_dict': [{doc_key: []}]
            }

        assert not (
                (text is not None and self.text_field_embedder is not None)
                and sequence is not None
        ), 'Cannot use both text and sequence features.'

        if sequence is not None:
            # Shape: (num_layers, batch_size, seq_len, embed_size)
            text_embeddings = sequence['sequence_tensor']
            text_mask = text['sequence_mask']
        elif text is not None and self.text_field_embedder is not None:
            # Shape: (num_layers, batch_size, document_length, embedding_size)
            if self.use_checkpoint:
                text_embeddings = checkpoint(self.text_field_embedder, text)
            else:
                text_embeddings = self.text_field_embedder(text)
            # Shape: (batch_size, seq_len)
            text_mask = text['tokens']['mask']
        else:
            raise ValueError('Must use either text or sequence features.')

        assert len(text_embeddings.shape) == 3, 'Text embeddings must be 3-dimensional'

        text_embeddings = self.lexical_dropout(text_embeddings)
        # Shape: (batch_size, seq_len)
        # text_mask = text['sequence_mask']

        # Shape: (document_length, embedding_size)
        position_embeddings = generate_sinusoidal_features(text_embeddings)

        device = text_embeddings.device
        batch_size, document_length, embedding_size = text_embeddings.shape

        # This is just the number of slots to be predicted
        # that do not take spans as fillers
        if non_span_slots is not None:
            # The number of valid labels across *all* non-spanset-valued slots
            num_non_span_slot_labels = self.vocab.get_vocab_size(namespace='non_span_slot_labels')
            num_non_spans = non_span_slots.shape[1]
            # The following lines create a mask for each non-spanset-valued slot that
            # is true only at positions corresponding to valid values for that slot.
            # This is used to constrain predictions further on.
            # Shape: (batch_size, num_non_spans, num_non_span_slot_labels)
            allowed_non_span_label_mask = torch.zeros((1, num_non_spans, num_non_span_slot_labels), dtype=bool,
                                                      device=device)
            none_label = self.vocab.get_token_index(token='none', namespace='non_span_slot_labels')
            none_label_tensor = torch.full(allowed_non_span_labels.shape, none_label, device=device)
            allowed_non_span_label_mask.scatter_(dim=2,
                                                 index=torch.where(allowed_non_span_labels >= 0,
                                                                   allowed_non_span_labels, none_label_tensor),
                                                 src=torch.ones(allowed_non_span_labels.shape, dtype=bool,
                                                                device=device))
            # Embeddings for non-spanset-valued slots
            # Currently just the embedding for that slot
            # Shape: (batch_size, graph_output_dim)
            if self.use_parallel:
                non_span_embeddings = self.non_span_embeddings(non_span_slots.to('cuda:0')).to('cuda:1')
            else:
                non_span_embeddings = self.non_span_embeddings(non_span_slots)
        else:
            num_non_spans = 0
            allowed_non_span_label_mask = None
            non_span_embeddings = torch.empty(0, device=device)

        num_spans = spans.shape[1]
        # Shape: (batch_size, num_spans)
        valid_span_mask = spans[:, :, 0] >= 0
        if num_spans > 1:
            valid_span_mask = valid_span_mask.squeeze(-1)
        if (
                is_training
                and self.span_sampling_distribution is not None
                and not self.full_span_sampling_distribution.sample().bool().item()
        ):
            sampled_span_mask = self.create_sampled_span_mask(
                num_spans=num_spans, gold_slot_spans=metadata[0]['gold_slot_spans']).to(device)
            span_mask = valid_span_mask & sampled_span_mask
        else:
            span_mask = valid_span_mask

        # Shape: (batch_size, num_spans, 2)
        spans = F.relu(spans.float()).long()

        # Shape: (batch_size, num_spans, 2 * encoding_dim)
        endpoint_span_embeddings = self.endpoint_span_extractor(text_embeddings, spans)
        # Shape: (batch_size, num_spans, embedding_size)
        attended_span_embeddings = self.attentive_span_extractor(text_embeddings, spans)

        # Shape: (batch_size, num_spans, emebedding_size + 2 * encoding_dim + 3 * feature_size)
        raw_span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        # Shape: (batch_size, num_spans, graph_dim)
        span_embeddings = self.compression_feedforward(raw_span_embeddings)

        # Shape: (batch_size, num_spans, graph_dim)
        if span_types is not None:
            span_type_embeddings = self.span_type_embeddings(span_types)
        else:
            span_type_embeddings = 0.
        if event_types is not None:
            event_type_embeddings = self.event_type_embeddings(event_types)
        else:
            event_type_embeddings = 0.
        span_positional_embeddings = position_embeddings[spans[:, :, 0]]
        # Augment spans with different features
        span_additive_embeddings = (
                span_type_embeddings + event_type_embeddings + span_positional_embeddings
        )
        # Shape: (batch_size, graph_dim)
        template_embeddings = self.template_embeddings(template_type)

        # Shape: (batch_size, num_spans, graph_dim)
        graph_span_embeddings = span_embeddings + span_additive_embeddings

        # Even if the GNN is used, non-spans are not encoded with it
        graph_non_span_embeddings = non_span_embeddings

        # Initialize the memory embeddings as all 0s - nothing in the memory state
        # Shape: (batch_size, num_spans + num_non_spans + 1, graph_dim)
        memory_embeddings = torch.zeros(batch_size, num_spans + num_non_spans + 1, self.graph_encoder.get_input_dim(),
                                        device=device)
        # Shape: (batch_size, num_spans + num_non_spans + 1)
        graph_mask = torch.cat([torch.ones(batch_size, 1, device=device, dtype=torch.bool),
                                span_mask,
                                torch.ones(batch_size, num_non_spans, device=device, dtype=torch.bool)], dim=1)
        # Shape: (batch_size, 1, graph_dim)
        revised_template_embeddings = template_embeddings.unsqueeze(1)

        # Shape: (batch_size, num_spans + num_non_span + 1, graph_dim)
        graph_embeddings = torch.cat([
            revised_template_embeddings, graph_span_embeddings, graph_non_span_embeddings
        ], dim=1) + memory_embeddings

        # Some output variables
        predicted_span_slot_sets: List[Set[IntSlot]] = []
        predicted_non_span_slot_sets: List[Set[StrSlot]] = []
        output_dict = {
            'predicted_span_slot_sets': [predicted_span_slot_sets],
            'predicted_non_span_slot_sets': [predicted_non_span_slot_sets]
        }

        # Starts the iteration
        # This part is not optimized for processing more than 1 document - batch_size = 1
        eos: Optional[bool] = None
        template_teacher: Optional[TemplateTeacher] = TemplateTeacher(
            gold_span_slot_sets=metadata[0]['gold_span_slot_sets'],
            gold_non_span_slot_sets=metadata[0]['gold_non_span_slot_sets'],
            span_slot_criterion_config=self.xentropy_span_slot_config,
            non_span_slot_criterion_config=self.xentropy_non_span_slot_config,
            gold_template_span_labels=gold_template_span_labels,
            gold_template_non_span_labels=gold_template_non_span_labels,
            teacher_state=self.teacher_state,
            span_reuse_discount_factor=self.span_reuse_discount_factor,
            none_slot_type_index=none_slot_type_index,
            **self.teacher_configs
        ) if is_training else None
        accumulated_loss: torch.Tensor = torch.tensor(0.0, device=device)
        iteration_num: int = 0
        # Memory inits
        memory_hx = memory_embeddings
        span_use_counts = torch.zeros(batch_size * num_spans, device=device)
        while (eos is None or not eos) and (iteration_num < self.iteration_cutoff):
            # Shape: (batch_size, num_spans + num_non_spans + 1, graph_dim)
            graph_state = self.graph_encoder(inputs=graph_embeddings, mask=graph_mask)
            # Shape: (batch_size, num_spans + num_non_spans, graph_dim)
            if self.pairwise_scoring:
                slot_embeddings = self.slot_feedforward(torch.cat([
                    graph_state[:, 0, :].view(batch_size, 1, -1).expand_as(graph_state[:, 1:, :]),
                    graph_state[:, 1:, :]
                ], dim=2))
            else:
                slot_embeddings = self.slot_feedforward(graph_state[:, 1:, :])

            # Note: we're scoring only span-valued slots here
            # Shape: (batch_size, num_spans, num_slot_types)

            span_slot_scores = torch.matmul(slot_embeddings[:, :num_spans, :], self.slot_type_embeddings.weight.T)
            non_span_slot_scores = torch.matmul(slot_embeddings[:, num_spans:, :],
                                                self.non_span_slot_label_embeddings.weight.T)

            predicted_span_slots, predicted_non_span_slots = self.template_set_decoder.decode(
                template_type=template_type,
                span_types=span_types,
                span_slot_logits=span_slot_scores,
                span_mask=span_mask,
                non_span_slots=non_span_slots,
                non_span_slot_logits=non_span_slot_scores,
                non_span_slot_logits_mask=allowed_non_span_label_mask,
                is_training=is_training)
            has_predicted_slots: bool = len(predicted_span_slots) + len(predicted_non_span_slots) > 0

            memory_state_to_update: List[int] = [none_slot_type_index for _ in range(num_spans)]
            for span_idx, slot_type_str in predicted_span_slots:
                memory_state_to_update[span_idx - 1] = self.vocab.get_token_index(token=slot_type_str,
                                                                                  namespace='slot_types')

            non_span_memory_state_to_update: List[int] = [none_non_span_label_index for _ in range(num_non_spans)]
            for non_span_slot, non_span_val in predicted_non_span_slots:
                non_span_slot_idx = metadata[0]['non_span_slots'].index(non_span_slot)
                non_span_memory_state_to_update[non_span_slot_idx] = self.vocab.get_token_index(token=non_span_val,
                                                                                                namespace='non_span_slot_labels')

            # Shape: (batch_size, 2)
            eos_prior = self.eos_feedforward(graph_state[:, 0, :])
            # Shape: (batch_size, 2)
            # eos_likelihood = torch.stack([
            #     (
            #             span_slot_scores[:, :, self.vocab.get_token_index('none', namespace='slot_types'):
            #             ].logsumexp(dim=2) / (self.vocab.get_vocab_size('slot_types') - 2)
            #     ).logsumexp(dim=1),
            #     span_slot_scores[:, :, self.vocab.get_token_index('none', namespace='slot_types')].logsumexp(dim=1),
            # ], dim=1)
            # Shape: (batch_size, 2)
            # eos_posterior = eos_prior + eos_likelihood
            eos_posterior = eos_prior

            eos: bool = not has_predicted_slots
            if is_training:
                # Obtain the teacher's eos state
                gold_eos = template_teacher.get_eos()
                # Shape: (batch_size)
                gold_eos_tensor = torch.tensor([gold_eos], device=device, dtype=torch.long)

                eos_loss = self.eos_criterion(eos_posterior, gold_eos_tensor)

                if not gold_eos:  # The teacher believes there is a template to predict
                    ref_template_id, ref_span_slot_set, ref_non_span_slot_set = \
                        template_teacher.get_reference_action(
                            span_slot_scores=span_slot_scores,
                            span_mask=span_mask,
                            non_span_slot_scores=non_span_slot_scores if num_non_spans > 0 else None,
                            span_use_counts=span_use_counts
                        )
                    # Determines whether to roll out the teacher's memory
                    if (
                            self.expert_roll_out is None
                            or len(memory_state_to_update) + len(non_span_memory_state_to_update) == 0
                            or self.expert_roll_out_distribution.sample().item() >= self.expert_roll_out
                    ):
                        memory_state_to_update: List[int] = [none_slot_type_index for _ in
                                                             range(num_spans)]
                        non_span_memory_state_to_update: List[int] = [none_non_span_label_index for _ in
                                                                      range(num_non_spans)]
                        for span_idx, slot_type_str in ref_span_slot_set:
                            memory_state_to_update[span_idx - 1] = self.vocab.get_token_index(token=slot_type_str,
                                                                                              namespace='slot_types')
                        for non_span_slot, non_span_val in ref_non_span_slot_set:
                            non_span_slot_idx = metadata[0]['non_span_slots'].index(non_span_slot)
                            non_span_memory_state_to_update[non_span_slot_idx] = self.vocab.get_token_index(
                                token=non_span_val,
                                namespace='non_span_slot_labels')

                    # Computes the slot loss w.r.t. the reference template
                    # Mask out those invalid spans
                    reference_template = gold_template_span_labels[
                        0, ref_template_id
                    ].view(batch_size * num_spans).contiguous().clone()
                    reference_template[~span_mask[0]] = self.xentropy_span_slot_ignore_index
                    span_slot_loss = self.span_slot_criterion(
                        span_slot_scores.view(batch_size * num_spans, -1),
                        reference_template
                    )
                    if num_spans > 0 and self.span_reuse_discount_factor is not None:
                        span_use_counts += (reference_template != none_slot_type_index)
                        # -1 to get *reuse* counts; relu because we want the exponent's floor to be 0
                        discount_factor_exponent = torch.relu(span_use_counts - 1)
                        span_slot_loss = torch.mean(
                            span_slot_loss * (self.span_reuse_discount_factor ** discount_factor_exponent))
                    if num_non_spans > 0:
                        non_span_slot_loss = self.non_span_slot_criterion(
                            non_span_slot_scores.view(batch_size * num_non_spans, -1),
                            gold_template_non_span_labels[0, ref_template_id].view(batch_size * num_non_spans)
                        )
                    else:
                        non_span_slot_loss = 0.

                else:  # The teacher asks to stop
                    reference_template = torch.tensor(
                        [self.vocab.get_token_index(token='none', namespace='slot_types')] * batch_size * num_spans,
                        device=device, dtype=torch.long
                    )
                    reference_template[~span_mask[0]] = self.xentropy_span_slot_ignore_index
                    span_slot_loss = self.span_slot_criterion(
                        span_slot_scores.view(batch_size * num_spans, -1),
                        reference_template
                    )
                    if num_spans > 0 and self.span_reuse_discount_factor is not None:
                        # No need to update span_use_counts here, since there
                        # are no more reference templates
                        discount_factor_exponent = torch.relu(span_use_counts - 1)
                        span_slot_loss = torch.mean(
                            span_slot_loss * (self.span_reuse_discount_factor ** discount_factor_exponent))
                    if num_non_spans > 0:
                        non_span_slot_loss = self.non_span_slot_criterion(
                            non_span_slot_scores.view(batch_size * num_non_spans, -1),
                            torch.tensor(
                                [
                                    self.vocab.get_token_index(token='none',
                                                               namespace='non_span_slot_labels')
                                ] * batch_size * num_non_spans,
                                device=device, dtype=torch.long
                            )
                        )
                    else:
                        non_span_slot_loss = 0.

                # loss = eos_loss + slot_loss
                # loss = slot_loss
                accumulated_loss += (
                                            eos_loss + span_slot_loss + non_span_slot_loss
                                    ) * self.grad_discount_factor ** iteration_num

                # During the training, the model would be following the teacher's action
                eos = gold_eos

            if not eos:  # The decoding should be added to the results.
                if has_predicted_slots or is_training:  # We should penalize for such cases...
                    predicted_span_slot_sets.append(predicted_span_slots)
                    predicted_non_span_slot_sets.append(set(predicted_non_span_slots))

                # Update the memory state
                # Update the graph embeddings with the memory cell
                # Shape: (batch_size, num_span_states_to_update)
                memory_state_to_update_tensor = torch.tensor([memory_state_to_update], device=device, dtype=torch.long)
                # Shape: (batch_size, num_non_span_states_to_update)
                non_span_memory_state_to_update_tensor = torch.tensor([non_span_memory_state_to_update], device=device,
                                                                      dtype=torch.long)
                # Shape: (batch_size, num_spans + 1, graph_dim)
                infused_memory_features = torch.cat([
                    self.template_memory_update_feedforward(graph_state[:, 0, :].view(batch_size, 1, -1)),
                    self.slot_memory_update_feedforward(torch.cat([
                        graph_state[:, 1:, :].view(batch_size, num_spans + num_non_spans, -1),
                        torch.cat([self.slot_type_embeddings(memory_state_to_update_tensor),
                                   self.non_span_slot_label_embeddings(non_span_memory_state_to_update_tensor)
                                   ], dim=1)
                    ], dim=-1))
                ], dim=1)
                # Shape: (batch_size, num_spans + 1, graph_dim)
                memory_hx = self.memory_updator(
                    infused_memory_features.squeeze(0),
                    memory_hx.squeeze(0)
                ).unsqueeze(0)

                # Update the graph embeddings
                # Shape: (batch_size, 1, graph_dim)
                revised_template_embeddings = graph_state[:, 0, :].view(batch_size, 1, -1)
                # Shape: (batch_size, num_spans + 1, graph_dim)
                graph_embeddings = torch.cat([
                    revised_template_embeddings, graph_span_embeddings, graph_non_span_embeddings,
                ], dim=1) + memory_hx

            iteration_num += 1

        # For span-valued slots, we need to group spans into clusters to compute
        # spanset-level match metrics. For non-span-valued slots, we assume there
        # is only ever a single filler per slot, and so we treat all such fillers
        # as singleton clusters (see "assumption" below).
        if "pred_span_ssid" in metadata[0]:
            pred_span_ssid = "pred_span_ssid"
        else:
            pred_span_ssid = "span_ssid"

        span_ssids = metadata[0][pred_span_ssid]

        predicted_cluster_span_slot_sets: List[Set[StrSlot]] = self.replace_span_slot_sets_mention_with_cluster(
            span_slot_sets=predicted_span_slot_sets, span_ssids=span_ssids)
        # Add non-span-valued slots
        predicted_cluster_span_slot_sets = [predicted_cluster_span_slot_set | predicted_non_span_slot_set
                                            for (predicted_cluster_span_slot_set, predicted_non_span_slot_set)
                                            in zip(predicted_cluster_span_slot_sets, predicted_non_span_slot_sets)]
        output_dict['predicted_cluster_span_slot_sets'] = [predicted_cluster_span_slot_sets]

        # Compute metrics
        if 'gold_span_slot_sets' in metadata[0]:
            # TODO(@Yunmo): span-level slot match should consider those spans that were masked out, ignore for now.
            gold_slot_sets = [gold_span_slots | gold_non_span_slots
                              for (gold_span_slots, gold_non_span_slots) in
                              zip(metadata[0]['gold_span_slot_sets'], metadata[0]['gold_non_span_slot_sets'])]
            predicted_slot_sets = [predicted_span_slot_set | predicted_non_span_slot_set
                                   for (predicted_span_slot_set, predicted_non_span_slot_set) in
                                   zip(predicted_span_slot_sets, predicted_non_span_slot_sets)]
            if 'proxy_slot_mention_match_metric' in self.metrics:
                self.metrics['proxy_slot_mention_match_metric'](gold_slot_sets=gold_slot_sets,
                                                                predicted_slot_sets=predicted_slot_sets)
            if 'proxy_non_span_slot_match_metric' in self.metrics:
                self.metrics['proxy_non_span_slot_match_metric'](gold_slot_sets=metadata[0]['gold_non_span_slot_sets'],
                                                                 predicted_slot_sets=predicted_non_span_slot_sets)
            gold_cluster_slot_sets: List[Set[StrSlot]] = self.replace_span_slot_sets_mention_with_cluster(
                span_slot_sets=metadata[0]['gold_span_slot_sets'],
                span_ssids=metadata[0]['span_ssid']
            )
            gold_cluster_slot_sets = [gold_span_slots | gold_non_span_slots
                                      for (gold_span_slots, gold_non_span_slots) in
                                      zip(gold_cluster_slot_sets, metadata[0]['gold_non_span_slot_sets'])]
            # Assumption: there is never more than one filler for a non-span valued slot,
            # meaning that these fillers do not have to be clustered as the spans do. This
            # is true for the Granular data, but may not be true of other datasets. For these
            # special slots only, slot match will always be equivalent to cluster match
            if 'proxy_slot_cluster_match_metric' in self.metrics:
                self.metrics['proxy_slot_cluster_match_metric'](gold_slot_sets=gold_cluster_slot_sets,
                                                                predicted_slot_sets=predicted_cluster_span_slot_sets)
            entry_k, _ = metadata[0]['instance_id'].split(':')
            if 'proxy_template_match_metric' in self.metrics:
                self.metrics['proxy_template_match_metric'](
                    gold_templates=[(entry_k, template_type_str) for _ in
                                    range(len(metadata[0]['gold_span_slot_sets']))],
                    predicted_templates=[(entry_k, template_type_str) for _ in range(len(predicted_span_slot_sets))]
                )

        # Compute BETTER official metric
        # Avoids spending too much time on computing official metrics during the training
        if not is_training:
            # TODO(@Yunmo): the decoding option should be decoupled from the metric.
            if 'better' in self.metrics:
                # This function assumes that the batch_size is 1
                predicted_bpjson_template_dict = self.convert_template_sets_to_bpjson_template_dict(
                    instance_id=metadata[0]['instance_id'],
                    template_type=template_type_str,
                    span_slot_sets=predicted_span_slot_sets,
                    non_span_slot_sets=predicted_non_span_slot_sets,
                    spans=spans[0].detach().tolist(),
                    span_ssids=metadata[0]['span_ssid'],
                )
                output_dict['predicted_bpjson_template_dict'] = [predicted_bpjson_template_dict]
                self.metrics['better'](predictions=(metadata[0]['data_path'], predicted_bpjson_template_dict))

            if 'muc' in self.metrics:
                # This function also assumes that the batch_size is 1
                predicted_muc_template_dict = self.convert_template_sets_to_muc_templates(
                    template_type=template_type_str,
                    slot_sets=predicted_span_slot_sets,
                    spans=spans[0].detach().tolist(),
                    metadata=metadata[0]
                )
                output_dict['predicted_muc_template_dict'] = [predicted_muc_template_dict]
                self.metrics['muc'](predictions=predicted_muc_template_dict,
                                    pred_src_file=metadata[0]['data_path'],
                                    normalize_role=False)
                # DEBUG line for prediction template dict
                print(f"predicted_muc_template_dict: {predicted_muc_template_dict}")
                
            if 'iterx_scirex' in self.metrics:
                # This function again assumes that the batch_size is 1
                predicted_scirex_template_dict = self.convert_template_sets_to_iterx_scirex_templates(
                    template_type=template_type_str,
                    slot_sets=predicted_span_slot_sets,
                    spans=spans[0].detach().tolist(),
                    metadata=metadata[0]
                )
                output_dict['predicted_scirex_template_dict'] = [predicted_scirex_template_dict]
                self.metrics['iterx_scirex'](predictions=predicted_scirex_template_dict,
                                             pred_src_file=metadata[0]['data_path'],
                                             dedup=False,
                                             cluster_substr=False,
                                             normalize_role=False)

        if is_training:
            loss = accumulated_loss / iteration_num
            output_dict['loss'] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        better_metrics = self.metrics['better'].get_metric(reset=reset) if 'better' in self.metrics else None
        muc_metrics = self.metrics['muc'].get_metric(reset=reset) if 'muc' in self.metrics else None
        iterx_scirex_metrics = (
            self.metrics['iterx_scirex'].get_metric(reset=reset) if 'iterx_scirex' in self.metrics else None
        )
        proxy_slot_match_metric = (
            self.metrics['proxy_slot_mention_match_metric'].get_metric(reset=reset)
            if 'proxy_slot_mention_match_metric' in self.metrics else None
        )
        proxy_slot_cluster_metric = (
            self.metrics['proxy_slot_cluster_match_metric'].get_metric(reset=reset)
            if 'proxy_slot_cluster_match_metric' in self.metrics else None
        )
        proxy_template_match_metric = (
            self.metrics['proxy_template_match_metric'].get_metric(reset=reset)
            if 'proxy_template_match_metric' in self.metrics else {}
        )
        span_finding_metric = (
            self.metrics['span_finding'].get_metric(reset=reset) if 'span_finding' in self.metrics else None
        )
        # proxy_non_span_slot_match_metric = self.metrics['proxy_non_span_slot_match_metric'].get_metric(reset=reset)
        output = {
            **(
                {
                    f'sf_{k}_{k2[0]}': v2
                    for k, v in span_finding_metric.items()
                    for k2, v2 in v.items()
                }
                if span_finding_metric else {}
            ),
            # scores for non-span-valued (NSV) slots only
            # currently commented out for testing on MUC, which
            # has not NSV slots
            #
            # 'nsv-prec': proxy_non_span_slot_match_metric[0],
            # 'nsv-rec': proxy_non_span_slot_match_metric[1],
            # 'nsv-f1': proxy_non_span_slot_match_metric[2],
            **proxy_template_match_metric
        }
        if proxy_slot_match_metric is not None:
            output['prec'] = proxy_slot_match_metric[0]
            output['rec'] = proxy_slot_match_metric[1]
            output['f1'] = proxy_slot_match_metric[2]
        if proxy_slot_cluster_metric is not None:
            output['sprec'] = proxy_slot_cluster_metric[0]
            output['srec'] = proxy_slot_cluster_metric[1]
            output['sf1'] = proxy_slot_cluster_metric[2]

        if better_metrics is not None and reset:
            for k, v in better_metrics.items():
                output[k] = v

        if muc_metrics is not None and reset:
            for k, v in muc_metrics.items():
                output[k] = v

        if iterx_scirex_metrics is not None and reset:
            for k, v in iterx_scirex_metrics.items():
                output[k] = v
        return output

    @staticmethod
    def replace_span_slot_sets_mention_with_cluster(
            span_slot_sets: List[Set[IntSlot]],
            span_ssids: List[str]
    ) -> List[Set[StrSlot]]:
        return [
            {
                (span_ssids[span_idx - 1], slot_name)
                for span_idx, slot_name in span_slot_set
            }
            for span_slot_set in span_slot_sets
        ]

    @staticmethod
    def convert_template_sets_to_bpjson_template_dict(
            instance_id: str,
            template_type: str,
            span_slot_sets: List[Set[IntSlot]],
            non_span_slot_sets: List[Set[StrSlot]],
            spans: List[Tuple[int, int]],
            span_ssids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        from better.utils.bpjson_utils import BPJsonSpan

        doc_key, _ = instance_id.split(':')
        templates: List[Dict[str, Any]] = []
        for span_slot_set in span_slot_sets:
            template = {
                'template_type': template_type,
                'template_slots': {}
            }
            for span_idx, slot_type in span_slot_set:
                span_idx -= 1
                if slot_type not in template['template_slots']:
                    template['template_slots'][slot_type] = []
                template['template_slots'][slot_type].append(BPJsonSpan(
                    range=spans[span_idx],
                    type=slot_type,
                    ssid=span_ssids[span_idx],
                    str='unspecified'
                ))
            templates.append(template)
        for i, non_span_slot_set in enumerate(non_span_slot_sets):
            for slot_type, slot_val in non_span_slot_set:
                # string- and bool-valued slots should only be filled once
                assert slot_type not in template['template_slots']
                if slot_val == 'true':
                    templates[i]['template_slots'][slot_type] = True
                elif slot_val == 'false':
                    templates[i]['template_slots'][slot_type] = False
                elif slot_val != 'none':
                    templates[i]['template_slots'][slot_type] = slot_val

        return {doc_key: templates}

    @staticmethod
    def convert_template_sets_to_iterx_scirex_templates(
            template_type: str,
            slot_sets: List[Set[IntSlot]],
            spans: List[Tuple[int, int]],
            metadata: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        doc_key, _ = metadata['instance_id'].split(':')
        document = metadata['raw_doc_ref']
        templates: List[Dict[str, Any]] = []
        for span_slot_set in slot_sets:
            template = {}
            for span_idx, slot_type in span_slot_set:
                span_idx -= 1
                if slot_type not in template:
                    template[slot_type] = set()
                span_start_tok = spans[span_idx][0]
                span_end_tok = spans[span_idx][1]
                span_text = ' '.join(document[span_start_tok:span_end_tok + 1])
                template[slot_type].add(span_text)
            template = {role: [[mention] for mention in cluster] for role, cluster in template.items()
                        if role != 'incident_type'}
            template['incident_type'] = template_type
            templates.append(template)
        return {doc_key: templates}

    @staticmethod
    def convert_template_sets_to_muc_templates(
            template_type: str,
            slot_sets: List[Set[IntSlot]],
            spans: List[Tuple[int, int]],
            metadata: Dict[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        doc_key, _ = metadata['instance_id'].split(':')
        templates: List[Dict[str, Any]] = []
        for span_slot_set in slot_sets:
            template = {
                'incident_type': template_type,
            }
            for span_idx, slot_type in span_slot_set:
                span_idx -= 1
                if slot_type not in template:
                    template[slot_type] = set()
                span_start_tok = spans[span_idx][0]
                span_end_tok = spans[span_idx][1]
                span_start_char = metadata['raw_doc_ref']['tok2char'][span_start_tok][0]
                span_end_char = metadata['raw_doc_ref']['tok2char'][span_end_tok][-1]
                span_text = metadata['raw_doc_ref']['doctext'][span_start_char:span_end_char + 1]
                # span = [span_text, span_start_char, span_end_char, span_start_tok, span_end_tok, slot_type]
                # currently assuming singleton entities
                span = normalize_string(span_text)
                template[slot_type].add(span)
            template = {role: [[mention] for mention in cluster] for role, cluster in template.items()
                        if role != 'incident_type'}
            template['incident_type'] = template_type
            templates.append(template)
        return {doc_key: templates}

    def _to_params(self) -> Dict[str, Any]:
        return super(IterativeTemplateExtraction, self)._to_params()

    default_predictor = 'iterative_template_extraction'