# Contextualized encoder configs
local transformer_model = "t5-large";
local transformer_dim = 1024; 

# How encoders chunk documents into smaller pieces during encoding
local max_length = 1024;

# Data paths
local definition_path = "resources/data/scirex/definitions.json";
local vocabulary_path = "resources/data/scirex/vocabulary";

# Gold data for training and evaluation
# Path format: <path_to_data>;<annotation_name>
# For Span Finder outputs, we have annotated Concrete files using "Span Finder"
# Therefore, using "Span Finder" means taking the Span Finder outputs as **upstream spans**
# Without any <annotation_name>, the default is to use the "gold spans" annotations (usually for training)
local train_data_path = "resources/data/scirex/preprocessed/sf-outputs/train";
local dev_data_path = "resources/data/scirex/preprocessed/sf-outputs/dev;Span Finder";
local test_data_path = "resources/data/scirex/preprocessed/sf-outputs/test;Span Finder";

# Model configs
local lexical_dropout = 0.2;
local feature_dropout = 0.3;
local iteration_cutoff = 10;  # 4 docs exceed this cutoff as 16: 2, 49: 1, 19:1
local span_slot_none_weight = 1.0;
local span_sampling_rate = 0.2;
local graph_encoder = {
  "type": "pytorch_transformer",
  "input_dim": transformer_dim,
  "num_layers": 1,
  "feedforward_hidden_dim": 2048,
  "num_attention_heads": 128,
  "dropout_prob": 0.2,
  "activation": "gelu"
};

local initializer = {
  "regexes": [
    [".*_span_updating_gated_sum.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Feedforwards
    [".*linear_layers.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # SelfAttentiveSpanExtractor
    [".*_global_attention.*weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Embeddings
    ["template_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["event_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["event_arg_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["span_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    ["slot_type_embeddings.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    # Graph encoder
    [".*_transformer.*linear.*\\.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*self_attn.*\\.in_proj_weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*self_attn.*\\.weight", {"mean": 0, "std": 0.02, "type": "normal"}],
    [".*_transformer.*norm.*\\.weight", {"type": "constant", "val": 1}],
    [".*_transformer.*\\.bias", {"type": "zero"}],
  ]
};

local metrics = {
  "iterx_scirex": {
    "type": "iterx_scirex",
    "doc_path": {
      [dev_data_path]: dev_data_path,
    }
  },
};

local gen_dataset_reader(is_training, training_with_empty_docs) = {
  "type": "concrete_extraction",
  "definition_file": definition_path,
  "extraction_task": "scirex-re",
  "is_training": is_training,
  "training_with_empty_docs": training_with_empty_docs,
  "token_indexers": {
    "tokens": {
      "type": "pretrained_transformer_mismatched",
      "model_name": transformer_model,
      "max_length": max_length
    },
  },
  "max_token_cutoff": 8800,
  # Debug options
  "max_instances": 10,
  # "verbose": true
};

local teacher_configs ={
  "temp_start": 10.0,
  "temp_end": 0.5,
  "temp_decay": 0.9995,
};

local model = {
  "type": "iterative_template_extraction",
  "definition_file": definition_path,
  # 1.0 means no discount for loss
  "grad_discount_factor": 1.0,
  "span_slot_none_weight": span_slot_none_weight,
  "text_field_embedder": {
    "token_embedders": {
      "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length
      }
    }
  },
  "pairwise_scoring": true,
  "span_sampling_rate": span_sampling_rate,
  "expert_roll_out": 0.6,
  "graph_encoder": graph_encoder,
  "iteration_cutoff": iteration_cutoff,
  "lexical_dropout": lexical_dropout,
  "feature_dropout": feature_dropout,
  "initializer": initializer,
  "metrics": metrics,
  "teacher_configs": teacher_configs,
  "template_set_decoder_use_sampling": true,
};

{
  # is_training: true; skip_docs_without_spans: true; skip_docs_without_templates: false
  "dataset_reader": gen_dataset_reader(true, false),
  # is_training: false; skip_docs_without_spans: true; skip_docs_without_templates: true
  # NOTE: if you wish to evaluate on documents with *no* templates, this setting must be changed
  "validation_dataset_reader": gen_dataset_reader(false, true),
  "train_data_path": train_data_path,
  "validation_data_path": dev_data_path,
  "vocabulary": {
    "type": "from_files",
    "directory": vocabulary_path,
  },
  "model": model,
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we are using a span field.
      "sorting_keys": ["text"],
      "batch_size": 1
    }
  },
  "trainer": {
    "num_epochs": 150, # originally 500
    "patience" : 30, # originally 50
    "validation_metric": "+iterx_scirex_f",
    "grad_norm": 1.0,
    "learning_rate_scheduler": {
      "type": "polynomial_decay",
      "warmup_steps": 1000,
      "power": 2.0,
      "end_learning_rate": 1e-9
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-5,
      "weight_decay": 1e-2,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    # Uncomment if logging to Tensorboard
    "callbacks": [
      {
        "type": "tensorboard",
        "should_log_parameter_statistics": true,
        "should_log_learning_rate": true
      }
    ],
    # Enables CUDA
    "cuda_device": 0
  },
  "random_seed": 13370,
  "numpy_seed": 1337,
  "pytorch_seed": 133
}