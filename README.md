# Iterative Document-level Information Extraction via Imitation Learning

This repo contains code for the following paper:

- Yunmo Chen, William Gantt, Weiwei Gu, Tongfei Chen, Aaron White, and Benjamin Van Durme.
    2023. [Iterative Document-level Information Extraction via Imitation Learning](https://arxiv.org/abs/2210.06600). In
          Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics,
          pages
          1858–1874, Dubrovnik, Croatia. Association for Computational Linguistics.

```
@inproceedings{chen-etal-2023-iterative,
    title = "Iterative Document-level Information Extraction via Imitation Learning",
    author = "Chen, Yunmo  and
      Gantt, William  and
      Gu, Weiwei  and
      Chen, Tongfei  and
      White, Aaron  and
      {Van Durme}, Benjamin",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.136",
    pages = "1858--1874",
    abstract = "We present a novel iterative extraction model, IterX, for extracting complex relations, or templates, i.e., N-tuples representing a mapping from named slots to spans of text within a document. Documents may feature zero or more instances of a template of any given type, and the task of template extraction entails identifying the templates in a document and extracting each template{'}s slot values. Our imitation learning approach casts the problem as a Markov decision process (MDP), and relieves the need to use predefined template orders to train an extractor. It leads to state-of-the-art results on two established benchmarks {--} 4-ary relation extraction on SciREX and template extraction on MUC-4 {--} as well as a strong baseline on the new BETTER Granular task.",
}
```

## Codebase Release Progress

We are gradually releasing materials related to our paper. The release includes the following:

- [x] [Model outputs for the IterX model](resources/model_outputs)
- [x] [Metric implementations](iterx/metrics/muc/ceaf_rme.py)
- [x] [Code for the IterX model](iterx/models/iterative_template_extraction.py)

Beyond our initial release, we are also planning to release the following;
however, they are subject to change (in terms of the release date and the content):

- [ ] Trained checkpoints of IterX models
- [ ] Trained checkpoints of Span Finder models
- [ ] Pipeline for running Span Finder and IterX together on new datasets
- [ ] Migrations of Span Finder and IterX to lightning (if requested by enough users)

# Runbook

## Environment Setup

### Set up Python and Poetry

*If you already have a running Python environment, you can skip this section.
We tested our codebase on Python 3.11.*

1. Install Python: for Python installation, you can either manage different Python versions
   through [Conda](https://docs.conda.io/en/latest/miniconda.html) or
   [pyenv](https://github.com/pyenv/pyenv).
   If you were using Conda, we recommend creating a new environment for this project;
2. Install Poetry using command line by
   following [the Poetry installation](https://python-poetry.org/docs#installation);
3. Make sure that you have added Poetry bin directory to your `PATH` environment variable.

### Set up the IterX Codebase

1. Clone IterX codebase to your local directory
2. Create and activate a new virtual environment

   If you were using Conda, you can create a new environment by running the following command:

    ```shell
    conda create -n <your_env_name> python=3.11
    conda activate <your_env_name>
    ```

3. Enter the project root directory and install the dependencies

    ```shell
    cd <project_root>
    poetry install
    ```

## Run IterX

Before running all IterX related commands, please make sure that you have activated the environment by running the
following command *from the project root*. IterX should be compatible with all AllenNLP commands.

```shell
poetry shell
```

### Model Training

You can kick off model training using the same command as ones used in AllenNLP. Here is an example template command:

```shell
PYTHONPATH=./src allennlp train \
  --include-package iterx \
  -s <serialization_dir> \
  <path_to_config_file>
```

If you would like to turn on the BETTER supports, you should also add `better` to `--include-package` argument.

## Run Evaluation Script

We provide a script for evaluating the model outputs using CEAF-RME metrics. There are two modes of the script, one is
to evaluate model outputs and generate corresponding performance scores, the other is to generate prediction
comparison results. They share the same CLI arguments and options. You can use `--help` to get detailed usages.

```shell
PYTHONPATH=./src python scripts/ceaf-scorer.py <score_or_compare> \
  --dataset <dataset_name> \
  <path_to_pred_file> \
  <path_to_ref_file>
```

