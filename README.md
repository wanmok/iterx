# Iterative Document-level Information Extraction via Imitation Learning
This repo contains code for the following paper:

- Yunmo Chen, William Gantt, Weiwei Gu, Tongfei Chen, Aaron White, and Benjamin Van Durme.
  2023. [Iterative Document-level Information Extraction via Imitation Learning](https://arxiv.org/abs/2210.06600). In
  Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages
  1858–1874, Dubrovnik, Croatia. Association for Computational Linguistics.

```
@inproceedings{chen-etal-2023-iterative,
    title = "Iterative Document-level Information Extraction via Imitation Learning",
    author = "Chen, Yunmo  and
      Gantt, William  and
      Gu, Weiwei  and
      Chen, Tongfei  and
      White, Aaron  and
      Van Durme, Benjamin",
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

## Codebase Release

We are gradually releasing materials related to our paper. The release includes the following:
- [x] [Model outputs for the IterX model](resources/model_outputs)
- [ ] Metric implementations
- [ ] Code for the IterX model
