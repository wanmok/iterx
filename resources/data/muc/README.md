# MUC-4 Data

## Contents
This directory contains:
-  `preprocessed`: Various preprocessed versions of the train, dev, and test data from the following repository: https://github.com/xinyadu/gtt/. There are four versions, each corresponding to a subdirectory of the `preprocessed` directory:
    - `untokenized`: the files here are taken directly from [here](https://github.com/xinyadu/gtt/tree/master/data/muc/processed). As suggested by the name, they are not tokenized.
    - `untokenized-merged`: same as the files in `untokenized`, except that the templates have been filtered based on the SF output files in `sf-outputs`. Practically, this means that any mention appearing in a template in the gold data that does not *also* appear among the upstream predicted spans for that document gets filtered from that template. We use the `scripts/muc-utils/merge-upstream-spans-into-gold-templates.py` script to do this.
    - `tokenized`: files constructed by running the `tokenize_doc` function from `scripts/data-processing/preprocess-muc-dataset.py` on each file in `untokenized`. In addition to tokenized document text and spans, they contain alignments between the raw and tokenized text, obtained from the `tokenizations` Python package.
    - `tokenized-merged`: same as the files in `tokenized`, except that (1) the templates are taken from `untokenized-merged` rather than from `untokenized`, and (2) the spans are upstream predicted spans, rather than gold spans. These files are obtained by running the `scripts/data-processing/preprocess-muc-dataset.py` with (1) the `untokenized-merged` version of the file as the `input-file` argument and (2) the `sf-outputs` version of the file as the (optional) `upstream-spans-file` argument.
    - `tokenized-sent`: files constructed by running the `tokenize_sent` function from `scripts/data-processing/preprocess-muc-dataset.py` on each file in `untokenized`. This function first splits a document into sentences using Spacy's sentence splitter before running tokenization. This sentence-level data is useful when using SpanFinder to generate candidate spans for the IL model.
    - `sf-outputs`: files identical to the `tokenized` files, but with three additional fields with information about *SpanFinder*-predicted spans. These fields are `all-pred-spans`, `all-pred-span-sets`, and `pred-spans-to-spanset`, corresponding to (respectively), the predicted spans, the predicted span *sets*, and the mappings between these. Currently, we assume singleton span sets, so these mappings are trivial. It is important to note that the spans are *not* merged into the original templates: these still contain the original gold span(set)s. For more details on the SpanFinder-to-IL model pipeline, contact Will Gantt.
- `vocabulary`: Files for constructing an AllenNLP Vocabulary for the MUC readers
- `definitions.json`: MUC template defintions, required by the IL model.
- `docids_event_n.json`: A metadata file taken from the repo linked to above that groups (transformed) MUC document IDs according to the number of templates those documents contain. Used when running the evaluation script (`scripts/muc-utils/eval.py`) on documents containing only a certain number of templates.

A final caveat: even though all of these files feature a `*.json` extension, they are in fact all jsonlines files, with one document per line. This is a naming error I have not yet bothered to correct.

## Data Fields

The data fields in the untokenized files are fairly straightforward, but the fields in the tokenized files require some explanation. I already described the `all-pred-spans`, `all-pred-span-sets`, and `pred-spans-to-spanset` fields above, which feature only in the `sf-outputs` versions of the files. Here are some notes on other fields:
- `all-spans`: all spans in the entire document. In the `tokenized` files, these are the actual gold spans. In the `tokenized-merged` files, these are the upstream spans.
- `all-span-sets`: all span sets in the entire document. With MUC, span sets are determined by the entities that appear in slots: each such entity is considered to be a span set, and there are no other span sets. I have so far made the assumption (though it is one I have not verified) that an entity will never be "split." That is, it will never be the case that a mention will be associated with more than one entity.
- `spans-to-spanset`: A list of indices indicating for each span in `all-spans` the span set in `all-span-sets` to which it belongs.
- `template-spans`: this field actually appears *inside* a template and lists indices into `all-spans` corresponding to the spans that appear in that template.
- `docid`: the document ID
- `doctext`: the untokenized text of the document
- `doctext-tok` the tokenized text of the document
- `char2tok`, `tok2char`: character-to-token and token-to-character mappings for the document text