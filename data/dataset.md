We recommend to read our [NAACL 2022 paper](https://arxiv.org/abs/2205.02289) as a first step.

# Trainset and Testset Format

For this task we have two JSON-lines file (final-train-set and final-test-set), each line is a JSON corresponding to a "question" in which we expected to get predictions for. Each JSON consists of the following fields:

* doc_id: the ID of the document the sentence was taken from. we expect the same ID in the predictions.
* sentence: the textual sentence containing of multiple drugs.
* spans: the spans of the drugs in the sentence, a list of JSONs of its own, each one consists of:
  - span_id: an ID of the drug in the sentence, used in the gold (and prediction) relations.
  - text: the verbatim of the drug.
  - start: character start offset
  - end: character end offset
  - token_start: token start offset
  - token_start: token end offset
* rels: a list of JSONs corresponding to the gold relations, notice for a single sentence/json-line in the input there may be multiple relations - so for each of these we expect a separate line in the predictions file. Each one consists of:
  - class: one of: "POS" (there is a combination and it has a positive efficacy), "NEG" (there is a combination and it has a negative efficacy) and "COMB" (there is a combination but there isn't enough information to determine the efficacy, even in the wider context). Please notice, the separation between "COMB" and "NEG" was experimental and for evaluation we consider only positive (class=2), "other" - both COMB and NEG (class=1), or no combination (class=0). So in our predictions, use the class numbers (2 for positive, 1 for other and 0 for no relation) to indicate the class.
  - spans: the list of drug span_id's that participate in the relation.
  - is_context_needed: a boolean indicator whether the context was needed to determine the efficacy of the combination or was the sentence enough.
* paragraph: a wider context in cases the sentence itself is not enough to determine the efficacy of the combination (the sentence itself can be found in the paragraph using a simple string lookup).

# Predictions Format

In case you intend to compete in the [leaderboard](https://leaderboard.allenai.org/drug_combo), or just want to train your own model to improve our line of work, your application should produce predictions in JSON-lines format, each JSON-line should correspond to a **relation** (not a sentence, as for each sentence in the input there may be more than one relations). The JSON should consist of:

* doc_id: the ID of the document the relation is extracted for (taken from the input).
* drug_idxs: list of drug ID's that participate in the relation, the IDs should correspond to the drug_id's in the input.
* relation_label: one of: 2 (positive combination), 1 (other combination), 0 (no combination). Notice, as there may be many ways to combine multiple drugs, you may omit the 0 relations - we would assume 0/no-combination for each combo that was not assigned 1/2.

For example:
  ```
  {"doc_id": "1234abcd", "drug_idxs": [0, 1], "relation_label": 2}
  {"doc_id": "1234abcd", "drug_idxs": [2, 3], "relation_label": 1}
  {"doc_id": "5678efgh", "drug_idxs": [0, 1, 2], "relation_label": 2}
  ```
