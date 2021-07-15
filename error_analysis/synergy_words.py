from collections import defaultdict
import jsonlines
from nltk.corpus import words as english_words
import numpy as np
from pathlib import Path
import os

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)

MIN_DOC_FREQUENCY = 5

def construct_english_dictionary_set():
    en_words = set()
    for w in english_words.words():
        en_words.add(w.lower())
    return en_words

def display_top_words(words_with_scores):
    print(f"Words\t\t\tMutual Information Scores")
    for w, s in words_with_scores:
        if len(w) > 7:
            column_space = "\t\t"
        else:
            column_space = "\t\t\t"
        print(f"{w}{column_space}{s}")

if __name__ == "__main__":
    dictionary = construct_english_dictionary_set()

    parent_directory = Path().resolve().parent
    train_file = os.path.join(parent_directory, "data", "train_set.jsonl")
    test_file = os.path.join(parent_directory, "data", "test_set.jsonl")

    train_set = list(jsonlines.open(train_file))
    test_set = list(jsonlines.open(test_file))

    full_dataset = train_set + test_set

    synergy_document_token_counts = defaultdict(int)
    all_token_counts = defaultdict(int)
    doc_appearances = defaultdict(int)
    for doc in full_dataset:
        pos_relations = [rel for rel in doc["rels"] if rel["class"] == "POS"]
        synergistic_relation_in_doc = len(pos_relations) > 0

        sentence_start = doc["paragraph"].find(doc["sentence"])
        context_outside_sentence = doc["paragraph"][:sentence_start] + " " + doc["paragraph"][sentence_start + len(doc["sentence"]):]

        tokens = tokenizer.tokenize(doc["paragraph"])
        # tokens = doc["paragraph"].lower().split()
        for t in tokens:
            if t not in dictionary:
                continue
            all_token_counts[t] += 1
            if synergistic_relation_in_doc:
                synergy_document_token_counts[t] += 1
        for t in set(tokens):
            doc_appearances[t] += 1

    total_synergy_words = sum(synergy_document_token_counts.values())
    total_words = sum(all_token_counts.values())

    pointwise_mutual_informations = []
    for word in synergy_document_token_counts:
        if doc_appearances[word] < MIN_DOC_FREQUENCY:
            continue
        p_w_given_synergy = synergy_document_token_counts[word] / float(total_synergy_words)
        p_w = all_token_counts[word] / float(total_words)
        pmi = np.log(p_w_given_synergy / p_w)
        pointwise_mutual_informations.append((word, pmi))

    pointwise_mutual_informations = sorted(pointwise_mutual_informations, key=lambda x: x[1], reverse=True)
    top_words = pointwise_mutual_informations[:20]
    bottom_words = pointwise_mutual_informations[-20:]
    print(f"Words with greatest mutual information with drug synergy")
    display_top_words(top_words)
    print()
    print(f"Words with least mutual information with drug synergy")
    display_top_words(bottom_words)