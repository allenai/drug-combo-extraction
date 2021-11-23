'''
Usage
python pretrain.py --pretrained-lm allenai/scibert_scivocab_uncased \
--training-file data/pretraining_data_small/train_dev.jsonl \
--test-file data/pretraining_data_small/test.jsonl \
--relation-counts data/pretraining_data_small/relation_counts.json \
--minimum-relation-frequency 5 \
--batch-size 18 \
--num-train-epochs 10 --lr 1e-3 \
--context-window-size 300 \
--max-seq-length 512 \
--unfreezing-strategy all \
--model-name pretraining_small
'''

import argparse
from collections import defaultdict
import json
import jsonlines
import os
import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import sys

sys.path.extend(['.', '..'])
from common.constants import ENTITY_END_MARKER, ENTITY_START_MARKER, LOW_FREQ_RELATION_IDX
from common.utils import construct_row_id_idx_mapping, ModelMetadata, save_metadata, set_seed, write_error_analysis_file
from modeling.pretraining_model import PretrainForRelation, Pretrainer
from preprocessing.data_loader import PretrainingDataModule
from preprocessing.preprocess import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-lm', type=str, required=False, default="allenai/scibert_scivocab_uncased", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--training-file', type=str, required=False, default="/Users/vijay/Downloads/distant_supervision_pretraining_data/train_dev.jsonl")
parser.add_argument('--test-file', type=str, required=False, default="/Users/vijay/Downloads/distant_supervision_pretraining_data/test.jsonl")
parser.add_argument('--batch-size', type=int, required=False, default=12) # This number is good for training on an 11GB Tesla K80 GPU.
parser.add_argument('--dev-train-split', type=float, required=False, default=0.1, help="Fraction of the training set to hold out for validation")
parser.add_argument('--max-seq-length', type=int, required=False, default=512, help="Maximum subword length of the document passed to the encoder, including inserted marker tokens")
parser.add_argument('--preserve-case', action='store_true')
parser.add_argument('--num-train-epochs', default=6, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--ignore-paragraph-context', action='store_true', help="If true, only look at each entity-bearing sentence and ignore its surrounding context.")
parser.add_argument('--minimum-relation-frequency', type=int, default=1, help="Only train on documents with relations up to a certain frequency")
parser.add_argument('--relation-counts', type=str, required=True, help="File generated from pretraining data preprocessing that describes the number of times each relation was observed in the pretraining data.")
parser.add_argument('--key-relations-file', type=str, required=True, help="File generated from pretraining data preprocessing that describes the key relations (contained in our supervised data) to represent.")
parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('--unfreezing-strategy', type=str, choices=["all", "final-bert-layer", "BitFit"], default="all", help="Whether to finetune all bert layers, just the final layer, or bias terms only.")
parser.add_argument('--context-window-size', type=int, required=False, default=None, help="Amount of cross-sentence context to use (including the sentence in question")
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--seed', type=int, required=False, default=2021)

if __name__ == "__main__":
    args = parser.parse_args()

    set_seed(args.seed)

    training_data_raw = list(jsonlines.open(args.training_file))
    test_data_raw = list(jsonlines.open(args.test_file))

    include_paragraph_context = not args.ignore_paragraph_context

    relation2idx = defaultdict(lambda: LOW_FREQ_RELATION_IDX)
    relation_counts = json.load(open(args.relation_counts))
    for rel_json, freq in relation_counts.items():
        rel = tuple(sorted(json.loads(rel_json)))
        if freq >= args.minimum_relation_frequency:
            relation2idx[rel] = len(relation2idx)

    new_entities_added = 0
    for i, rel_json in enumerate(json.load(open(args.key_relations_file))):
        rel = tuple(sorted(json.loads(rel_json)))
        for entity in rel:
            entity_tuple = (entity,)
            if entity_tuple not in relation2idx:
                new_entities_added += 1
                relation2idx[entity_tuple] = len(relation2idx)

    print(f"Number of relations in embedding matrix: {len(relation2idx)}")
    print(f"{new_entities_added} entities added from key entities list")

    training_data = create_dataset(training_data_raw,
                                   label2idx=relation2idx,
                                   add_no_combination_relations=False,
                                   include_paragraph_context=include_paragraph_context,
                                   context_window_size=args.context_window_size)
    test_data = create_dataset(test_data_raw,
                                   label2idx=relation2idx,
                                   add_no_combination_relations=False,
                                   include_paragraph_context=include_paragraph_context,
                                   context_window_size=args.context_window_size)

    # Remove documents with low-frequency relations.
    training_data = [doc for doc in training_data if doc["target"] != LOW_FREQ_RELATION_IDX]
    test_data = [doc for doc in test_data if doc["target"] != LOW_FREQ_RELATION_IDX]

    # Remove training examples with frequency below args.minimum_relation_frequency
    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(training_data + test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm, do_lower_case=not args.preserve_case)
    tokenizer.add_tokens([ENTITY_START_MARKER, ENTITY_END_MARKER])
    dm = PretrainingDataModule(training_data,
                               test_data,
                               tokenizer,
                               row_id_idx_mapping,
                               train_batch_size=args.batch_size,
                               dev_batch_size=args.batch_size,
                               test_batch_size=args.batch_size,
                               dev_train_ratio=args.dev_train_split,
                               max_seq_length=args.max_seq_length)
    dm.setup()

    model = PretrainForRelation.from_pretrained(
            args.pretrained_lm,
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            relation2idx=relation2idx,
            max_seq_length=args.max_seq_length,
            unfreeze_all_bert_layers=args.unfreezing_strategy=="all",
            unfreeze_final_bert_layer=args.unfreezing_strategy=="final-bert-layer",
            unfreeze_bias_terms_only=args.unfreezing_strategy=="BitFit")

    # Add rows to embedding matrix if not large enough to accomodate special tokens.
    if len(tokenizer) > len(model.bert.embeddings.word_embeddings.weight):
        model.bert.resize_token_embeddings(len(tokenizer))

    num_train_optimization_steps = len(dm.train_dataloader()) * float(args.num_train_epochs)

    system = Pretrainer(model, num_train_optimization_steps, lr=args.lr, tokenizer=tokenizer)
    trainer = pl.Trainer(
        num_nodes=1,
        gpus=4,
        precision=16,
        max_epochs=args.num_train_epochs,
        accelerator="dp"
    )
    trainer.fit(system, datamodule=dm)
    os.makedirs("pretraining_models", exist_ok=True)
    model_dir = "pretraining_models/checkpoints_" + args.model_name
    model.save_pretrained(model_dir)
    trainer.save_checkpoint(os.path.join(model_dir, "model.chkpt"))
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))

    relation2idx_hashable = {}
    for relation, idx in relation2idx.items():
        relation2idx_hashable[json.dumps(relation)] = idx

    metadata = ModelMetadata(args.pretrained_lm,
                             args.max_seq_length,
                             len(relation2idx_hashable),
                             relation2idx_hashable,
                             False,
                             False,
                             include_paragraph_context,
                             args.context_window_size)
    save_metadata(metadata, model_dir)
    trainer.test(system, datamodule=dm)
    test_predictions = system.test_predictions
    test_row_ids = [idx_row_id_mapping[row_idx] for row_idx in system.test_row_idxs]
    os.makedirs("outputs", exist_ok=True)
    ground_truths = [data['target'] for data in test_data]
    json.dump(test_predictions, open(os.path.join("outputs", "test_predictions.json"), 'w'))
    json.dump(ground_truths, open(os.path.join("outputs", "ground_truths.json"), 'w'))
    write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, os.path.join("outputs", args.model_name + ".tsv"))
