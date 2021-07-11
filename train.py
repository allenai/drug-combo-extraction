# Usage
# python train.py --balance-training-batch-labels

import argparse
import json
import jsonlines
import pytorch_lightning as pl
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from constants import ENTITY_END_MARKER, ENTITY_START_MARKER, NOT_COMB
from data_loader import DrugSynergyDataModule
from model import BertForRelation, RelationExtractor
from preprocess import create_dataset
from utils import construct_row_id_idx_mapping, write_error_analysis_file

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-lm', type=str, required=False, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--training-file', type=str, required=False, default="data/examples2_80.jsonl")
parser.add_argument('--test-file', type=str, required=False, default="data/examples2_20.jsonl")
parser.add_argument('--label2idx', type=str, required=False, default="data/label2idx.json")
parser.add_argument('--batch-size', type=int, required=False, default=12) # This number is good for training on an 11GB Tesla K80 GPU.
parser.add_argument('--dev-train-split', type=float, required=False, default=0.1, help="Fraction of the training set to hold out for validation")
parser.add_argument('--max-seq-length', type=int, required=False, default=512, help="Maximum subword length of the document passed to the encoder, including inserted marker tokens")
parser.add_argument('--preserve-case', action='store_true')
parser.add_argument('--num-train-epochs', default=6, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--label-sampling-ratios', default=None, type=float, help="Upsample or downsample training examples of each class for training (due to label imbalance)")
parser.add_argument('--label-loss-weights', default=None, type=float, help="Loss weight for negative class labels in training (to help with label imbalance)")
parser.add_argument('--ignore-no-comb-relations', action='store_true', help="If true, then don't mine NO_COMB negative relations from the relation annotations.")
parser.add_argument('--only-include-binary-no-comb-relations', action='store_true', help="If true, and we are including no-comb relations, then only mine binary no-comb relations (ignoring n-ary no-comb relations)")
parser.add_argument('--ignore-paragraph-context', action='store_true', help="If true, only look at each entity-bearing sentence and ignore its surrounding context.")
parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate")
parser.add_argument('--unfreezing-strategy', type=str, choices=["all", "final-bert-layer", "BitFit"], default="BitFit", help="Whether to finetune all bert layers, just the final layer, or bias terms only.")
parser.add_argument('--context-window-size', type=int, required=False, default=None, help="Amount of cross-sentence context to use (including the sentence in question")
parser.add_argument('--balance-training-batch-labels', action='store_true', help="If true, load training batches to ensure that each batch contains samples of each class.")
parser.add_argument('--output-file', type=str, required=False, default="test_output.tsv")

if __name__ == "__main__":
    args = parser.parse_args()

    training_data_raw = list(jsonlines.open(args.training_file))
    test_data_raw = list(jsonlines.open(args.test_file))
    label2idx = json.load(open(args.label2idx))
    label2idx[NOT_COMB] = 0

    if args.label_sampling_ratios is None:
        label_sampling_ratios = [1.0 for _ in label2idx]
    else:
        label_sampling_ratios = args.label_sampling_ratios

    if args.label_loss_weights is None:
        label_loss_weights = [1.0 for _ in label2idx]
    else:
        label_loss_weights = args.label_loss_weights

    training_data = create_dataset(training_data_raw,
                                   label2idx=label2idx,
                                   label_sampling_ratios=label_sampling_ratios,
                                   add_no_combination_relations=not args.ignore_no_comb_relations,
                                   only_include_binary_no_comb_relations=args.only_include_binary_no_comb_relations,
                                   include_paragraph_context=not args.ignore_paragraph_context,
                                   context_window_size=args.context_window_size)
    label_values = sorted(set(label2idx.values()))
    num_labels = len(label_values)
    assert label_values == list(range(num_labels))
    assert len(label_sampling_ratios) == num_labels
    assert len(label_loss_weights) == num_labels
    test_data = create_dataset(test_data_raw,
                               label2idx=label2idx,
                               add_no_combination_relations=not args.ignore_no_comb_relations,
                               only_include_binary_no_comb_relations=args.only_include_binary_no_comb_relations,
                               include_paragraph_context=not args.ignore_paragraph_context,
                               context_window_size=args.context_window_size)
    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(training_data + test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm, do_lower_case=not args.preserve_case)
    tokenizer.add_tokens([ENTITY_START_MARKER, ENTITY_END_MARKER])
    dm = DrugSynergyDataModule(training_data,
                               test_data,
                               tokenizer,
                               label2idx,
                               row_id_idx_mapping,
                               train_batch_size=args.batch_size,
                               dev_batch_size=args.batch_size,
                               test_batch_size=args.batch_size,
                               dev_train_ratio=args.dev_train_split,
                               max_seq_length=args.max_seq_length,
                               balance_training_batch_labels=args.balance_training_batch_labels)
    dm.setup()

    model = BertForRelation.from_pretrained(
            args.pretrained_lm,
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            num_rel_labels=num_labels,
            unfreeze_all_bert_layers=args.unfreezing_strategy=="all",
            unfreeze_final_bert_layer=args.unfreezing_strategy=="final-bert-layer",
            unfreeze_bias_terms_only=args.unfreezing_strategy=="BitFit")

    # Add rows to embedding matrix if not large enough to accomodate special tokens.
    if len(tokenizer) > len(model.bert.embeddings.word_embeddings.weight):
        model.bert.resize_token_embeddings(len(tokenizer))

    num_train_optimization_steps = len(dm.train_dataloader()) * float(args.num_train_epochs)

    if set(label_loss_weights) != {1.0}:
        # Unless all labels are being weighted equally, then compute specific label weights for class-weighted loss.
        label_loss_weighting = [w / sum(label_loss_weights) for w in label_loss_weights]
    else:
        label_loss_weighting = None

    system = RelationExtractor(model, num_train_optimization_steps, lr=args.lr, tokenizer=tokenizer, label_weights=label_loss_weighting)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=args.num_train_epochs,
    )
    trainer.fit(system, datamodule=dm)
    trainer.test(system, datamodule=dm)
    test_predictions = system.test_predictions
    test_row_ids = [idx_row_id_mapping[row_idx] for row_idx in system.test_row_idxs]
    write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, args.output_file)
    print("Done!")