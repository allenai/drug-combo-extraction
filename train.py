import argparse
import json
import jsonlines
import pytorch_lightning as pl
import tempfile
import shutil
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from constants import ENTITY_END_MARKER, ENTITY_START_MARKER
from data_loader import DrugSynergyDataModule
from model import BertForRelation, RelationExtractor
from preprocess import create_dataset, LABEL2IDX

dirpath = tempfile.mkdtemp()

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=str, required=False, default=dirpath)
parser.add_argument('--pretrained-lm', type=str, required=False, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--training-file', type=str, required=False, default="data/examples2_80.jsonl")
parser.add_argument('--test-file', type=str, required=False, default="data/examples2_20.jsonl")
parser.add_argument('--batch-size', type=int, required=False, default=12) # This number is good for training on an 11GB Tesla K80 GPU.
parser.add_argument('--dev-train-split', type=float, required=False, default=0.1, help="Fraction of the training set to hold out for validation")
parser.add_argument('--max-seq-length', type=int, required=False, default=512, help="Maximum subword length of the document passed to the encoder, including inserted marker tokens")
parser.add_argument('--preserve-case', action='store_true')
parser.add_argument('--num-train-epochs', default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument('--negative-sampling-rate', default=1.0, type=float, help="Upsample or downsample negative training examples for training (due to label imbalance)")
parser.add_argument('--positive-sampling-rate', default=25.0, type=float, help="Upsample or downsample positive training examples for training (due to label imbalance)")

if __name__ == "__main__":
    args = parser.parse_args()

    training_data = list(jsonlines.open(args.training_file))
    test_data = list(jsonlines.open(args.test_file))
    training_data = create_dataset(training_data, sample_negatives_ratio=args.negative_sampling_rate, sample_positives_ratio=args.positive_sampling_rate)
    test_data = create_dataset(test_data)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_lm, do_lower_case=not args.preserve_case)
    tokenizer.add_tokens([ENTITY_START_MARKER, ENTITY_END_MARKER])
    dm = DrugSynergyDataModule(training_data,
                            test_data,
                            tokenizer,
                            LABEL2IDX,
                            train_batch_size=args.batch_size,
                            dev_batch_size=args.batch_size,
                            test_batch_size=args.batch_size,
                            dev_train_ratio=args.dev_train_split,
                            max_seq_length=args.max_seq_length)
    dm.setup()

    num_labels=len(set(dm.label_to_idx.values()))
    model = BertForRelation.from_pretrained(
            args.pretrained_lm, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)
    num_train_optimization_steps = len(dm.train_dataloader()) * float(args.num_train_epochs)
    system = RelationExtractor(model, num_train_optimization_steps)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=args.num_train_epochs,
    )
    trainer.fit(system, datamodule=dm)
    metrics = trainer.test(system, datamodule=dm)
    print(f"METRICS:\n{json.dumps(metrics, indent=4)}")
    test_predictions = system.test_predictions
