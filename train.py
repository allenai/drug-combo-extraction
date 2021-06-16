import argparse
import jsonlines
import pytorch_lightning as pl

from data_loader import DrugSynergyDataModule
from model import BertForRelation, RelationExtractor
from preprocess import create_dataset

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained-lm', type=str, required=False, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", help="path to Huggingface Transformers model path")
parser.add_argument('--training-file', type=str, required=False, default="data/examples2.jsonl")
parser.add_argument('--test-file', type=str, required=False, default="data/examples.jsonl")

if __name__ == "__main__":
    args = parser.parse_args()

    training_data = list(jsonlines.open(args.training_file))
    test_data = list(jsonlines.open(args.test_file))
    training_data = create_dataset(training_data)
    test_data = create_dataset(test_data)

    data_module = DrugSynergyDataModule(training_data, test_data)
    data_module.setup()

    num_labels=None
    num_labels=4
    assert num_labels is not None, "Not Implemented"

    model = BertForRelation.from_pretrained(
            args.pretrained_lm, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_rel_labels=num_labels)

    marker=2
    breakpoint()
    num_train_optimization_steps = None
    assert num_train_optimization_steps is not None, "Not Implemented"
    system = RelationExtractor(model=model, )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        max_epochs=3,
    )

    trainer.fit(model, data_module)