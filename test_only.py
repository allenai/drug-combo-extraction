# Usage
# python test_only.py --checkpoint-path checkpoints/ --test-file data/dev_set_error_analysis.jsonl \
#                     --outputs-directory /tmp/outputs/ --error-analysis-file /tmp/error_analysis.csv

import argparse
import jsonlines
import os
import pytorch_lightning as pl

from data_loader import  DrugSynergyDataModule
from pretraining_model import load_model as load_pretraining_model
from model import RelationExtractor, load_model
from preprocess import create_dataset
from utils import construct_row_id_idx_mapping, set_seed, write_error_analysis_file, write_jsonl, adjust_data, filter_overloaded_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', type=str, required=False, default="checkpoints", help="Path to pretrained Huggingface Transformers model")
parser.add_argument('--self-supervised-lm', type=str, required=False, default=None, help="Path to self-supervised model path")
parser.add_argument('--test-file', type=str, required=False, default="data/dev_set_error_analysis.jsonl")
parser.add_argument('--batch-size', type=int, default=32, help="Batch size for testing (larger batch -> faster evaluation)")
parser.add_argument('--outputs-directory', type=str, required=False, help="Output directory where we write predictions, for offline evaluation", default="/tmp/outputs/.tsv")
parser.add_argument('--error-analysis-file', type=str, required=False, help="Output file containing error analysis information", default="test_output.tsv")
parser.add_argument('--seed', type=int, required=False, default=2021)
parser.add_argument('--produce_all_subsets', action='store_true', help="If true, and we are including no-comb relations, then include all subsets of existing relations as NO_COMB as well")

if __name__ == "__main__":
    args = parser.parse_args()
    set_seed(args.seed)
    model, tokenizer, metadata = load_model(args.checkpoint_path)

    if args.self_supervised_lm is not None:
        pretrained_model, pretrained_tokenizer, pretraining_metadata = load_pretraining_model(args.self_supervised_lm)
        model.relation2idx = pretrained_model.relation2idx
        model.entity2idx = pretrained_model.relation2idx
    else:
        model.relation2idx = metadata.pretraining_data["relation2idx"]

    test_data_raw = list(jsonlines.open(args.test_file))
    # TODO(Vijay): add `add_no_combination_relations`, `only_include_binary_no_comb_relations`, `include_paragraph_context`,
    # `context_window_size` to the model's metadata

    test_data = create_dataset(test_data_raw,
                               label2idx=metadata.label2idx,
                               add_no_combination_relations=metadata.add_no_combination_relations,
                               only_include_binary_no_comb_relations=metadata.only_include_binary_no_comb_relations,
                               include_paragraph_context=metadata.include_paragraph_context,
                               context_window_size=metadata.context_window_size,
                               produce_all_subsets=args.produce_all_subsets,
                               relation2idx=model.relation2idx,
                               entity2idx=model.entity2idx)

    row_id_idx_mapping, idx_row_id_mapping = construct_row_id_idx_mapping(test_data)
    dm = DrugSynergyDataModule(None,
                            test_data,
                            tokenizer,
                            metadata.label2idx,
                            row_id_idx_mapping,
                            train_batch_size=args.batch_size,
                            dev_batch_size=args.batch_size,
                            test_batch_size=args.batch_size,
                            max_seq_length=metadata.max_seq_length,
                            balance_training_batch_labels=False,
                            entity_embedding_size=len(model.entity_embeddings),
                            relation_embedding_size=len(model.relation_embeddings))
    dm.setup()

    system = RelationExtractor(model, 0, tokenizer=tokenizer)
    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        resume_from_checkpoint=os.path.join(args.checkpoint_path, "model.chkpt")
    )
    trainer.test(system, datamodule=dm)

    test_predictions = system.test_predictions
    test_row_ids = [idx_row_id_mapping[row_idx] for row_idx in system.test_row_idxs]

    fixed_test = filter_overloaded_predictions(adjust_data(test_row_ids, test_predictions))
    os.makedirs(args.outputs_directory, exist_ok=True)
    test_output = os.path.join(args.outputs_directory, "predictions.jsonl")

    write_jsonl(fixed_test, test_output)
    # write_error_analysis_file(test_data, test_data_raw, test_row_ids, test_predictions, args.error_analysis_file)