'''
python prepare_continued_pretraining_data.py \
    --lowercase-text \
    --pretraining-input-file /Users/vijay/Downloads/distant_supervision_large.csv \
    --pretraining-output-prefix /Users/vijay/Downloads/continued_pretraining

python prepare_continued_pretraining_data.py \
    --pretraining-input-file /Users/vijay/Downloads/distant_supervision_large.csv \
    --pretraining-output-prefix /Users/vijay/Downloads/continued_pretraining
'''
import argparse
import csv
import hashlib
import json
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pretraining-input-file', type=str, help="Path to pretraining data CSV")
parser.add_argument('--pretraining-output-prefix', type=str, help="Path to pretraining data txt file (text only)")
parser.add_argument('--lowercase-text', action="store_true", help="If true, write out lowercased text")
parser.add_argument('--validation-ratio', type=float, default=0.2, help="Fraction of data to hold out for testing/validation")

if __name__ == "__main__":
    args = parser.parse_args()
    sources = set()
    sentences = set()
    paragraphs = set()
    for row in tqdm(csv.DictReader(open(args.pretraining_input_file))):
        if row['article_link'] in sources or row["sentence_text"] in sentences or row["paragraph_text"] in paragraphs:
            # Process each document once
            continue
        sources.add(row['article_link'])
        sentences.add(row['sentence_text'])
        paragraphs.add(row["paragraph_text"])

    if args.lowercase_text:
        paragraphs = [p.lower() for p in paragraphs]
        file_prefix = args.pretraining_output_prefix + "_lowercased"
    else:
        paragraphs = list(paragraphs)
        file_prefix = args.pretraining_output_prefix
    random.shuffle(paragraphs)

    train_paragraphs = paragraphs[:-int(args.validation_ratio * len(paragraphs))]
    validation_paragraphs = paragraphs[-int(args.validation_ratio * len(paragraphs)):]

    train_file = file_prefix + "_train.txt"
    validation_file = file_prefix + "_val.txt"
    with open(train_file, 'w') as train_writer:
        train_writer.write("\n".join(train_paragraphs))
    with open(validation_file, 'w') as val_writer:
        val_writer.write("\n".join(validation_paragraphs))

    print(f"Wrote raw text to\n{train_file}\nand\n{validation_file}.")

