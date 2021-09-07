'''
python prepare_continued_pretraining_data.py \
    --lowercase-text \
    --pretraining-input-file /Users/vijay/Downloads/distant_supervision_large.csv \
    --pretraining-output-file /Users/vijay/Downloads/continued_pretraining_lowercased.txt

python prepare_continued_pretraining_data.py \
    --pretraining-input-file /Users/vijay/Downloads/distant_supervision_large.csv \
    --pretraining-output-file /Users/vijay/Downloads/continued_pretraining.txt
'''
import argparse
import csv
import hashlib
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pretraining-input-file', type=str, help="Path to pretraining data CSV")
parser.add_argument('--pretraining-output-file', type=str, help="Path to pretraining data txt file (text only)")
parser.add_argument('--lowercase-text', action="store_true", help="If true, write out lowercased text")

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

    with open(args.pretraining_output_file, 'w') as outfile:
        outfile.write("\n".join(paragraphs))

    print(f"Wrote raw text to {args.pretraining_output_file}")

