'''
python prepare_continued_pretraining_data.py \
    --pretraining-input-file x \
    --pretraining-output-file y
'''
import argparse
import csv
import hashlib
import json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pretraining-input-file', type=str, help="Path to pretraining data CSV")
parser.add_argument('--pretraining-output-file', type=str, help="Path to pretraining data txt file (text only)")


if __name__ == "__main__":
    args = parser.parse_args()
    sources = set()
    sentences = set()
    paragraphs = set()
    for row in tqdm(csv.DictReader(open(args.pretraining_data_file))):
        if row['article_link'] in sources or row["sentence_text"] in sentences or row["paragraph_text"] in paragraphs:
            # Process each document once
            continue
        sources.add(row['article_link'])
        sentences.add(row['sentence_text'])
        paragraphs.add(row["paragraph_text"])

    with open(args.pretraining_output_file, 'w') as outfile:
        outfile.write("\n".join(paragraphs))

    print(f"Wrote raw text to {args.pretraining_output_file}")

