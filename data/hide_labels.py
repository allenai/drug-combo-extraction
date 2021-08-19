'''
Usage:
python hide_labels.py --input-file test_set.jsonl  --output-file test_set_labels_hidden.jsonl 
'''

import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-file",
    type=str,
    help="JSONL dataset file to remove labels from",
)
parser.add_argument(
    "--output-file",
    type=str,
    help="JSONL dataset file to remove labels from",
)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = list(jsonlines.open(args.input_file))
    num_rows = 0
    with jsonlines.Writer(open(args.output_file, 'wb')) as writer:
        for doc in dataset:
            del doc["rels"]
            writer.write(doc)
            num_rows += 1
    print(f"Wrote {num_rows} rows to {args.output_file}.")