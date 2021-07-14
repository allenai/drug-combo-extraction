'''
Usage:
python add_document_ids_to_dataset.py train_set.jsonl
'''

import argparse
import jsonlines
import hashlib

from typing import Dict, Tuple

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_file",
    type=str,
    help="JSONL dataset file to create document IDs for",
)

def hash_string(string):
    return hashlib.md5((string).encode()).hexdigest()

def add_document_id_to_doc(doc: Dict, document_id_field: str = "doc_id") -> Tuple[Dict, bool]:
    if document_id_field in doc:
        return doc, False
    else:
        doc_id = hash_string(doc["sentence"])
        doc[document_id_field] = doc_id
        return doc, True

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = list(jsonlines.open(args.input_file))
    num_added_ids = 0
    with jsonlines.Writer(open(args.input_file, 'wb')) as writer:
        for doc in dataset:
            modified_dataset, id_added = add_document_id_to_doc(doc)
            writer.write(modified_dataset)
            num_added_ids += int(id_added)
    print(f"Added document IDs to {num_added_ids} documents.")