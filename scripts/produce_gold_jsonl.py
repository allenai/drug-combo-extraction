import argparse
import json

from postprocessing import hash_string

parser = argparse.ArgumentParser()
parser.add_argument('ann_infile', type=str, help="Path to the gold file")
parser.add_argument('gold_outfile', type=str, help="Path to the predictions file")

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.ann_infile) as f:
        ls = f.readlines()
    with open(args.gold_outfile, "w") as f:
        for line in ls:
            doc = json.loads(line)
            for rel in doc["rels"]:
                d = {"doc_id": hash_string(doc["sentence"]), "drug_idxs": rel["spans"], "relation_label": rel["class"]}
                json.dump(d, f)
                f.write("\n")
