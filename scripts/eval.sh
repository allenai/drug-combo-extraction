#!/bin/bash

python produce_gold_jsonl.py "$1" temp_out.jsonl
python ../leaderboard/eval.py --gold-file temp_out.jsonl --pred-file "$2"
rm temp_out.jsonl
