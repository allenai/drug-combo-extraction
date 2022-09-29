#!/bin/bash

if [ $# -eq 3 ]; then
  output="$3"
else
  output="output/metrics.json"
fi
python produce_gold_jsonl.py "$1" temp_out.jsonl
python ../leaderboard/eval.py --gold-file temp_out.jsonl --pred-file "$2" --output "$output"
rm temp_out.jsonl
