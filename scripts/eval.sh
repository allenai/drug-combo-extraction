#!/bin/bash

python scripts/produce_gold_jsonl.py "$1" temp_out.jsonl
if [ $# -eq 3  ] && [ $3 = "--exact-match" ] ; then
  echo "yes"
  python scripts/eval.py --gold-file temp_out.jsonl --pred-file "$2" --exact-match ;
elif [ $# -eq 3  ] && [ $3 = "--micro-f1" ] ; then
  python scripts/eval_micro_f1.py --gold-file temp_out.jsonl --pred-file "$2"
else
  python scripts/eval.py --gold-file temp_out.jsonl --pred-file "$2"
fi
rm temp_out.jsonl
