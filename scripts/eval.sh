#!/bin/bash

python produce_gold_jsonl.py "$1" temp_out.jsonl
if [ $# -eq 3 ] ; then
  python eval.py --gold-file temp_out.jsonl --pred-file "$2" --exact-match ;
else
  python eval.py --gold-file temp_out.jsonl --pred-file "$2"
fi
rm temp_out.jsonl
