mkdir ~/logs/
HOME_DIR=/home/vijay
declare -a arr=("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" "${HOME_DIR}/continued_pretraining_directory_pubmedbert_10_epochs/")

declare -a names=("ddi_pubmedbert_no_dapt" "ddi_pubmedbert")

for i in {1..1};
do
	for seed in {2021..2024};
	do
		echo "${arr[i]}"
		echo "${names[i]}"
		# time python scripts/train.py --pretrained-lm microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --num-train-epochs 10 --lr 2e-4 --batch-size 28 --training-file /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/train.jsonl --test-file /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/test.jsonl --ignore-no-comb-relations --context-window-size 256 --max-seq-length 400 --label2idx /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/label2idx.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}  |& tee ~/logs/${names[i]}_${seed}.log
		# python scripts/test_only.py --checkpoint-path /home/vijay/drug-synergy-models/checkpoints_ddi_pubmedbert_no_dapt_${seed} --test-file /home/vijay/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/test.jsonl  --batch-size 28 --outputs-directory ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/ --seed $seed
		./scripts/eval.sh ${HOME_DIR}/drug-synergy-models/dataset_conversion/DDICorpus/synergy_format/test.jsonl  ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/predictions.jsonl --micro-f1 # > ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/eval_micro_f1.txt
	done
done
