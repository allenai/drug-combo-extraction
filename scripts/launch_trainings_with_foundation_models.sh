mkdir ~/logs/
HOME_DIR=/home/vijay
declare -a arr=("${HOME_DIR}/continued_pretraining_directory_large_scale_10_epochs/" "${HOME_DIR}/continued_pretraining_directory_bluebert_10_epochs/" "${HOME_DIR}/continued_pretraining_directory_biobert_10_epochs/" "${HOME_DIR}/continued_pretraining_directory_pubmedbert_10_epochs/" "allenai/scibert" "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12" "dmis-lab/biobert-v1.1" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

declare -a names=("scibert_cpt" "bluebert_cpt" "biobert_cpt" "pubmedbert_cpt" "scibert" "bluebert" "biobert" "pubmedbert")

for i in {0..7};
do
	for seed in {2021..2025};
	do
		echo "${arr[i]}"
		echo "${names[i]}"
		time python scripts/train.py --pretrained-lm "${arr[i]}" --num-train-epochs 10 --lr 2e-4 --batch-size 18 --training-file data/final_train_set.jsonl --test-file data/final_test_set.jsonl --context-window-size 400 --max-seq-length 512 --label2idx data/label2idx_multiclass.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}  |& tee ~/logs/${names[i]}_${seed}_multiclass.log
		# (cd .. && time python scripts/train.py --pretrained-lm "${arr[i]}" --num-train-epochs 10 --lr 2e-4 --batch-size 18 --training-file data/final_train_set.jsonl --test-file data/final_test_set.jsonl --context-window-size 400 --max-seq-length 512 --label2idx data/label2idx_multiclass.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}  |& tee ~/logs/${names[i]}_${seed}_multiclass.log)
	done
done
