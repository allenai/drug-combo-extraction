mkdir ~/logs/
HOME_DIR=/home/vijay
declare -a arr=("${HOME_DIR}/continued_pretraining_directory_large_scale_10_epochs/" "${HOME_DIR}/continued_pretraining_directory_bluebert_10_epochs/")

declare -a names=("scibert_cpt" "bluebert_cpt")

for seed in {2021..2024};
do
	for i in {0..1};
	do
		echo loop $i $seed
		echo "${arr[i]}"
		echo "${names[i]}"
		#if [ $i -ge 0 ] && [ $seed -ge 2021 ]; then
		time python scripts/train.py --pretrained-lm "${arr[i]}" --num-train-epochs 10 --lr 2e-4 --batch-size 18 --training-file data/final_train_set.jsonl --test-file data/final_test_set.jsonl --context-window-size 400 --max-seq-length 512 --label2idx data/label2idx_three_class.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}_three_class  |& tee ~/logs/${names[i]}_${seed}_three_class.log
		#fi
		# (cd .. && time python scripts/train.py --pretrained-lm "${arr[i]}" --num-train-epochs 10 --lr 2e-4 --batch-size 18 --training-file data/final_train_set.jsonl --test-file data/final_test_set.jsonl --context-window-size 400 --max-seq-length 512 --label2idx data/label2idx_multiclass.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}  |& tee ~/logs/${names[i]}_${seed}_multiclass.log)
		#if [ $i == 1 ]; then
		python scripts/test_only.py --checkpoint-path ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class --test-file ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl --batch-size 100 --outputs-directory ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class/outputs/ --seed $seed --produce_all_subsets
		(cd scripts && ./eval_three_class.sh ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class/outputs/predictions.jsonl > ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class/outputs/eval_partial.txt)
		(cd scripts && ./eval_three_class.sh ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class/outputs/predictions.jsonl --exact-match > ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}_three_class/outputs/eval_exact.txt)
		#fi
	done
done
