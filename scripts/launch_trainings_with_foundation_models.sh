mkdir ~/logs/
HOME_DIR=/home/vijay
declare -a arr=(50 100 200 300)

declare -a names=("50" "100" "200" "300")

for i in {0..3};
do
	for seed in {2021..2024};
	do
		echo "${names[i]}"
		time python scripts/train.py --pretrained-lm "${HOME_DIR}/continued_pretraining_directory_pubmedbert_10_epochs/" --num-train-epochs 10 --lr 2e-4 --batch-size 18 --training-file data/final_train_set.jsonl --test-file data/final_test_set.jsonl --context-window-size 400 --max-seq-length 512 --label2idx data/label2idx.json --seed $seed --unfreezing-strategy final-bert-layer --model-name ${names[i]}_${seed}  |& tee ~/logs/${names[i]}_${seed}.log
		python scripts/test_only.py --checkpoint-path ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed} --test-file ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl --batch-size 100 --outputs-directory ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/ --seed $seed
		(cd scripts && ./eval.sh ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/predictions.jsonl > ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/eval_partial.txt)
		(cd scripts && ./eval.sh ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/predictions.jsonl --exact-match > ${HOME_DIR}/drug-synergy-models/checkpoints_${names[i]}_${seed}/outputs/eval_exact.txt)
	done
done
