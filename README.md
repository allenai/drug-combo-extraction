# Drug Combination Extraction

Want to help researchers and clinicians plan treatments for acute medical conditions? Want to contribute to the health community by reducing drug research times? You came to the right place! This project created a dataset of drugs that go together well according to the biomedical literature (we've alse created appropriate solid baseline models). To participate, you will need to train a model on the data and when given a new sentence, predict which of the drugs in it combine together, and whether they combine in a positive/beneficial way.
To participate take a look at our [Leaderboard](https://leaderboard.allenai.org/drug_combo/submissions/public)

-----

## Dependencies
Create a [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) environment:
```
conda create --name drug_combo python=3.8.5
conda activate drug_combo

```
On this virtual environment, install all required dependencies via pip:
```
pip install -r requirements.txt
```
-----

## Dataset
Our dataset splits are `data/final_train_set.jsonl` and `data/final_test_set.jsonl`.

-----
## Models and Code
### Pretrained Baseline Model
You can find our strongest off-the-shelf model for this task on [Huggingface](https://huggingface.co/allenai/drug-combo-classifier-pubmedbert-dapt).

### Training
To reproduce or tweak the baseline model above, you can train your own with our provided scripts. We recommend training on a GPU machine. We trained our models on machines with a 15GB Nvidia Tesla T4 GPU running Ubuntu 18.04.

**Single command to train a relation extractor based on PubmedBERT:**
```
python scripts/train.py \
    --model-name pubmedbert_2021 \
    --pretrained-lm microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --num-train-epochs 10 \
    --lr 2e-4 \
    --batch-size 18 \
    --training-file data/final_train_set.jsonl \
    --test-file data/final_test_set.jsonl \
    --context-window-size 400 \
    --max-seq-length 512 \
    --label2idx data/label2idx.json \
    --seed 2022 \
    --unfreezing-strategy final-bert-layer
```

**Full training script options:**
```
python scripts/train.py \
                --model-name            MODEL_NAME
                [--pretrained-lm        PRETRAINED_LM (defaults to "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext") \
                [--num-train-epochs     NUM_TRAIN_EPOCHS (defaults to 10)] \
                [--lr                   LEARNING RATE (defaults to 2e-4)] \
                [--batch-size           BATCH SIZE (defaults to 18)] \
                [--training-file        data/final_train_set.jsonl] \
                [--test-file            data/final_test_set.jsonl] \
                [--context-window-size  CONTEXT LENGTH (defaults to 400)] \
                [--max-seq-length       512] \
                [--seed                 RANDOM SEED (defaults to 2021)] \
                [--unfreezing-strategy  UNFREEZING STRATEGY (defaults to final-bert-layer)]

```

### Testing and Evaluation

**Three commands to evaluate the relation extractor based on PubmedBERT:**
```
python scripts/test_only.py \
    --checkpoint-path checkpoints_pubmedbert_2022 \
    --test-file ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl \
    --batch-size 100 \
    --outputs-directory checkpoints_pubmedbert_2022/outputs/ \
    --seed 2022
(cd scripts && ./eval.sh ../data/final_test_set.jsonl ../checkpoints_pubmedbert_2022/outputs/predictions.jsonl > ../checkpoints_pubmedbert_2022/outputs/eval.txt)

```

**Full options for testing and evaluation scripts:**

```
Testing:
python scripts/test_only.py
            [--checkpoint-path      PATH TO CHECKPOINT CREATED IN TRAINING (checkpoints_${MODEL_NAME})] \
            [--test-file            data/final_test_set.jsonl] \
            [--batch-size           TEST BATCH SIZE (100)] \
            [--outputs-directory    OUTPUT DIRECTORY (checkpoints_${MODEL_NAME}/outputs/)] \
            [--seed                 RANDOM SEED (defaults to 2021)]

Evaluation (using exact-match or partial-match metrics):
(cd scripts && ./eval.sh \
            ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl \
            checkpoints_${MODEL_NAME}/outputs/predictions.jsonl) \
            ${OPTIONAL_OUTPUT_PATH}

```

### Analysis
We can now run analysis to capture model behavior along different aspects, using the output of several different models at different seeds
```
python scripts/produce_gold_jsonl.py ${HOME_DIR}/drug-synergy-models/data/final_test_set.jsonl ${HOME_DIR}/drug-synergy-models/data/final_test_rows.jsonl

python scripts/bucketing_analysis.py --pred-files \
    $MODEL_ONE_OUTPUT/outputs/predictions.jsonl \
    ... \
    $MODEL_N_OUTPUT/outputs/predictions.jsonl \
    --gold-file ${HOME_DIR}/drug-synergy-models/data/final_test_rows.jsonl \
    --bucket-type {arity OR relations_seen_in_training} \
    [--exact-match]
```


To train 8 models with different foundation models (SciBERT, PubmedBert, etc), run:
```
./scripts/launch_trainings_with_foundation_models.sh
```

### Domain-Adaptive Pretraining
You can find our strongest domain-adapted contextualized encoder (PubmedBERT

To perform domain-adaptive pretraining yourself, unzip the pretraining data we have prepared in our data directory: `continued_pretraining_large_lowercased_train.txt.tgz` and `continued_pretraining_large_lowercased_val.txt.tgz` (these text files are 166M and 42M unzipped).

Then do:
```
git clone https://github.com/huggingface/transformers.git
cd examples/pytorch/language-modeling/
```

Then
```
python run_mlm.py \
    --model_name_or_path microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --train_file $PATH_TO/continued_pretraining_large_lowercased_train.txt \
    --validation_file $PATH_TO/continued_pretraining_large_lowercased_val.txt \
    --do_train \
    --do_eval \
    --output_dir ~/continued_pretraining_directory_pubmedbert_10_epochs \
    --max_seq_length 512                                      \
    --overwrite_output_dir
```

------

## Cite Our Paper

If you use the data or models from this work in your own work, cite [A Dataset for N-ary Relation Extraction of Drug Combinations](https://arxiv.org/abs/2205.02289).

```bibtex
@inproceedings{Tiktinsky2022ADF,
    title = "A Dataset for N-ary Relation Extraction of Drug Combinations",
    author = "Tiktinsky, Aryeh and Viswanathan, Vijay and Niezni, Danna and Meron Azagury, Dana and Shamay, Yosi and Taub-Tabib, Hillel and Hope, Tom and Goldberg, Yoav",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.233",
    doi = "10.18653/v1/2022.naacl-main.233",
    pages = "3190--3203",
}
```
