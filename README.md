# Drug Combination Extraction

### Dependencies
Create a [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) environment:
```
conda create --name drug_combo python=3.8.5
conda activate drug_combo
```

On this virtual environment, install all required dependencies via pip:
```
pip install -r requirements.txt
```
### Data
Consumes drug synergy dataset, in jsonlines format. The dataset is found in `data/final_train_set.jsonl` and `data/final_test_set.jsonl`.

### Training
We recommend training on a GPU machine. We trained our models on machines with a 15GB Nvidia Tesla T4 GPU running Ubuntu 18.04.

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
(cd scripts && ./eval.sh ../data/final_test_set.jsonl ../checkpoints_pubmedbert_2022/outputs/predictions.jsonl > ../checkpoints_pubmedbert_2022/outputs/eval_partial.txt)
(cd scripts && ./eval.sh ../data/final_test_set.jsonl ../checkpoints_pubmedbert_2022/outputs/predictions.jsonl --exact-match > ../checkpoints_pubmedbert_2022/outputs/eval_exact.txt)
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
            checkpoints_${MODEL_NAME}/outputs/predictions.jsonl)

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
To modify pretrained language models with domain-adaptive pretraining, we need to use a separate training script from the HugggingFace library. First, unzip the pretraining data we have prepared in our data directory: `continued_pretraining_large_lowercased_train.txt.tgz` and `continued_pretraining_large_lowercased_val.txt.tgz` (these text files are 166M and 42M unzipped).

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

### Requirements
[PyTorch](https://pytorch.org/get-started/locally/)

pytorch_lightning

jsonlines

streamlit

