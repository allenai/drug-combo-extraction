# drug-synergy-models

### Data
Consumes drug synergy dataset, in jsonlines format.

### Usage
To run with default settings, you can simply run `python train.py`.

```
usage: train.py [--pretrained-lm PRETRAINED_LM (defaults to "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")] \
                [--training-file TRAINING_FILE  (defaults to data/examples2_80.jsonl)] \
                [--test-file TEST_FILE  (defaults to data/examples2_20.jsonl)] \
                [--batch-size BATCH_SIZE (defaults to 12)] \
                [--dev-train-split DEV_TRAIN_SPLIT (defaults to 0.1)] \
                [--max-seq-length MAX_SEQ_LENGTH (defaults to 512)]
                [--preserve-case (defaults to False)] \
                [--num-train-epochs NUM_TRAIN_EPOCHS (defaults to 3)]
```

To train 8 models with different foundation models (SciBERT, PubmedBert, etc), run:
```
./scripts/launch_trainings_with_foundation_models.sh
```

### Requirements
[PyTorch](https://pytorch.org/get-started/locally/)

pytorch_lightning

jsonlines


