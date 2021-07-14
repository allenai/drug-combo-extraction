import os
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
                            AutoTokenizer,
                            BertModel,
                            BertPreTrainedModel,
                            PretrainedConfig
)
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from typing import Callable, Dict, List, Optional, Tuple

from constants import ENTITY_PAD_IDX
from optimizers import adamw_with_linear_warmup, simple_adamw
from utils import accuracy, compute_f1, load_metadata

BertLayerNorm = torch.nn.LayerNorm

class ModelOutput:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss

# Adapted from https://github.com/princeton-nlp/PURE
class BertForRelation(BertPreTrainedModel):
    def __init__(self,
                 config: PretrainedConfig,
                 num_rel_labels: int,
                 max_seq_length: int,
                 unfreeze_all_bert_layers: bool = False,
                 unfreeze_final_bert_layer: bool = False,
                 unfreeze_bias_terms_only: bool = True):
        """Initialize simple BERT-based relation extraction model

        Args:
            config: Pretrained model config (loaded from model)
            num_rel_labels: Size of label set that each relation could take
            unfreeze_all_bert_layers: Finetune all layers of BERT
            unfreeze_final_bert_layer: Finetune only the final encoder layer of BERT
            unfreeze_bias_terms_only: Finetune only the bias terms in BERT (aka BitFit)
        """
        super(BertForRelation, self).__init__(config)
        self.num_rel_labels = num_rel_labels
        self.max_seq_length = max_seq_length
        self.bert = BertModel(config)
        for name, param in self.bert.named_parameters():
            if unfreeze_final_bert_layer:
                if "encoder.layer.11" not in name:
                    param.requires_grad = False
            elif unfreeze_bias_terms_only:
                if "bias" not in name:
                    param.requires_grad = False
            elif not unfreeze_all_bert_layers:
                param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_rel_labels)
        self.init_weights()


    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                all_entity_idxs: Optional[torch.Tensor] = None,
                input_position: Optional[torch.Tensor] = None) -> ModelOutput:
        """BertForRelation model, forward pass.

        Args:
            input_ids: Subword indices in the vocabulary for words in the document
            token_type_ids: Sequence segment IDs (currently set to all 0's) - TODO(Vijay): update this
            attention_mask: Mask which describes which tokens should be ignored (i.e. padding tokens)
            labels: Tensor of numerical labels
            all_entity_idxs: Tensor of indices of each drug entity's special start token in each document
            input_position: Just here to satisfy the interface (TODO(Vijay): remove this if possible)
        """
        # TODO(Vijay): delete input_positions, since it's seemingly not used
        # TODO(Vijay): analyze the output with the `output_attentions` flag, to help interpret the model's predictions.
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            output_attentions=False,
                            position_ids=input_position)
        sequence_output = outputs[0]

        entity_vectors = []
        for a, entity_idxs in zip(sequence_output, all_entity_idxs):
            # We store the entity-of-interest indices as a fixed-dimension matrix with padding indices.
            # Ignore padding indices when computing the average entity representation.
            assert torch.max(entity_idxs).item() < self.max_seq_length, "Entity is out of bounds in truncated text seqence, make --max-seq-length larger"
            entity_idxs = entity_idxs[torch.where(entity_idxs != ENTITY_PAD_IDX)]
            entity_vectors.append(torch.mean(a[entity_idxs], dim=0).unsqueeze(0))
        mean_entity_embs = torch.cat(entity_vectors, dim=0)
        rep = self.layer_norm(mean_entity_embs)
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        return logits

    def make_predictions(self, inputs):
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs, _ = inputs
        logits = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        predictions = torch.argmax(logits, dim=1)
        return predictions

class RelationExtractor(pl.LightningModule):
    def __init__(self,
                 model: BertForRelation,
                 num_train_optimization_steps: int,
                 tokenizer: AutoTokenizer,
                 lr: float = 5e-4,
                 correct_bias: bool = True,
                 warmup_proportion: float = 0.1,
                 optimizer_strategy: Callable = simple_adamw,
                 label_weights: Optional[List] = None,
    ):
        """PyTorch Lightning module which wraps the BERT-based model.

        Args:
            model: Base BERT model for relation classification
            num_train_optimization_steps: Number of optimization steps in training
            lr: Learning rate
            correct_bias: Whether to correct bias in AdamW
            warmup_proportion: How much data to reserve for linear learning rate warmup (https://paperswithcode.com/method/linear-warmup)
            optimizer_strategy: Constructor to create an optimizer to use (e.g. AdamW with a linear warmup schedule)
        """
        # TODO(Vijay): configure these parameters via command line arguments.
        super().__init__()
        self.model = model
        self.num_train_optimization_steps = num_train_optimization_steps
        self.lr = lr
        self.correct_bias = correct_bias
        self.warmup_proportion = warmup_proportion
        self.tokenizer = tokenizer
        self.test_sentences = []
        self.test_row_idxs = []
        self.test_predictions = []
        self.test_batch_idxs = []
        self.optimizer_strategy = optimizer_strategy
        if label_weights is not None:
            # This defines a class variable, but automatically moves the tensor to the
            # device that the module trains on.
            self.register_buffer("label_weights", torch.tensor(label_weights))
        else:
            self.label_weights = None

    def configure_optimizers(self):
        return self.optimizer_strategy(self.named_parameters(), self.lr, self.correct_bias, self.num_train_optimization_steps, self.warmup_proportion)

    def forward(self, inputs, pass_text = True):
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs, _ = inputs
        logits = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        return logits

    def training_step(self, inputs, batch_idx):
        """Training step in PyTorch Lightning.

        Args:
            inputs: Batch of data points
            batch_idx: Not used in this model (just here to satisfy the interface)

        Return:
            Loss tensor
        """
        # outputs: TokenClassifierOutput
        _, _, _, labels, _, _ = inputs
        logits = self(inputs, pass_text = True)
        loss = F.cross_entropy(logits.view(-1, self.model.num_rel_labels), labels.view(-1), weight=self.label_weights)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        predictions = torch.argmax(logits, dim=1)
        acc = accuracy(predictions, labels)
        metrics_dict = compute_f1(predictions, labels)
        f, prec, rec = metrics_dict["f1"], metrics_dict["precision"], metrics_dict["recall"]
        self.log("accuracy", acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("precision", prec, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("recall", rec, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("f1", f, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, inputs, batch_idx):
        """Validation step in PyTorch Lightning.

        Args:
            inputs: Batch of data points
            batch_idx: Not used in this model (just here to satisfy the interface)

        Return:
            Loss tensor
        """
        # outputs: TokenClassifierOutput
        _, _, _, labels, _, _ = inputs
        logits = self(inputs, pass_text = True)
        loss = F.cross_entropy(logits.view(-1, self.model.num_rel_labels), labels.view(-1), weight=self.label_weights)

        self.log("val_loss", loss, prog_bar=False, logger=True, on_step=False, on_epoch=False)
        return loss

    def test_step(self, inputs, batch_idx):
        """Testing step in PyTorch Lightning.

        Args:
            inputs: Batch of data points
            batch_idx: Not used in this model (just here to satisfy the interface)

        Return:
            Accuracy value (float) on the test set
        """
        input_ids, _, _, labels, _, row_ids = inputs
        logits = self(inputs, pass_text = True)
        raw_text = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        loss = F.cross_entropy(logits.view(-1, self.model.num_rel_labels), labels.view(-1), weight=self.label_weights)

        self.log("test_loss", loss, prog_bar=True, logger=True)
        predictions = torch.argmax(logits, dim=1)
        self.test_sentences.extend(raw_text)
        self.test_row_idxs.extend(row_ids.tolist())
        self.test_predictions.extend(predictions.tolist())
        self.test_batch_idxs.extend([batch_idx for _ in predictions.tolist()])
        acc = accuracy(predictions, labels)
        metrics_dict = compute_f1(predictions, labels)
        f, prec, rec = metrics_dict["f1"], metrics_dict["precision"], metrics_dict["recall"]
        self.log("accuracy", acc, prog_bar=True, logger=True)
        self.log("precision", prec, prog_bar=True, logger=True)
        self.log("recall", rec, prog_bar=True, logger=True)
        self.log("f1", f, prog_bar=True, logger=True)

def load_model(checkpoint_directory: str) -> Tuple[BertForRelation, AutoTokenizer, int, Dict, bool]:
    '''Given a directory containing a model checkpoint, return the model, and other metadata regarding the data
    preprocessing that the model expects.

    Args:
        checkpoint_directory: Path to local directory where model is serialized

    Returns:
        model: Pretrained BertForRelation model
        tokenizer: Hugging Face tokenizer loaded from disk
        max_seq_length: Maximum number of subwords in a document allowed by the model (if longer, truncate input)
        label2idx: Mapping from label strings to numerical label indices
        include_paragraph_context: Whether or not to include paragraph context in addition to the relation-bearing sentence
    '''
    model_name, max_seq_length, num_labels, label2idx, include_paragraph_context = load_metadata(checkpoint_directory)
    model = BertForRelation.from_pretrained(
                checkpoint_directory,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                num_rel_labels=num_labels,
                max_seq_length=max_seq_length
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer.from_pretrained(os.path.join(checkpoint_directory, "tokenizer"))
    return model, tokenizer, max_seq_length, label2idx, include_paragraph_context