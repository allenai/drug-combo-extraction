import os
import pytorch_lightning as pl
import time
import torch
from torch import nn
from torch.autograd import Variable
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
from model import ModelOutput
from optimizers import adamw_with_linear_warmup, simple_adamw
from utils import accuracy, compute_f1, load_metadata

BertLayerNorm = torch.nn.LayerNorm

def get_relations_with_arity(relations, arity):
    relations_with_arity = [r for r in relations if len(r) == arity]
    assert len(relations_with_arity) == len(set(relations_with_arity))
    assert len(relations_with_arity) > 0
    return relations_with_arity

# Adapted from https://github.com/princeton-nlp/PURE
class PretrainForRelation(BertPreTrainedModel):
    def __init__(self,
                 config: PretrainedConfig,
                 entity2idx: Dict,
                 relation2idx: Dict,
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
        super(PretrainForRelation, self).__init__(config)

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
        self.max_seq_length = max_seq_length

        self.entity2idx = entity2idx
        self.relation2idx = relation2idx
        self.idx2relation = {idx:rel for rel, idx in self.relation2idx.items()}
        self.register_parameter("entity_embeddings", nn.Parameter(torch.randn(len(self.entity2idx), config.hidden_size), requires_grad=True))
        self.register_parameter("relation_embeddings", nn.Parameter(torch.randn(len(self.relation2idx), config.hidden_size), requires_grad=True))
        # self.relation_embeddings = Variable(torch.randn(len(self.relation2idx), config.hidden_size, device='cuda'), requires_grad=True)

        ''' 
        self.embeddings_by_arity = {}
        self.relation2idx_by_arity = {}
        self.idx2relation_by_arity = {}

        for arity in range(1, max_arity + 1):
            relations_with_arity = get_relations_with_arity(relations, arity)
            self.embeddings_by_arity[arity] = torch.nn.Embedding(len(relations_with_arity), config.hidden_size)
            self.relation2idx_by_arity[arity] = {}
            self.idx2relation_by_arity[arity] = {}
            for i, rel in relations_with_arity:
                self.relation2idx_by_arity[arity][rel] = i
                self.idx2relation_by_arity[arity][i] = rel
        '''
        self.init_weights()


    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                all_entity_idxs: Optional[torch.Tensor] = None,
                input_position: Optional[torch.Tensor] = None) -> ModelOutput:
        """PretrainForRelation model, forward pass.

        Args:
            input_ids: Subword indices in the vocabulary for words in the document
            token_type_ids: Sequence segment IDs (currently set to all 0's) - TODO(Vijay): update this
            attention_mask: Mask which describes which tokens should be ignored (i.e. padding tokens)
            labels: Tensor of numerical labels
            all_entity_idxs: Tensor of indices of each drug entity's special start token in each document
            input_position: Just here to satisfy the interface (TODO(Vijay): remove this if possible)

            TODO(Vijay): pass in the true number of slots, so we can look up the appropriate embedding matrix
        """
        # TODO(Vijay): delete input_positions, since it's seemingly not used
        # TODO(Vijay): analyze the output with the `output_attentions` flag, to help interpret the model's predictions.
        forward_start = time.perf_counter()
        outputs = self.bert(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=False,
                            output_attentions=False,
                            position_ids=input_position)
        sequence_output = outputs[0]
        batch_size = len(sequence_output)

        sequence_output_transposed = sequence_output.transpose(1, 2)
        all_entity_idxs_transposed = all_entity_idxs.transpose(1, 2)
        # all_entity_idxs_transposed contains weighted values of each entity in the document, giving effectively
        # a weightec average of entity embeddings across the document.
        entity_vectors = torch.matmul(sequence_output_transposed, all_entity_idxs_transposed)
        entity_vectors = entity_vectors.squeeze(2) # Squeeze 768 x 1 vector into a single row of dimension 768

        text_rep = self.layer_norm(entity_vectors)
        text_rep_repeated = torch.unsqueeze(text_rep, dim=2).repeat(1, 1, len(self.relation_embeddings))
        relation_rep_repeated = torch.unsqueeze(self.relation_embeddings.T, dim=0).repeat(batch_size, 1, 1)
        consine_similarities = torch.nn.functional.cosine_similarity(text_rep_repeated, relation_rep_repeated)
        return consine_similarities, time.perf_counter() - forward_start

    def make_predictions(self, inputs):
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs, _ = inputs
        logits, forward_time = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        predictions = torch.argmax(logits, dim=1)
        return predictions, forward_time

class Pretrainer(pl.LightningModule):
    def __init__(self,
                 model: PretrainForRelation,
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
        logits, forward_time = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        return logits, forward_time

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
        logits, forward_time = self(inputs, pass_text = True)
        loss = F.cross_entropy(logits, labels.view(-1), weight=self.label_weights)

        self.log("forward_time", forward_time, prog_bar=False, logger=True, on_step=True, on_epoch=True)
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
        logits, forward_time = self(inputs, pass_text = True)
        loss = F.cross_entropy(logits, labels.view(-1), weight=self.label_weights)

        self.log("val_forward_time", forward_time, prog_bar=False, logger=True, on_step=False, on_epoch=False)
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
        logits, _ = self(inputs, pass_text = True)
        raw_text = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
        loss = F.cross_entropy(logits, labels.view(-1), weight=self.label_weights)

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

def load_model(checkpoint_directory: str) -> Tuple[PretrainForRelation, AutoTokenizer, int, Dict, bool]:
    '''Given a directory containing a model checkpoint, return the model, and other metadata regarding the data
    preprocessing that the model expects.

    Args:
        checkpoint_directory: Path to local directory where model is serialized

    Returns:
        model: Pretrained PretrainForRelation model
        tokenizer: Hugging Face tokenizer loaded from disk
        max_seq_length: Maximum number of subwords in a document allowed by the model (if longer, truncate input)
        label2idx: Mapping from label strings to numerical label indices
        include_paragraph_context: Whether or not to include paragraph context in addition to the relation-bearing sentence
    '''
    metadata = load_metadata(checkpoint_directory)
    model = PretrainForRelation.from_pretrained(
                checkpoint_directory,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                relation2idx=metadata.label2idx,
                max_seq_length=metadata.max_seq_length
    )
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name, do_lower_case=True)
    tokenizer.from_pretrained(os.path.join(checkpoint_directory, "tokenizer"))
    return model, tokenizer, metadata