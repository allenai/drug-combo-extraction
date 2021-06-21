import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
                            AdamW,
                            BertModel,
                            BertPreTrainedModel,
                            get_linear_schedule_with_warmup
)

from constants import ENTITY_PAD_IDX
from utils import accuracy, f1

BertLayerNorm = torch.nn.LayerNorm

class ModelOutput:
    def __init__(self, logits, loss):
        self.logits = logits
        self.loss = loss

# Adapted from https://github.com/princeton-nlp/PURE
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_rel_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_rel_labels)
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, all_entity_idxs=None, input_position=None):
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
            entity_idxs = entity_idxs[torch.where(entity_idxs != ENTITY_PAD_IDX)]
            entity_vectors.append(torch.mean(a[entity_idxs], dim=0).unsqueeze(0))
        mean_entity_embs = torch.cat(entity_vectors, dim=0)
        rep = self.layer_norm(mean_entity_embs)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_rel_labels), labels.view(-1))
        else:
            loss = None
        return ModelOutput(logits, loss)

class RelationExtractor(pl.LightningModule):

    def __init__(self,
                 model: BertForRelation,
                 num_train_optimization_steps,
                 lr=1e-3,
                 correct_bias=True,
                 warmup_proportion=0.1,
    ):
        # TODO(Vijay): configure these parameters via command line arguments.
        super().__init__()
        self.model = model
        self.num_train_optimization_steps = num_train_optimization_steps
        self.lr = lr
        self.correct_bias = correct_bias
        self.warmup_proportion = warmup_proportion

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        optimization_steps = int(self.num_train_optimization_steps * self.warmup_proportion)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    optimization_steps,
                                                    self.num_train_optimization_steps)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs = inputs
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        self.log("loss", output.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return output.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs = inputs
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        self.log("val_loss", output.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return output.loss

    def test_step(self, inputs, batch_idx):
        input_ids, token_type_ids, attention_mask, labels, all_entity_idxs = inputs
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels, all_entity_idxs=all_entity_idxs)
        self.log("test_loss", output.loss, prog_bar=True, logger=True)
        logits = output.logits
        predictions = torch.argmax(logits, dim=1)
        accuracy = accuracy(predictions, labels)
        f1, prec, rec = f1(predictions, labels)
        self.log("accuracy", accuracy, prog_bar=True, logger=True)
        self.log("precision", prec, prog_bar=True, logger=True)
        self.log("recall", rec, prog_bar=True, logger=True)
        self.log("f1", f1, prog_bar=True, logger=True)
        return accuracy
