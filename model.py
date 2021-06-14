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

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm

# Adapted from https://github.com/princeton-nlp/PURE
class BertForRelation(BertPreTrainedModel):
    def __init__(self, config, num_rel_labels):
        super(BertForRelation, self).__init__(config)
        self.num_rel_labels = num_rel_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(config.hidden_size * 2)
        self.classifier = nn.Linear(config.hidden_size * 2, self.num_rel_labels)
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, all_entity_idxs=None, input_position=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=False, output_attentions=False, position_ids=input_position)
        sequence_output = outputs[0]
        mean_entity_embs = torch.cat([torch.mean(a[entity_idxs]) for a, entity_idxs in zip(sequence_output, all_entity_idxs)])
        rep = self.layer_norm(mean_entity_embs)
        rep = self.dropout(rep)
        logits = self.classifier(rep)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_rel_labels), labels.view(-1))
            return loss
        else:
            return logits

class RelationExtractor(pl.LightningModule):

    def __init__(self,
                 model: BertForRelation,
                 num_train_optimization_steps,
                 lr=1e-3,
                 correct_bias=True,
                 warmup_proportion=0.1,
    ):
        # TODO(Vijay): supply `num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs`

        # TODO(Vijay): configure these parameters via command line arguments.
        super().__init__()
        self.model = model
        self.num_train_optimization_steps = num_train_optimization_steps
        self.lr = lr
        self.correct_bias = correct_bias
        self.warmup_proportion = warmup_proportion

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(self.num_train_optimization_steps * self.warmup_proportion), self.num_train_optimization_steps)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def training_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: TokenClassifierOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs["labels"]
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss