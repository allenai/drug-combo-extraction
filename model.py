import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel

from allennlp.modules import FeedForward
from allennlp.nn.util import batched_index_select
import torch.nn.functional as F

BertLayerNorm = torch.nn.LayerNorm

# Adapted from https://github.com/princeton-nlp/PURE
class BertForRelation(BertPreTrainedModel):
    def __init__(self, num_rel_labels, hidden_dropout_prob, hidden_size, model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super(BertForRelation, self).__init__()
        self.num_labels = num_rel_labels
        self.bert = BertModel.from_pretrained(model)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = BertLayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, self.num_labels)
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
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
