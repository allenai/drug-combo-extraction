from model import BertForRelation
import os
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from data_loader import tokenize_sentence, vectorize_subwords
from utils import load_metadata

st.write("Enter marked abstract text:")
message_text = st.text_area("Enter a message for relation extraction (entities marked with <<m>> and <</m>>)", height=500)

@st.cache(allow_output_mutation=True)
def load_model(checkpoint_directory):
    model_name, max_seq_length, num_labels, label2idx = load_metadata(checkpoint_directory)
    model = BertForRelation.from_pretrained(
                checkpoint_directory,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                num_rel_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer.from_pretrained(os.path.join(checkpoint_directory, "tokenizer"))
    return model, tokenizer, max_seq_length, label2idx

def classify_message(message, model, tokenizer, max_seq_length, label2idx):
    subwords, entity_start_tokens = tokenize_sentence(message, tokenizer)
    vectorized_row = vectorize_subwords(tokenizer, subwords, max_seq_length)
    input_ids = torch.tensor(vectorized_row.input_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(vectorized_row.attention_mask, dtype=torch.long).unsqueeze(0)
    segment_ids = torch.tensor(vectorized_row.segment_ids, dtype=torch.long).unsqueeze(0)
    entity_start_tokens = torch.tensor(entity_start_tokens, dtype=torch.long).unsqueeze(0)

    logits = model(input_ids, token_type_ids=segment_ids, attention_mask=attention_mask, all_entity_idxs=entity_start_tokens)
    probabilities = torch.nn.functional.softmax(logits)[0]
    label = torch.argmax(probabilities).item()
    relation_probabilities = [round(prob, 4) for prob in probabilities.tolist()]
    return {'label': label, 'relation_probabilities': relation_probabilities}

if message_text != '':
    CHECKPOINT_DIRECTORY = "checkpoints"
    model, tokenizer, max_seq_length, label2idx = load_model(CHECKPOINT_DIRECTORY)

    model_output = classify_message(message_text, model, tokenizer, max_seq_length, label2idx)
    st.write(model_output)
