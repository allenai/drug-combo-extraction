import json
from model import BertForRelation
import os
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from constants import ENTITY_PAD_IDX
from data_loader import make_fixed_length, tokenize_sentence, vectorize_subwords
from preprocess import add_entity_markers, process_doc_with_unknown_relations
from utils import load_metadata

@st.cache(allow_output_mutation=True)
def load_model(checkpoint_directory):
    model_name, max_seq_length, num_labels, label2idx, include_paragraph_context = load_metadata(checkpoint_directory)
    model = BertForRelation.from_pretrained(
                checkpoint_directory,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                num_rel_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    tokenizer.from_pretrained(os.path.join(checkpoint_directory, "tokenizer"))
    return model, tokenizer, max_seq_length, label2idx, include_paragraph_context

def find_all_relations(message, model, tokenizer, max_seq_length, threshold, label2idx, label_of_interest=1, include_paragraph_context=True):
    '''TODO: docstrings and code cleanup
    '''
    paragraph = message["paragraph"]
    sentence = message["sentence"]
    spans = message["spans"]

    doc_with_unknown_relations = process_doc_with_unknown_relations(message, label2idx, include_paragraph_context=include_paragraph_context)
    marked_sentences = []
    relations = []
    for relation in doc_with_unknown_relations.relations:
        # Mark drug entities with special tokens.
        marked_sentence = add_entity_markers(doc_with_unknown_relations.text, relation.drug_entities)
        marked_sentences.append(marked_sentence)
        relations.append(tuple([f"{drug.drug_name} ({drug.span_start} - {drug.span_end})" for drug in relation.drug_entities]))

    all_entity_idxs = []
    all_input_ids = []
    all_token_type_ids = []
    all_attention_masks = []
    for sentence in marked_sentences:
        subwords, entity_start_tokens = tokenize_sentence(sentence, tokenizer)
        vectorized_row = vectorize_subwords(tokenizer, subwords, max_seq_length)
        all_input_ids.append(vectorized_row.input_ids)
        all_token_type_ids.append(vectorized_row.attention_mask)
        all_attention_masks.append(vectorized_row.segment_ids)
        all_entity_idxs.append(make_fixed_length(entity_start_tokens, len(spans), padding_value=ENTITY_PAD_IDX))

    all_entity_idxs = torch.tensor(all_entity_idxs, dtype=torch.long)
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)

    logits = model(all_input_ids, token_type_ids=all_token_type_ids, attention_mask=all_attention_masks, all_entity_idxs=all_entity_idxs)
    probability = torch.nn.functional.softmax(logits)
    label_probabilities = probability[:, label_of_interest].tolist()

    relation_probabilities = []
    for i, probability in enumerate(label_probabilities):
        if probability > threshold:
            relation_probabilities.append({"drugs": relations[i], "positive probability": probability})
    relation_probabilities = sorted(relation_probabilities, key=lambda x: x["positive probability"], reverse=True)
    return {'relations': relation_probabilities}

def get_ground_truth_relations(message):
    relations = message["rels"]
    formatted_relations = []
    for relation in relations:
        relation_drugs = []
        for span_idx in relation["spans"]:
            span = message["spans"][span_idx]
            drug_name = span["text"]
            span_start = span["start"]
            span_end = span["end"]
            drug_identifier = f"{drug_name} ({span_start} - {span_end})"
            relation_drugs.append(drug_identifier)
        formatted_relations.append({"drugs": relation_drugs, "label": relation["class"]})
    return formatted_relations

def app():
    example_dataset_row = '{"sentence": "Phase Ib Study of the Oral Proteasome Inhibitor Ixazomib ( MLN9708 ) and Fulvestrant in Advanced ER+ Breast Cancer Progressing on Fulvestrant .", "spans": [{"span_id": 0, "text": "Ixazomib", "start": 48, "end": 56, "token_start": 8, "token_end": 9}, {"span_id": 1, "text": "Fulvestrant", "start": 73, "end": 84, "token_start": 13, "token_end": 14}, {"span_id": 2, "text": "Fulvestrant", "start": 130, "end": 141, "token_start": 21, "token_end": 22}], "rels": [{"class": "POS", "spans": [0, 1]}], "paragraph": "Phase Ib Study of the Oral Proteasome Inhibitor Ixazomib ( MLN9708 ) and Fulvestrant in Advanced ER+ Breast Cancer Progressing on Fulvestrant . fulvestrant is a selective estrogen receptor (ER)-downregulating anti-estrogen that blocks ER transcriptional activity and is approved for ER+ breast cancer. fulvestrant also induces accumulation of insoluble ER and activates an unfolded protein response; proteasome inhibitors have been shown to enhance these effects in preclinical models. ### background fulvestrant is a selective estrogen receptor (ER)-downregulating anti-estrogen that blocks ER transcriptional activity and is approved for ER+ breast cancer. fulvestrant also induces accumulation of insoluble ER and activates an unfolded protein response; proteasome inhibitors have been shown to enhance these effects in preclinical models. ### methods This is a single-center phase Ib study with a 3+3 design of fulvestrant and the proteasome inhibitor ixazomib (MLN9708) in patients with advanced ER+ breast cancer that was progressing on fulvestrant. A dose-escalation design allowed establishment of the ixazomib maximum tolerated dose (MTD). Secondary objectives included progression-free survival, pharmacokinetics, and tumor molecular analyses. ### results Among nine evaluable subjects, treatment was well-tolerated without dose-limiting toxicities The MTD of ixazomib was 4 mg in combination with fulvestrant. Plasma concentrations of the active form of ixazomib (MLN2238) in the 4-mg dose cohort had a median (range) Cmax of 155 (122-171) ng/mL; Tmax of 1 (1-1.5) h; terminal elimination half-life of 66.6 (57.3-102.6) hr after initial dose; AUC of 5,025 (4,160-5,345) ng*h/mL. One partial response was observed, and median progression-free survival was 51\u2009days (range 47-137). ### conclusion This drug combination has a favorable safety profile and anti-tumor activity in patients with fulvestrant-resistant advanced ER+ breast cancer that justifies future testing.", "source": "https://pubmed.ncbi.nlm.nih.gov/33641211/"}'
    st.write("Copy and paste a row from the Drug Synergy Dataset:")
    message_json = st.text_area("Copy-and-paste a dataset row (in JSON format)",
                                example_dataset_row,
                                height=500)
    message = json.loads(message_json)
    ground_truth_relations = get_ground_truth_relations(message)

    DEFAULT_THRESHOLD = 0.3
    threshold = st.slider('Relation Threshold', min_value=0.0, max_value=1.0, value=DEFAULT_THRESHOLD, step=0.01)
    if threshold > 0.0:
        CHECKPOINT_DIRECTORY = "checkpoints"
        model, tokenizer, max_seq_length, label2idx, include_paragraph_context = load_model(CHECKPOINT_DIRECTORY)

        model_output = find_all_relations(message, model, tokenizer, max_seq_length, threshold, label2idx, label_of_interest=1, include_paragraph_context=include_paragraph_context)
        st.write("Predicted Relations:")
        st.write(model_output)

    st.write("Ground Truth Relations:")
    st.write(ground_truth_relations)
