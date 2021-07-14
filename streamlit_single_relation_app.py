from model import BertForRelation
import os
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from typing import Dict, Tuple

from data_loader import tokenize_sentence, vectorize_subwords
from utils import load_metadata

@st.cache(allow_output_mutation=True)
def load_model(checkpoint_directory: str) -> Tuple[BertForRelation, AutoTokenizer, int]:
    '''Given a directory containing a model checkpoint, return the model, and other metadata regarding the data
    preprocessing that the model expects.

    Args:
        checkpoint_directory: Path to local directory where model is serialized

    Returns:
        model: Pretrained BertForRelation model
        tokenizer: Hugging Face tokenizer loaded from disk
        max_seq_length: Maximum number of subwords in a document allowed by the model (if longer, truncate input)
    '''
    metadata = load_metadata(checkpoint_directory)
    model = BertForRelation.from_pretrained(
                checkpoint_directory,
                cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
                num_rel_labels=metadata.num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(metadata.model_name, do_lower_case=True)
    tokenizer.from_pretrained(os.path.join(checkpoint_directory, "tokenizer"))
    return model, tokenizer, metadata

def classify_message(message: str,
                     model: BertForRelation,
                     tokenizer: AutoTokenizer,
                     max_seq_length: int) -> Dict:
    '''Given a string of text from a biomedical abstract (with drug entities marked), classify whether a relation exists between these entities.

    Args:
        message: JSON row from the Drug Synergy dataset
        model: Pretrained BertForRelation model object
        tokenizer: Hugging Face tokenizer loaded from disk
        max_seq_length: Maximum number of subwords in a document allowed by the model (if longer, truncate input)

    Returns:
        output: Dictionary containing the top predicted relation label, and the predicted probabilities of all relations.
    '''
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
    return {'predicted label': label, 'relation probabilities': relation_probabilities}

def app():
    st.write("Enter marked abstract text:")
    message_text = st.text_area("Enter a message for relation extraction (entities marked with <<m>> and <</m>>)",
                                "Phase Ib Study of the Oral Proteasome Inhibitor <<m>> Ixazomib <</m>> ( MLN9708 ) and <<m>> Fulvestrant <</m>> in Advanced ER+ Breast Cancer Progressing on Fulvestrant . fulvestrant is a selective estrogen receptor (ER)-downregulating anti-estrogen that blocks ER transcriptional activity and is approved for ER+ breast cancer. fulvestrant also induces accumulation of insoluble ER and activates an unfolded protein response; proteasome inhibitors have been shown to enhance these effects in preclinical models. ### background fulvestrant is a selective estrogen receptor (ER)-downregulating anti-estrogen that blocks ER transcriptional activity and is approved for ER+ breast cancer. fulvestrant also induces accumulation of insoluble ER and activates an unfolded protein response; proteasome inhibitors have been shown to enhance these effects in preclinical models. ### methods This is a single-center phase Ib study with a 3+3 design of fulvestrant and the proteasome inhibitor ixazomib (MLN9708) in patients with advanced ER+ breast cancer that was progressing on fulvestrant. A dose-escalation design allowed establishment of the ixazomib maximum tolerated dose (MTD). Secondary objectives included progression-free survival, pharmacokinetics, and tumor molecular analyses. ### results Among nine evaluable subjects, treatment was well-tolerated without dose-limiting toxicities The MTD of ixazomib was 4 mg in combination with fulvestrant. Plasma concentrations of the active form of ixazomib (MLN2238) in the 4-mg dose cohort had a median (range) Cmax of 155 (122-171) ng/mL; Tmax of 1 (1-1.5) h; terminal elimination half-life of 66.6 (57.3-102.6) hr after initial dose; AUC of 5,025 (4,160-5,345) ng*h/mL. One partial response was observed, and median progression-free survival was 51 days (range 47-137). ### conclusion This drug combination has a favorable safety profile and anti-tumor activity in patients with fulvestrant-resistant advanced ER+ breast cancer that justifies future testing.",
                                height=500)

    if message_text != '':
        CHECKPOINT_DIRECTORY = "checkpoints"
        model, tokenizer, metadata = load_model(CHECKPOINT_DIRECTORY)

        model_output = classify_message(message_text, model, tokenizer, metadata.max_seq_length)
        st.write(model_output)
