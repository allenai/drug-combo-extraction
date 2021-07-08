from model import BertForRelation
import os
import streamlit as st
import torch
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from data_loader import tokenize_sentence, vectorize_subwords
from utils import load_metadata

st.write("Enter marked abstract text:")
message_text = st.text_area("Enter a message for relation extraction (entities marked with <<m>> and <</m>>)",
                            "The impact of schedule on acute toxicity and dose-intensity of high-dose chemotherapy with epirubicin and cyclophosphamide plus colony stimulating factors in advanced breast cancer.  To increase the dose-intensity of two drugs in metastatic breast cancer , we tested the feasibility , in phase I studies , of two schedules of <<m>> epirubicin <</m>> ( E ) and <<m>> cyclophosphamide <</m>> ( C ) - sequential ( E-- > C ) and alternating ( E/C ) - with respect to the standard combination ( EC ) . Drugs were given at three planned-dose levels, plus either G-CSF or GM-CSF. Patients with metastatic (30), inoperable stage IIIb (2) or inflammatory (7) breast cancer were treated. The doses of EC, given every 21 days (4 cycles), were 75/1500, 82.5/2250, 90/3000 mg/m2. In the E/C schedule, epirubicin was given at cycles 1, 3 and 5, and cyclophosphamide at cycles 2, 4 and 6. In the E--> C schedule, three cycles of epirubicin then three cycles of cyclophosphamide were administered. In both experimental schedules, drugs were given every 14 days for 6 cycles at doses of 100, 110, 120 mg/m2 (E) and 2000, 3000, 4000 mg/m2 (C). The average relative dose-intensity was 1.2-fold and 2-fold greater with E/C and E--> C, respectively, than with EC. The third level dose was feasible with all schedules. Grade 4 leucopenia occurred in 77% of patients. Thrombocytopenia was absent in 6 cases and grade 4 in 12 (30.8%). Eighty-one percent of patients on experimental schedules required red blood cell support versus 44.4% of patients on EC. At the third level, platelet transfusions were more frequent among patients treated with EC (27. 8%). Non-haematological toxicity was mild: about 20% of patients experienced grade 3 vomiting, irrespective of schedule. Only 2 patients had grade 3 mucositis; no patient developed heart failure. Fever (61% of patients) and bone pain (55.5% of patients) were relevant in the GM-CSF treated groups and 12 patients shifted to G-CSF. The overall response rate was 84.6%: 5/39 (12.8%) complete response and 28/39 (71.8%) partial response. At 30/9/98, median survival was 29.5 months, with no difference between patients with metastatic and stage IIIb/inflammatory breast cancer. Median follow-up of surviving patients was 62 months (range 17-83). The 5-year estimated survival was 19% (95% confidence intervals = 7-31%). Rapidly alternating or sequential cycles of epirubicin and cyclophosphamide with CSF support is a feasible strategy that allows a higher increase of dose-intensity of the single drugs. Hospitalization and anemia were more frequent with the experimental schedules, and thrombocytopenia with the standard schedule. Overall, this intensified therapy was very active.",
                            height=500)

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
