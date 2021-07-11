from .convert_scierc import detokenize_sentence, convert_scierc_rows

def test_detokenize_sentence():
    tokens = ["This", "is", "a", "test", "sentence", "."]
    detokenized_sentence, token_char_mapping = detokenize_sentence(tokens, 20)
    correct_detokenized_sentence = "This is a test sentence ."
    assert detokenized_sentence == correct_detokenized_sentence
    correct_token_char_mapping = {20: (0, 4), 21: (5, 7), 22: (8, 9), 23: (10, 14), 24: (15, 23), 25: (24, 25)}
    assert token_char_mapping == correct_token_char_mapping

def test_convert_scierc_rows():
    sentences = [["Self-attention", "models", "are", "used", "for", "NLP", "."], ["Moreover", ",", "self-attention", "is", "one", "kind", "of", "neural", "network", "component", "."]]
    test_scierc_row = {
        "sentences": sentences,
        "ner": [[[0, 0, "Method"], [5, 5, "OtherScientificTerm"]], [[9, 9, "Method"], [14, 16, "OtherScientificTerm"]]],
        "relations": [[[0, 0, 5, 5, "USED-FOR"]], [[9, 9, 14, 16, "HYPONYM-OF"]]],
        "doc_key": "test"
    }

    converted_row = convert_scierc_rows(test_scierc_row)
    correct_paragraph = "Self-attention models are used for NLP . Moreover , self-attention is one kind of neural network component ."
    correct_first_sentence = "Self-attention models are used for NLP ."
    correct_second_sentence = "Moreover , self-attention is one kind of neural network component ."
    assert converted_row[0]["paragraph"] == correct_paragraph
    assert converted_row[1]["paragraph"] == correct_paragraph
    assert converted_row[0]["sentence"] == correct_first_sentence
    assert converted_row[1]["sentence"] == correct_second_sentence
    correct_spans = [[{
            "span_id": 0,
            "text": "Self-attention",
            "start": 0,
            "end": 14,
            "token_start": 0,
            "token_end": 0
        },
        {
            "span_id": 1,
            "text": "NLP",
            "start": 35,
            "end": 38,
            "token_start": 5,
            "token_end": 5
        }],
        [{
            "span_id": 2,
            "text": "self-attention",
            "start": 11,
            "end": 25,
            "token_start": 2,
            "token_end": 2
        },
        {
            "span_id": 3,
            "text": "neural network component",
            "start": 41,
            "end": 65,
            "token_start": 7,
            "token_end": 9
        }]]
    assert converted_row[0]["spans"] == correct_spans[0]
    assert converted_row[1]["spans"] == correct_spans[1]
    assert converted_row[0]["sentence"][0:14] == "Self-attention"
    assert converted_row[0]["sentence"][35:38] == "NLP"
    assert converted_row[1]["sentence"][11:25] == "self-attention"
    assert converted_row[1]["sentence"][41:65] == "neural network component"