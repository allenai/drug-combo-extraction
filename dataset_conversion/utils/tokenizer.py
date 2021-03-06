import spacy
from spacy import Language

def make_tok_seg():
    '''
    Add a few special cases to spacy tokenizer so it works with ACe mistakes.
    '''
    # Prevent edge case where there are sentence breaks in bad places
    @Language.component('custom_seg')
    def custom_seg(doc):
        for index, token in enumerate(doc):
            if '--' in token.text:
                doc[index].sent_start = False
                if index < len(doc) - 1:
                    doc[index + 1].sent_start = False
            # Comma followed by whitespace doesn't end a sentence.
            if token.text == "," and  index < len(doc) - 2 and doc[index + 1].is_space:
                doc[index + 2].sent_start = False
            # "And" only starts a sentence if preceded by period or question mark.
            if token.text in ["and", "but"] and index >= 1 and doc[index - 1].text not in [".", "?", "!"]:
                doc[index].sent_start = False
            if (not ((token.is_punct and token.text not in [",", "_", ";", "...", ":", "(", ")", '"']) or token.is_space)
                and index < len(doc) - 1):
                doc[index + 1].sent_start = False
            if "\n" in token.text:
                if index + 1 < len(doc):
                    next_token = doc[index + 1]
                    if len(token) > 1:
                        next_token.sent_start = True
                    else:
                        next_token.sent_start = False
            if token.text == "-":
                if index > 0 and index < len(doc) - 1:
                    before = doc[index - 1]
                    after = doc[index + 1]
                    if not (before.is_space or before.is_punct or after.is_space or after.is_punct):
                        after.sent_start = False
        return doc

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe('custom_seg', before='parser')

    single_tokens = ['sgt.',
                        'sen.',
                        'col.',
                        'brig.',
                        'gen.',
                        'maj.',
                        'sr.',
                        'lt.',
                        'cmdr.',
                        'u.s.',
                        'mr.',
                        'p.o.w.',
                        'u.k.',
                        'u.n.',
                        'ft.',
                        'dr.',
                        'd.c.',
                        'mt.',
                        'st.',
                        'snr.',
                        'rep.',
                        'ms.',
                        'capt.',
                        'sq.',
                        'jr.',
                        'ave.']
    for special_case in single_tokens:
        nlp.tokenizer.add_special_case(special_case, [dict(ORTH=special_case)])
        upped = special_case.upper()
        nlp.tokenizer.add_special_case(upped, [dict(ORTH=upped)])
        capped = special_case.capitalize()
        nlp.tokenizer.add_special_case(capped, [dict(ORTH=capped)])

    split_tokens = [
                        ("Fe.", ("Fe", ".")),
                        ("A.", ("A", ".")),
                   ]
    for (orig, (split_a, split_b)) in split_tokens:
        nlp.tokenizer.add_special_case(orig, [dict(ORTH=split_a), dict(ORTH=split_b)])

    return nlp