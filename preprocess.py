from itertools import chain, combinations
import random

from constants import ENTITY_END_MARKER, ENTITY_START_MARKER

random.seed(2021)

LABEL2IDX = {
    "POS": 1,
    "COMB": 0,
    "NEG": 0,
    "NOT-COMB": 0
}
NOT_COMB = "NOT-COMB"

class DrugEntity:
    def __init__(self, drug_name, span_start, span_end):
        self.drug_name = drug_name
        self.span_start = span_start
        self.span_end = span_end

class DrugRelation:
    def __init__(self, drug_entities, relation_label):
        self.drug_entities = drug_entities
        self.relation_label = relation_label

class Document:
    def __init__(self, relations, text):
        self.relations = relations
        self.text = text

def powerset(iterable):
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes.
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [set(subset) for subset in powerset if len(subset) > 1]

def find_no_combination_examples(relations, entities):
    # Find the set of all pairs of entities that belong in some relation (other than NOT-COMB) together in the same sentence.
    entity_cooccurrences = []
    for relation in relations:
        if relation["class"] != NOT_COMB:
            span_idxs = sorted(relation["spans"])
            entity_cooccurrences.append(set(span_idxs))

    entity_idxs = range(len(entities))
    candidate_no_combinations = powerset(entity_idxs)

    no_comb_relations = []
    # Add implicit NOT-COMB relations.
    for candidate in candidate_no_combinations:
        entity_found = False
        for c in entity_cooccurrences:
            if candidate.issubset(c):
                entity_found = True
        # If a set of drugs is not contained in any other relation, then consider it as an implicit
        # NOT-COMB relation.
        if not entity_found:
            no_comb_relation = {'class': NOT_COMB, 'spans': list(candidate)}
            no_comb_relations.append(no_comb_relation)
    return no_comb_relations

def process_doc(raw, add_no_combination_relations=True):
    text = raw['paragraph']
    sentence_start_idx = text.find(raw['sentence'])
    assert sentence_start_idx != -1, "Sentence must be a substring of the containing paragraph."

    # Construct DrugEntity objects.
    drug_entities = []
    for span in raw['spans']:
        entity = DrugEntity(span['text'], span['start'] + sentence_start_idx, span['end'] + sentence_start_idx)
        drug_entities.append(entity)


    relations = raw['rels']
    if add_no_combination_relations:
        # Construct "NOT-COMB" relation pairs from pairs of annotated entities that do not co-occur in any other relation.
        relations = relations + find_no_combination_examples(relations, drug_entities)

    # Construct DrugRelation objects, which contain full information about the document's annotations.
    final_relations = []
    for relation in relations:
        entities = [drug_entities[entity_idx] for entity_idx in relation['spans']]
        rel_label = LABEL2IDX[relation['class']]
        final_relations.append(DrugRelation(entities, rel_label) )
    return Document(final_relations, text)

def add_entity_markers(text, relation_entities):
    relation_entities = sorted(relation_entities, key=lambda entity: entity.span_start)
    position_offset = 0
    for drug in relation_entities:
        # Insert "<m> " before each entity. Assuming that each entity is preceded by a whitespace, this will neatly
        # result in a whitespace-delimited "<m>" token before the entity.
        assert text[drug.span_start + position_offset - 1] == " "
        text = text[:drug.span_start + position_offset] + ENTITY_START_MARKER + " " + text[drug.span_start + position_offset:]
        position_offset += len(ENTITY_START_MARKER + " ")

        # Insert "</m> " after each entity.
        assert text[drug.span_end + position_offset] == " "
        text = text[:drug.span_end + position_offset + 1] + ENTITY_END_MARKER + " " + text[drug.span_end + position_offset + 1:]
        position_offset += len(ENTITY_END_MARKER + " ")
    return text

def create_datapoints(raw, mark_entities=True):
    processed_document = process_doc(raw)
    samples = []
    for relation in processed_document.relations:
        # Mark drug entities with special tokens.
        if mark_entities:
            text = add_entity_markers(processed_document.text, relation.drug_entities)
        else:
            text = processed_document.text
        samples.append({"text": text, "target": relation.relation_label})
    return samples

def create_dataset(raw_data, shuffle=True):
    dataset = []
    for row in raw_data:
        datapoints = create_datapoints(row)
        dataset.extend(datapoints)
    if shuffle:
        random.shuffle(dataset)
    return dataset
