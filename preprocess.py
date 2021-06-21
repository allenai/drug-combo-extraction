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

def find_no_combination_examples(relations, entities):
    # Find the set of all pairs of entities that belong in some relation (other than NOT-COMB) together in the same sentence.
    entity_cooccurrences = set()
    for relation in relations:
        if relation["class"] != NOT_COMB:
            span_idxs = sorted(relation["spans"])
            for i in range(len(span_idxs)):
                for j in range(i+1, len(span_idxs)):
                    entity_cooccurrences.add((i, j))

    no_comb_relations = []
    # Add implicit NOT-COMB relations.
    for i in range(len(entities)):
        for j in range(i+1, len(entities)):
            if (i, j) not in entity_cooccurrences:
                not_comb_relation = {'class': NOT_COMB, 'spans': [i, j]}
                no_comb_relations.append(not_comb_relation)
    return no_comb_relations

def relations_mergeable(r1, r2, recurse=True):
    r1_spans = set(r1['spans'])
    r2_spans = set(r2['spans'])
    intersection = r1_spans.intersection(r2_spans)
    same_label = r1['class'] == r2['class']
    common_entity = len(intersection) >= 1 and len(intersection) < min(len(r1_spans), len(r2_spans))
    # Merge the two relations if they:
    # - have the same relation label
    # - share at least one drug in common
    # - are not subsets/supersets of each other
    return same_label and common_entity

def merge(r1, r2):
    merged_relation = {
        "class": r1['class'],
        "spans": list(set(r1['spans']).union(set(r2['spans'])))
    }
    return merged_relation

def consolidate_relations(relations):
    # Iteratively merge relations that can be merged (and insert as new relations).
    # This is to capture transitivity among relations - i.e. r(a, b) ∧ r(b, c) => r(a, b, c).
    make_relation_hashable = lambda relation: (relation["class"], str(sorted(relation["spans"])))
    consolidated = relations
    consolidated_hashable = set([make_relation_hashable(c) for c in consolidated])
    while True:
        newly_consolidated = []
        newly_consolidated_hashable = consolidated_hashable
        for r1 in consolidated:
            for r2 in consolidated:
                if relations_mergeable(r1, r2):
                    merged_relation = merge(r1, r2)
                    merged_relation_hashable = (merged_relation["class"], str(sorted(merged_relation["spans"])))
                    if merged_relation_hashable not in consolidated_hashable and merged_relation_hashable not in newly_consolidated_hashable:
                        newly_consolidated.append(merged_relation)
                        newly_consolidated_hashable.add(merged_relation_hashable)
        if len(newly_consolidated) == 0:
            break
        consolidated.extend(newly_consolidated)
    return consolidated

def process_doc(raw, add_no_combination_relations=True, merge_relations_transitively=True):
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
    if merge_relations_transitively:
        # Merge relations transitively to form additional higher-order relations, when possible.
        relations = consolidate_relations(relations)

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
