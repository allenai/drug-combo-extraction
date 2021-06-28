from itertools import chain, combinations
import random

from constants import ENTITY_END_MARKER, ENTITY_START_MARKER
from typing import Dict, Iterable, List, Set

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
        self.drug_name: str = drug_name
        self.span_start: int = span_start
        self.span_end: int = span_end

class DrugRelation:
    def __init__(self, drug_entities, relation_label):
        self.drug_entities: List[DrugEntity] = drug_entities
        self.relation_label: int = relation_label

class Document:
    def __init__(self, relations, text):
        self.relations: List[DrugRelation] = relations
        self.text: str = text

def powerset(iterable: Iterable) -> List[Set]:
    """Return the powerset of an iterable.
    Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes.

    Args:
        iterable: The iterable to take the powerset of.

    Returns:
        powerset: A list containing the powerset of `iterable`.
    """
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    powerset = [set(subset) for subset in powerset if len(subset) > 1]
    return powerset

def find_no_combination_examples(relations: List[Dict], entities: List[DrugEntity]) -> List[Dict]:
    """Construct NOT-COMB relations - relations that are not mentioned as being used in combination.
    We do this by exclusion - any set of entities that are not explicitly mentioned as being combined,
    are treated as NOT-COMB.

    Args:
        relations: list of relations (each represented as a dict), directly taken from annotated data.
        entities: list of all drugs mentioned in the sentence of interest.

    Returns:
        no_comb_relations: list of relations between entities implicitly labeled as NOT-COMB (by exclusion).
    """
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

def process_doc(raw: Dict, add_no_combination_relations: bool = True, include_paragraph_context: bool = True) -> Document:
    """Convert a raw annotated document into a Document class.

    Args:
        raw: Document from the Drug Synergy dataset, corresponding to one annotated sentence.
        add_no_combination_relations: Whether to add implicit NOT-COMB relations.
        include_paragraph_context: Whether to include full-paragraph context around each drug-mention sentence

    Returns:
        document: Processed version of the input document.
    """
    if include_paragraph_context:
        text = raw['paragraph']
        sentence_start_idx = text.find(raw['sentence'])
        assert sentence_start_idx != -1, "Sentence must be a substring of the containing paragraph."
    else:
        text = raw['sentence']
        sentence_start_idx = 0

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
        final_relations.append(DrugRelation(entities, rel_label))
    document = Document(final_relations, text)
    return document

def add_entity_markers(text: str, relation_entities: List[DrugEntity]) -> str:
    """Add special entity tokens around each drug entity in the annotated text.
    We specifically add "<<m>>" and "<</m>>" before and after (respectively) each drug entity span,
    and construct these tokens in such a way that they are always delimited by whitespace from the
    surrounding text.

    Args:
        text: Raw, un-tokenized text that has been annotated with relations and entity spans.
        relation_entities: List of entity objects, each describing the span of a drug mention.

    Returns:
        text: Raw text, with special entity tokens inserted around drug entities.
    """

    relation_entities: List = sorted(relation_entities, key=lambda entity: entity.span_start)
    position_offset = 0
    for drug in relation_entities:
        # Insert "<m> " before each entity. Assuming that each entity is preceded by a whitespace, this will neatly
        # result in a whitespace-delimited "<m>" token before the entity.
        assert text[drug.span_start + position_offset - 1] == " " or drug.span_start + position_offset == 0
        text = text[:drug.span_start + position_offset] + ENTITY_START_MARKER + " " + text[drug.span_start + position_offset:]
        position_offset += len(ENTITY_START_MARKER + " ")

        # Insert "</m> " after each entity.
        assert text[drug.span_end + position_offset] == " " or drug.span_end + position_offset == len(text) - 1
        text = text[:drug.span_end + position_offset + 1] + ENTITY_END_MARKER + " " + text[drug.span_end + position_offset + 1:]
        position_offset += len(ENTITY_END_MARKER + " ")
    return text

def create_datapoints(raw: Dict, mark_entities: bool = True, add_no_combination_relations=True, include_paragraph_context=True):
    """Given a single document, process it, add entity markers, and return a (text, relation label) pair.

    Args:
        raw: Dictionary of key-value pairs representing raw annotated document.
        mark_entities: Whether or not to add special entity token markers around each drug entity (default: True).

    Returns:
        samples: List of (text, relation label) pairs representing all positive/negative relations
                 contained in the sentence.
    """
    processed_document = process_doc(raw,
                                     add_no_combination_relations=add_no_combination_relations,
                                     include_paragraph_context=include_paragraph_context)
    samples = []
    for relation in processed_document.relations:
        # Mark drug entities with special tokens.
        if mark_entities:
            text = add_entity_markers(processed_document.text, relation.drug_entities)
        else:
            text = processed_document.text
        samples.append({"text": text, "target": relation.relation_label})
    return samples

def create_dataset(raw_data: List[Dict], shuffle: bool = True, sample_negatives_ratio=1.0, sample_positives_ratio=1.0) -> List[Dict]:
    """Given the raw Drug Synergy dataset (directly read from JSON), convert it to a list of pairs
    consisting of marked text and a relation label, for each candidate relation in each document.

    Args:
        raw_data: List of documents in the dataset.
        shuffle: Whether or not to randomly reorder the relation instances in the dataset before returning.
        sample_negatives_ratio: Ratio at which to sample negatives, to mitigate label imbalance.
        sample_positives_ratio: Ratio at which to sample positives, to mitigate label imbalance.

    Returns:
        dataset: A list of text, label pairs (represented as a dictionary), ready to be consumed by a model.
    """
    dataset = []
    for row in raw_data:
        datapoints = create_datapoints(row)
        dataset.extend(datapoints)
    if sample_negatives_ratio != 1.0 or sample_positives_ratio != 1.0:
        non_negatives = [d for d in dataset if d["target"] != 0]
        negatives = [d for d in dataset if d["target"] == 0]
        non_negatives = random.choices(non_negatives, k=int(len(non_negatives) * sample_positives_ratio))
        negatives = random.choices(negatives, k=int(len(negatives) * sample_negatives_ratio))
        dataset = non_negatives + negatives
    if shuffle:
        random.shuffle(dataset)
    return dataset
