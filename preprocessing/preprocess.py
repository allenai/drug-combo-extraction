from itertools import chain, combinations
import json
import random
from tqdm import tqdm
from typing import Dict, Iterable, List, Optional, Set

from common.constants import ENTITY_END_MARKER, ENTITY_START_MARKER, NOT_COMB, RELATION_UNKNOWN
from common.types import DrugEntity, DrugRelation, Document

random.seed(2021)

def pairset(iterable: Iterable) -> List[Set]:
    """Return the set of pairs of an iterable.
    Args:
        iterable: The iterable to take the set of pairs of.

    Returns:
        pairset: A list containing the set of pairs of `iterable`.
    """
    pairset = []
    for i, item_i in enumerate(iterable):
        for j, item_j in enumerate(iterable):
            if j > i:
                pairset.append(set([item_i, item_j]))
    return pairset

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

def find_no_combination_examples(relations: List[Dict], entities: List[DrugEntity], only_include_binary_no_comb_relations: bool = True, produce_all_subsets: bool = False) -> List[Dict]:
    """Construct NOT-COMB relations - relations that are not mentioned as being used in combination.
    We do this by exclusion - any set of entities that are not explicitly mentioned as being combined,
    are treated as NO_COMB.

    Args:
        relations: list of relations (each represented as a dict), directly taken from annotated data.
        entities: list of all drugs mentioned in the sentence of interest.

    Returns:
        no_comb_relations: list of relations between entities implicitly labeled as NO_COMB (by exclusion).
    """
    # Find the set of all pairs of entities that belong in some relation (other than NO_COMB) together in the same sentence.
    entity_cooccurrences = []
    for relation in relations:
        if relation["class"] != NOT_COMB:
            span_idxs = sorted(relation["spans"])
            entity_cooccurrences.append(set(span_idxs))

    entity_idxs = range(len(entities))
    if only_include_binary_no_comb_relations:
        # Under this option, construct only binary no-comb relations
        candidate_no_combinations = pairset(entity_idxs)
    else:
        candidate_no_combinations = powerset(entity_idxs)

    no_comb_relations = []
    # Add implicit NO_COMB relations.
    for candidate in candidate_no_combinations:
        entity_found = False
        for c in entity_cooccurrences:
            if (not produce_all_subsets and candidate.issubset(c)) or (candidate == c):
                entity_found = True
        # If a set of drugs is not contained in any other relation, then consider it as an implicit
        # NO_COMB relation.
        if not entity_found:
            no_comb_relation = {'class': NOT_COMB, 'spans': list(candidate)}
            no_comb_relations.append(no_comb_relation)
    return no_comb_relations

def process_doc(raw: Dict, label2idx: Dict, add_no_combination_relations: bool = True, only_include_binary_no_comb_relations: bool = False, include_paragraph_context: bool = True, produce_all_subsets: bool = False, max_candidate_entities: int = 15) -> Document:
    """Convert a raw annotated document into a Document class.

    Args:
        raw: Document from the Drug Synergy dataset, corresponding to one annotated sentence.
        label2idx: Mapping from relation class strings to integer values.
        add_no_combination_relations: Whether to add implicit NO_COMB relations.
        only_include_binary_no_comb_relations: If true, ignore n-ary no-comb relations.
        include_paragraph_context: Whether to include full-paragraph context around each drug-mention sentence
        produce_all_subsets: Whether to include all subsets of existing relations as NO_COMB or not

    Returns:
        document: Processed version of the input document.
    """
    if include_paragraph_context:
        text = raw['paragraph']
        sentence_start_idx = text.find(raw['sentence'])
        assert sentence_start_idx != -1, breakpoint()
    else:
        text = raw['sentence']
        sentence_start_idx = 0

    # Construct DrugEntity objects.
    drug_entities = []
    for idx, span in enumerate(raw['spans']):
        entity = DrugEntity(span['text'], idx, span['start'] + sentence_start_idx, span['end'] + sentence_start_idx)
        drug_entities.append(entity)

    if len(drug_entities) > max_candidate_entities:
        return None

    if produce_all_subsets:
        relation_label_mapping = {}
        if 'rels' in raw:
            for rel in raw['rels']:
                relation_label_mapping[tuple(sorted(rel["spans"]))] = rel["class"]
        relations = []
        for candidate in powerset(range(len(drug_entities))):
            relation_class = relation_label_mapping.get(tuple(sorted(candidate)), NOT_COMB)
            relation_dict = {'class': relation_class, 'spans': list(candidate)}
            relations.append(relation_dict)
    else:
        relations = raw['rels']
        if add_no_combination_relations:
            # Construct "NOT-COMB" relation pairs from pairs of annotated entities that do not co-occur in any other relation.
            relations = relations + find_no_combination_examples(relations, drug_entities, only_include_binary_no_comb_relations=only_include_binary_no_comb_relations, produce_all_subsets=produce_all_subsets)

    # Construct DrugRelation objects, which contain full information about the document's annotations.
    final_relations = []
    for relation in relations:
        entities = [drug_entities[entity_idx] for entity_idx in relation['spans']]
        rel_label = label2idx[relation['class']]
        final_relations.append(DrugRelation(entities, rel_label))
    document = Document(raw["doc_id"], final_relations, text)
    return document

def process_doc_with_unknown_relations(raw: Dict, label2idx: Dict, include_paragraph_context: bool = True) -> Document:
    doc_with_no_relations = raw.copy()
    doc_with_no_relations['rels'] = []
    document_with_unknown_relations = process_doc(raw, label2idx, add_no_combination_relations=True, include_paragraph_context=include_paragraph_context, produce_all_subsets=True)
    if document_with_unknown_relations is None:
        return None
    for relation in document_with_unknown_relations.relations:
        # Set all relation labels to be UNKNOWN_RELATION, to ensure no confusion
        relation.relation_label = RELATION_UNKNOWN
    return document_with_unknown_relations

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
    # This list keeps track of all the indices where special entity marker tokens were inserted.
    position_offsets = []
    for drug in relation_entities:
        # Insert "<m> " before each entity. Assuming that each entity is preceded by a whitespace, this will neatly
        # result in a whitespace-delimited "<m>" token before the entity.
        position_offset = sum([offset for idx, offset in position_offsets if idx <= drug.span_start])
        if not(drug.span_start + position_offset == 0 or text[drug.span_start + position_offset - 1] == " "):
            start_marker = " " + ENTITY_START_MARKER
        else:
            start_marker = ENTITY_START_MARKER
        assert drug.span_start + position_offset == 0 or text[drug.span_start + position_offset - 1] == " ", breakpoint()
        text = text[:drug.span_start + position_offset] + start_marker + " " + text[drug.span_start + position_offset:]
        position_offsets.append((drug.span_start, len(start_marker + " ")))

        # Insert "</m> " after each entity.
        position_offset = sum([offset for idx, offset in position_offsets if idx <= drug.span_end])
        if not(drug.span_end + position_offset == len(text) or text[drug.span_end + position_offset] == " "):
            end_marker = ENTITY_END_MARKER + " "
        else:
            end_marker = ENTITY_END_MARKER
        text = text[:drug.span_end + position_offset] + " " + end_marker + text[drug.span_end + position_offset:]
        position_offsets.append((drug.span_end, len(end_marker + " ")))
    return text

def truncate_text_into_window(text, context_window_size):
    tokens = text.split()
    first_entity_start_token = min([i for i, t in enumerate(tokens) if t == "<<m>>"])
    final_entity_end_token = max([i for i, t in enumerate(tokens) if t == "<</m>>"])
    entity_distance = final_entity_end_token - first_entity_start_token
    add_left = (context_window_size - entity_distance) // 2
    start_window_left = max(0, first_entity_start_token - add_left)
    add_right = (context_window_size - entity_distance) - add_left
    start_window_right = min(len(tokens), final_entity_end_token + add_right)
    return " ".join(tokens[start_window_left:start_window_right])

def create_datapoints(raw: Dict, label2idx: Dict, mark_entities: bool = True, add_no_combination_relations=True, only_include_binary_no_comb_relations: bool = False, include_paragraph_context=True, context_window_size: Optional[int] = None, produce_all_subsets: bool = False):
    """Given a single document, process it, add entity markers, and return a (text, relation label) pair.

    Args:
        raw: Dictionary of key-value pairs representing raw annotated document.
        label2idx: Mapping from relation class strings to integer values.
        mark_entities: Whether or not to add special entity token markers around each drug entity (default: True).
        add_no_combination_relations: If true, identify implicit "No-Combination" relations by negation.
        only_include_binary_no_comb_relations: If true, ignore n-ary no-comb relations.
        include_paragraph_context: If true, include paragraph context around each entity-bearing sentence.
        context_window_size: If set, we limit our paragraph context to this number of words
        produce_all_subsets: Whether to include all subsets of existing relations as NO_COMB or not

    Returns:
        samples: List of (text, relation label) pairs representing all positive/negative relations
                 contained in the sentence.
    """
    processed_document = process_doc(raw,
                                     label2idx,
                                     add_no_combination_relations=add_no_combination_relations,
                                     only_include_binary_no_comb_relations=only_include_binary_no_comb_relations,
                                     include_paragraph_context=include_paragraph_context,
                                     produce_all_subsets=produce_all_subsets)
    if processed_document is None:
        return []
    samples = []
    for relation in processed_document.relations:
        # Mark drug entities with special tokens.
        if mark_entities:
            text = add_entity_markers(processed_document.text, relation.drug_entities)
        else:
            text = processed_document.text
        if context_window_size is not None:
            text = truncate_text_into_window(text, context_window_size)
        drug_idxs = sorted([drug.drug_idx for drug in relation.drug_entities])
        row_metadata = {"doc_id": raw["doc_id"], "drug_idxs": drug_idxs, "relation_label": relation.relation_label}
        row_id = json.dumps(row_metadata)
        samples.append({"text": text, "target": relation.relation_label, "row_id": row_id, "drug_indices": drug_idxs})
    return samples

def create_dataset(raw_data: List[Dict],
                   label2idx: Dict,
                   shuffle: bool = True,
                   label_sampling_ratios=[1.0, 1.0],
                   add_no_combination_relations=True,
                   only_include_binary_no_comb_relations: bool = False,
                   include_paragraph_context=True,
                   context_window_size: Optional[int] = None,
                   produce_all_subsets: bool = False) -> List[Dict]:
    """Given the raw Drug Synergy dataset (directly read from JSON), convert it to a list of pairs
    consisting of marked text and a relation label, for each candidate relation in each document.

    Args:
        raw_data: List of documents in the dataset.
        label2idx: Mapping from relation class strings to integer values.
        shuffle: Whether or not to randomly reorder the relation instances in the dataset before returning.
        label_sampling_ratios: Ratio at which to downsample/upsample each class, to mitigate label imbalance.
        add_no_combination_relations: If true, identify implicit "No-Combination" relations by negation.
        only_include_binary_no_comb_relations: If true, ignore n-ary no-comb relations.
        include_paragraph_context: If true, include paragraph context around each entity-bearing sentence.
        context_window_size: If set, we limit our paragraph context to this number of words
        produce_all_subsets: Whether to include all subsets of existing relations as NO_COMB or not

    Returns:
        dataset: A list of text, label pairs (represented as a dictionary), ready to be consumed by a model.
    """
    label_values = sorted(list(set(label2idx.values())))
    dataset = []
    for row in tqdm(raw_data):
        datapoints = create_datapoints(row,
                                       label2idx,
                                       add_no_combination_relations=add_no_combination_relations,
                                       only_include_binary_no_comb_relations=only_include_binary_no_comb_relations,
                                       include_paragraph_context=include_paragraph_context,
                                       context_window_size=context_window_size,
                                       produce_all_subsets=produce_all_subsets)
        dataset.extend(datapoints)
    if set(label_sampling_ratios) != {1.0}:
        # If all classes' sampling ratios are uniform, then we can simply use the dataset as is.
        # Otherwise, sample points from each class and then accumulate them all together.
        upsampled_dataset = []
        for class_label in label_values:
            matching_points = [d for d in dataset if d["target"] == class_label]
            upsampled_points = random.choices(matching_points, k=int(len(matching_points) * label_sampling_ratios[class_label]))
            upsampled_dataset.extend(upsampled_points)
        dataset = upsampled_dataset
    if shuffle:
        random.shuffle(dataset)
    return dataset
