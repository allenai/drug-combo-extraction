from typing import List

class DrugEntity:
    def __init__(self, drug_name, drug_idx, span_start, span_end):
        self.drug_name: str = drug_name
        self.drug_idx: int = drug_idx
        self.span_start: int = span_start
        self.span_end: int = span_end

class DrugRelation:
    def __init__(self, drug_entities, relation_label):
        self.drug_entities: List[DrugEntity] = drug_entities
        self.relation_label: int = relation_label

class Document:
    def __init__(self, doc_id, relations, text):
        self.doc_id: str = doc_id
        self.relations: List[DrugRelation] = relations
        self.text: str = text