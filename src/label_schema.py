RELATION_LABELS = [
    "announced_by",
    "mentions",
    "requires_qualification",
    "requires_document",
    "has_deadline",
    "HAS_FEE",
    "HAS_DEADLINE",
    "REQUIRES_DOCUMENT",
    "HAS_CONTACT_EMAIL",
    "HAS_CONTACT_PHONE",
    "REFERENCES_ATTACHMENT",
    "REFERENCES_EXTERNAL_RESOURCE",
    "MENTIONS_EXAM_LEVEL",
    "NO_RELATION",
]


ENTITY_TYPES = [
    "Notice",
    "Department",
    "Person",
    "Event",
    "Scholarship",
    "Target_Audience",
    "Visa",
    "Document",
    "Deadline",
    "Fee",
    "Email",
    "Phone",
    "ExternalResource",
    "Attachment",
    "ExamLevel",
    "Unknown",
]


KG_TO_RE_LABEL = {
    "REFERENCES_ATTACHMENT": "REFERENCES_ATTACHMENT",
    "REFERENCES_EXTERNAL_RESOURCE": "REFERENCES_EXTERNAL_RESOURCE",
    "REFERENCES_APPLICATION": "REFERENCES_EXTERNAL_RESOURCE",
    "CONTACT_EMAIL": "HAS_CONTACT_EMAIL",
    "CONTACT_POINT": "HAS_CONTACT_EMAIL",
    "CONTACT_PHONE": "HAS_CONTACT_PHONE",
    "MENTIONS_DATE": "HAS_DEADLINE",
    "HAS_FEE": "HAS_FEE",
    "MENTIONS_EXAM_LEVEL": "MENTIONS_EXAM_LEVEL",
}


def is_supported_relation(relation_type: str) -> bool:
    return relation_type in KG_TO_RE_LABEL
