import argparse
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from src.io_utils import read_json, write_csv, write_json, write_jsonl
from src.label_schema import KG_TO_RE_LABEL


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(text: str) -> str:
    if "<" not in text:
        return compact(text)
    soup = BeautifulSoup(text, "html.parser")
    return compact(soup.get_text(" ", strip=True))


def marker_text(context: str, head: str, tail: str) -> str:
    context = compact(context)
    head = compact(head)
    tail = compact(tail)
    if not context:
        context = f"{head} {tail}"
    original = context
    head_pos = original.find(head) if head else -1
    tail_pos = original.find(tail) if tail else -1
    spans = []
    if head_pos >= 0:
        spans.append((head_pos, head_pos + len(head), "E1", head))
    if tail_pos >= 0:
        spans.append((tail_pos, tail_pos + len(tail), "E2", tail))

    if len(spans) == 2:
        spans = sorted(spans, key=lambda item: item[0])
        first, second = spans
        if first[1] <= second[0]:
            marked = (
                original[: first[0]]
                + f"[{first[2]}] {first[3]} [/{first[2]}]"
                + original[first[1] : second[0]]
                + f"[{second[2]}] {second[3]} [/{second[2]}]"
                + original[second[1] :]
            )
            return marked

    if head:
        context = context.replace(head, f"[E1] {head} [/E1]", 1) if head in context else f"[E1] {head} [/E1] {context}"
    if tail:
        context = context.replace(tail, f"[E2] {tail} [/E2]", 1) if tail in context else f"{context} [E2] {tail} [/E2]"
    return context


def make_sample(
    source_url: str,
    sentence: str,
    head_text: str,
    head_type: str,
    tail_text: str,
    tail_type: str,
    relation: str,
    weak_label: bool = True,
    kg_relation: str = "",
) -> dict[str, Any]:
    sentence = compact(sentence)
    head_text = compact(head_text)
    tail_text = compact(tail_text)
    return {
        "id": stable_id(f"{source_url}|{sentence[:160]}|{head_text}|{tail_text}|{relation}"),
        "source_url": source_url,
        "sentence": sentence,
        "marked_text": marker_text(sentence, head_text, tail_text),
        "head": {"id": stable_id(f"{head_type}|{head_text}"), "text": head_text, "type": head_type},
        "tail": {"id": stable_id(f"{tail_type}|{tail_text}"), "text": tail_text, "type": tail_type},
        "relation": relation,
        "kg_relation": kg_relation or relation,
        "weak_label": weak_label,
    }


def load_documents(path: str) -> dict[str, dict[str, Any]]:
    docs = read_json(path)
    return {doc["url"]: doc for doc in docs}


def find_context(source_url: str, relation: dict[str, Any], target_node: dict[str, Any], docs_by_url: dict[str, dict[str, Any]]) -> str:
    props = relation.get("properties", {})
    for key in ("evidence_text", "anchor_text"):
        if props.get(key):
            return props[key]

    doc = docs_by_url.get(source_url)
    target_name = compact(target_node.get("name", ""))
    if not doc:
        return target_name

    for section in doc.get("sections", []):
        text = compact(section.get("text", ""))
        if target_name and target_name in text:
            return text[:600]
    return compact(doc.get("full_text", ""))[:600]


def build_positive_samples(kg_path: str, docs_path: str) -> list[dict[str, Any]]:
    kg = read_json(kg_path)
    nodes = {node["id"]: node for node in kg["nodes"]}
    docs_by_url = load_documents(docs_path)
    samples: list[dict[str, Any]] = []

    for relation in kg["relations"]:
        kg_type = relation["type"]
        if kg_type not in KG_TO_RE_LABEL:
            continue
        label = KG_TO_RE_LABEL[kg_type]
        source_node = nodes.get(relation["source"], {})
        target_node = nodes.get(relation["target"], {})
        props = relation.get("properties", {})
        source_url = props.get("source_url") or source_node.get("url") or target_node.get("source_url", "")
        head = compact(source_node.get("name") or source_node.get("url") or relation["source"])
        tail = compact(target_node.get("name") or target_node.get("url") or relation["target"])
        context = find_context(source_url, relation, target_node, docs_by_url)
        head_type = source_node.get("type", "Unknown")
        tail_type = target_node.get("type", "Unknown")
        if label == "HAS_CONTACT_EMAIL":
            tail_type = "Email"
        elif label == "HAS_CONTACT_PHONE":
            tail_type = "Phone"
        elif label == "REFERENCES_ATTACHMENT":
            tail_type = "Attachment"
        elif label == "REFERENCES_EXTERNAL_RESOURCE":
            tail_type = "ExternalResource"
        elif label == "HAS_DEADLINE":
            tail_type = "Deadline"
        elif label == "MENTIONS_EXAM_LEVEL":
            tail_type = "ExamLevel"
        sample = make_sample(source_url, context, head, head_type, tail, tail_type, label, True, kg_type)
        sample["head"]["id"] = relation["source"]
        sample["tail"]["id"] = relation["target"]
        if len(sample["marked_text"]) > 30:
            samples.append(sample)

    deduped = {}
    for sample in samples:
        key = (sample["marked_text"], sample["relation"], sample["head"]["type"], sample["tail"]["type"])
        deduped[key] = sample
    return list(deduped.values())


def split_sentences(text: str) -> list[str]:
    text = compact(text)
    parts = re.split(r"(?<=[.!?。])\s+|(?=□)|(?=\[)|(?=-\s)|\n+", text)
    sentences = [compact(part) for part in parts if len(compact(part)) >= 12]
    if not sentences and text:
        sentences = [text]
    return sentences


def extract_emails(text: str) -> list[str]:
    return sorted(set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)))


def extract_phones(text: str) -> list[str]:
    return sorted(set(re.findall(r"(?:\+82[-\s]?)?0?\d{1,2}[-\s]\d{3,4}[-\s]\d{4}", text)))


def extract_fees(text: str) -> list[str]:
    return sorted(set(re.findall(r"\d{1,3}(?:,\d{3})+\s?원", text)))


def extract_dates(text: str) -> list[str]:
    patterns = [
        r"20\d{2}[./-]\s?\d{1,2}[./-]\s?\d{1,2}(?:\.\([^)]+\))?(?:\s?\d{1,2}:\d{2})?",
        r"\d{1,2}월\s*\d{1,2}일(?:\s?\d{1,2}:\d{2})?",
    ]
    dates: list[str] = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return sorted(set(compact(date) for date in dates))


def strip_dates(text: str) -> str:
    text = re.sub(r"20\d{2}[./-]\s?\d{1,2}[./-]\s?\d{1,2}(?:\.\([^)]+\))?(?:\s?\d{1,2}:\d{2})?", "", text)
    text = re.sub(r"\d{1,2}월\s*\d{1,2}일(?:\s?\d{1,2}:\d{2})?", "", text)
    return compact(text).strip("-_./: ")


def extract_exam_levels(text: str) -> list[str]:
    return sorted(set(re.findall(r"TOPIK\s?[ⅠⅡI]{1,2}|TOPIK\s?\d급|토픽\s?\d급|TOPIK\d급", text, flags=re.IGNORECASE)))


def extract_documents(text: str) -> list[str]:
    candidates = re.findall(r"[가-힣A-Za-z0-9·()\s]{2,30}(?:신청서|증명서|사본|원본|여권|외국인등록증|서류|사진|수정테이프)", text)
    cleaned = []
    for candidate in candidates:
        candidate = compact(candidate).strip("-: ")
        if 2 <= len(candidate) <= 40:
            cleaned.append(candidate)
    return sorted(set(cleaned))


def split_document_names(text: str) -> list[str]:
    text = html_to_text(text)
    phrase_patterns = [
        r"입학원서",
        r"자기소개서",
        r"유학생 보험가입 서약서",
        r"재정 및 신원보증서",
        r"개인정보 제공 및 활용 동의서",
        r"입학서약서",
        r"사진",
        r"본인 여권 사본",
        r"재정보증인 신분증 사본",
        r"재정보증인 재직증명서",
        r"사업자등록증",
        r"신분증 사본\(본인 및 부모\)",
        r"가족관계증명서",
        r"출생증명서 번역공증본 원본",
        r"친족관계증명서",
        r"호구부 번역공증본 원본",
        r"이혼증명서 번역공증본 원본",
        r"사망증명서 번역공증본 원본",
        r"은행예금잔고증명서 원본",
        r"최종학력 졸업증명서\(사본\)",
        r"성적증명서\(사본\)",
        r"학력인증서\(원본\)",
    ]
    phrase_docs = []
    for pattern in phrase_patterns:
        phrase_docs.extend(re.findall(pattern, text))

    text = re.sub(r"▶\s*첨부서류\s*:", " ", text)
    text = re.sub(r"※[^,，;；]+", " ", text)
    text = re.sub(r"▶[^,，;；]+", " ", text)
    text = re.sub(r"\([^)]{15,}\)", " ", text)
    text = re.sub(r"\b\d+\.\s*", " ", text)
    text = re.sub(r"\s+\d+부\b|\s+\d+매\b|각\s+\d+부\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    raw_parts = re.split(r",|，|;|；| 또는 |\s/\s", text)
    docs: list[str] = list(phrase_docs)
    for part in raw_parts:
        part = normalize_document_name(part)
        part = re.sub(r"\s*추가 제출$", "", part).strip()
        if not part:
            continue
        if len(part) > 45:
            continue
        if " 및 " in part and any(doc in part for doc in phrase_docs):
            continue
        if any(bad in part for bad in ["수 있는", "이어야 함", "에서 발급", "첨부서류", "신청 서류"]):
            continue
        if any(
            keyword in part
            for keyword in [
                "신청서",
                "자기소개서",
                "서약서",
                "보증서",
                "동의서",
                "사진",
                "여권",
                "신분증",
                "증명서",
                "사본",
                "원본",
                "학력인증서",
                "사업자등록증",
                "호구부",
                "원서",
                "자기소개서",
            ]
        ):
            docs.append(part)
    return sorted(set(normalize_document_name(doc) for doc in docs if normalize_document_name(doc)))


def normalize_document_name(text: str) -> str:
    text = compact(text)
    text = re.sub(r"\s*\(본교 양식\)?$", "", text)
    text = re.sub(r"\s*\(사본\)?$", "(사본)", text)
    text = re.sub(r"\s*\(원본\)?$", "(원본)", text)
    text = text.strip("-: ")
    text = re.sub(r"\s*각\s*$", "", text)
    return text


def extract_table_document_samples(doc: dict[str, Any], topic_name: str, topic_type: str) -> list[dict[str, Any]]:
    samples = []
    source_url = doc["url"]
    table_context = " ".join(section.get("title", "") + " " + section.get("text", "")[:200] for section in doc.get("sections", []))
    if not any(keyword in table_context for keyword in ["서류", "비자", "신청"]):
        return samples

    document_topic_name = topic_name
    document_topic_type = topic_type
    for section in doc.get("sections", []):
        section_title = compact(section.get("title", ""))
        section_text = compact(section.get("text", ""))
        if "서류" in section_title or "비자 신청" in section_text[:120]:
            document_topic_name = section_title or extract_topic_title(section_text, topic_name)
            document_topic_type = "Visa" if any(token in document_topic_name + section_text[:200] for token in ["비자", "D-2", "D-4"]) else topic_type
            break

    for table in doc.get("tables", []):
        for row in table.get("rows", []):
            if len(row) < 2:
                continue
            row_text = html_to_text(" ".join(row[1:]))
            if not any(keyword in row_text for keyword in ["원서", "자기소개서", "신청서", "서약서", "보증서", "동의서", "사진", "여권", "신분증", "증명서", "사본", "원본", "학력인증서", "사업자등록증", "호구부"]):
                continue
            for doc_name in split_document_names(row_text):
                samples.append(make_sample(source_url, row_text, document_topic_name, document_topic_type, doc_name, "Document", "requires_document"))
                samples.append(make_sample(source_url, row_text, document_topic_name, document_topic_type, doc_name, "Document", "REQUIRES_DOCUMENT"))
    return samples


def has_document_table(doc: dict[str, Any]) -> bool:
    for table in doc.get("tables", []):
        for row in table.get("rows", []):
            row_text = html_to_text(" ".join(row))
            if any(keyword in row_text for keyword in ["신청서", "서약서", "증명서", "사본", "원본", "여권", "신분증"]):
                return True
    return False


def extract_departments(text: str) -> list[str]:
    candidates = re.findall(r"(?:국제교류처|[가-힣A-Za-z0-9]+(?:팀|센터|처|부서|사무소))", text)
    return sorted(set(compact(candidate) for candidate in candidates if len(compact(candidate)) >= 3))


def extract_audiences(text: str) -> list[str]:
    audiences = []
    for sentence in split_sentences(text):
        if any(keyword in sentence for keyword in ["대상", "자격", "TOPIK", "재학생", "졸업예정자", "외국인유학생", "학부"]):
            audiences.append(sentence[:180])
    return sorted(set(audiences))


def infer_topic_entity(text: str, fallback_title: str) -> tuple[str, str]:
    title = extract_topic_title(text, fallback_title)
    if any(keyword in text for keyword in ["장학", "장학금", "지원금"]):
        return title or "장학금", "Scholarship"
    if any(keyword in title + " " + text[:250] for keyword in ["시험", "TOPIK", "토픽", "설명회", "수여식", "프로그램", "접수"]):
        return title or "행사/일정", "Event"
    if any(keyword in text for keyword in ["비자", "체류", "D-2", "D-4", "외국인등록"]):
        visa_match = re.search(r"D-[24]", text)
        return visa_match.group(0) if visa_match else title or "비자/체류", "Visa"
    return title or "행사/일정", "Event"


def extract_topic_title(text: str, fallback_title: str) -> str:
    title = compact(fallback_title)
    generic_titles = {"", "공지", "공지사항", "section-1", "section-2", "section-3"}
    if title not in generic_titles and not title.startswith("section-"):
        return strip_dates(title) or title

    text = compact(text)
    patterns = [
        r"20\d{2}\s?학년도\s?[^\[\]□\n]{4,80}?(?:시험|설명회|수여식|모집|접수|프로그램|신청|발표|안내)",
        r"20\d{2}[-\s]?[^\[\]□\n]{4,80}?(?:시험|설명회|수여식|모집|접수|프로그램|신청|발표|안내)",
        r"\[\s*([^\]]{4,80}?(?:시험|설명회|수여식|모집|접수|프로그램|신청|발표|안내))\s*\]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            value = re.sub(r"^공지사항\s*\d{1,2}\s*20\d{2}\s*\.\s*\d{1,2}\s*", "", value)
            return strip_dates(value) or compact(value)

    for sentence in split_sentences(text):
        if any(keyword in sentence for keyword in ["시험", "설명회", "모집", "접수", "프로그램", "신청", "발표", "안내"]):
            return strip_dates(sentence[:90]) or sentence[:90]
    return strip_dates(text[:80]) or "행사/일정"


def build_domain_samples(docs_path: str) -> list[dict[str, Any]]:
    docs = read_json(docs_path)
    samples: list[dict[str, Any]] = []

    for doc in docs:
        source_url = doc["url"]
        notice_blocks = doc.get("notices") or []
        if not notice_blocks:
            notice_blocks = [
                {
                    "notice_id": doc.get("page_id", ""),
                    "title": section.get("title", doc.get("title", "")),
                    "text": section.get("text", ""),
                    "source_url": source_url,
                }
                for section in doc.get("sections", [])
            ]

        for notice in notice_blocks:
            text = compact(notice.get("text", ""))
            title = compact(notice.get("title") or doc.get("title", "공지"))
            if len(text) < 30:
                continue
            notice_name = extract_topic_title(text, title)
            topic_name, topic_type = infer_topic_entity(text, title)

            samples.append(make_sample(source_url, text[:500], notice_name, "Notice", topic_name, topic_type, "mentions"))

            for dept in extract_departments(text):
                samples.append(make_sample(source_url, text[:500], notice_name, "Notice", dept, "Department", "announced_by"))

            for email in extract_emails(text):
                samples.append(make_sample(source_url, text[:500], notice_name, "Notice", email, "Email", "HAS_CONTACT_EMAIL"))
            for phone in extract_phones(text):
                samples.append(make_sample(source_url, text[:500], notice_name, "Notice", phone, "Phone", "HAS_CONTACT_PHONE"))
            for fee in extract_fees(text):
                samples.append(make_sample(source_url, text[:500], topic_name, topic_type, fee, "Fee", "HAS_FEE"))
            for date in extract_dates(text):
                samples.append(make_sample(source_url, text[:500], topic_name, topic_type, date, "Deadline", "has_deadline"))
                samples.append(make_sample(source_url, text[:500], topic_name, topic_type, date, "Deadline", "HAS_DEADLINE"))
            if not has_document_table(doc):
                for doc_name in extract_documents(text):
                    samples.append(make_sample(source_url, text[:500], topic_name, topic_type, doc_name, "Document", "requires_document"))
                    samples.append(make_sample(source_url, text[:500], topic_name, topic_type, doc_name, "Document", "REQUIRES_DOCUMENT"))
            for audience in extract_audiences(text):
                samples.append(make_sample(source_url, text[:500], topic_name, topic_type, audience, "Target_Audience", "requires_qualification"))
            for level in extract_exam_levels(text):
                samples.append(make_sample(source_url, text[:500], topic_name, topic_type, level, "ExamLevel", "MENTIONS_EXAM_LEVEL"))

            samples.extend(extract_table_document_samples(doc, topic_name, topic_type))

        for attachment in [link for link in doc.get("links", []) if link.get("target_type") == "attachment"]:
            head = doc.get("title") or source_url
            tail = attachment.get("anchor_text") or attachment.get("target_url")
            samples.append(make_sample(source_url, tail, head, "Notice", tail, "Attachment", "REFERENCES_ATTACHMENT"))
        for external in [link for link in doc.get("links", []) if link.get("target_type") == "external"]:
            head = doc.get("title") or source_url
            tail = external.get("anchor_text") or external.get("target_url")
            samples.append(make_sample(source_url, tail, head, "Notice", tail, "ExternalResource", "REFERENCES_EXTERNAL_RESOURCE"))

    deduped = {}
    for sample in samples:
        key = (sample["source_url"], sample["marked_text"], sample["relation"], sample["head"]["type"], sample["tail"]["type"])
        deduped[key] = sample
    return list(deduped.values())


def build_negative_samples(positive_samples: list[dict[str, Any]], max_count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_url = defaultdict(list)
    for sample in positive_samples:
        by_url[sample["source_url"]].append(sample)

    negatives: list[dict[str, Any]] = []
    for source_url, group in by_url.items():
        if len(group) < 2:
            continue
        for sample in group:
            other = rng.choice(group)
            if sample["tail"]["id"] == other["tail"]["id"] or sample["relation"] == other["relation"]:
                continue
            context = sample["sentence"]
            negative = {
                "id": stable_id(f"neg|{sample['id']}|{other['tail']['id']}"),
                "source_url": source_url,
                "sentence": context,
                "marked_text": marker_text(context, sample["head"]["text"], other["tail"]["text"]),
                "head": sample["head"],
                "tail": other["tail"],
                "relation": "NO_RELATION",
                "kg_relation": "NO_RELATION",
                "weak_label": True,
            }
            negatives.append(negative)
            if len(negatives) >= max_count:
                return negatives
    return negatives


def stratified_split(samples: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample["relation"]].append(sample)

    splits = {"train": [], "dev": [], "test": []}
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        n = len(label_samples)
        train_end = max(1, int(n * 0.7))
        dev_end = train_end + max(1, int(n * 0.15)) if n >= 3 else train_end
        splits["train"].extend(label_samples[:train_end])
        splits["dev"].extend(label_samples[train_end:dev_end])
        splits["test"].extend(label_samples[dev_end:])

    for split_samples in splits.values():
        rng.shuffle(split_samples)
    return splits


def write_labeling_csv(path: Path, samples: list[dict[str, Any]], max_rows: int, seed: int) -> None:
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for sample in samples:
        by_label[sample["relation"]].append(sample)

    selected: list[dict[str, Any]] = []
    per_label = max(1, max_rows // max(1, len(by_label)))
    leftovers: list[dict[str, Any]] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        selected.extend(label_samples[:per_label])
        leftovers.extend(label_samples[per_label:])
    rng.shuffle(leftovers)
    selected.extend(leftovers[: max(0, max_rows - len(selected))])

    rows = []
    for sample in selected[:max_rows]:
        rows.append(
            {
                "id": sample["id"],
                "gold_relation": "",
                "suggested_relation": sample["relation"],
                "head_type": sample["head"]["type"],
                "head_text": sample["head"]["text"],
                "tail_type": sample["tail"]["type"],
                "tail_text": sample["tail"]["text"],
                "source_url": sample["source_url"],
                "sentence": context_window(sample["sentence"], sample["head"]["text"], sample["tail"]["text"]),
                "marked_text": context_window(sample["marked_text"], "[E1]", "[E2]"),
                "memo": "",
            }
        )
    write_csv(
        path,
        rows,
        [
            "id",
            "gold_relation",
            "suggested_relation",
            "head_type",
            "head_text",
            "tail_type",
            "tail_text",
            "source_url",
            "sentence",
            "marked_text",
            "memo",
        ],
    )

    by_relation_dir = path.parent / "labeling_by_relation"
    by_relation_dir.mkdir(parents=True, exist_ok=True)
    rows_by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_relation[row["suggested_relation"]].append(row)
    for relation, relation_rows in rows_by_relation.items():
        write_csv(
            by_relation_dir / f"{relation}.csv",
            relation_rows,
            [
                "id",
                "gold_relation",
                "suggested_relation",
                "head_type",
                "head_text",
                "tail_type",
                "tail_text",
                "source_url",
                "sentence",
                "marked_text",
                "memo",
            ],
        )


def context_window(text: str, head: str, tail: str, size: int = 180) -> str:
    text = compact(text)
    anchors = [anchor for anchor in [head, tail] if anchor and anchor in text]
    if not anchors:
        return text[: size * 2]
    positions = [text.find(anchor) for anchor in anchors]
    center = min(positions)
    start = max(0, center - size)
    end = min(len(text), center + size)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weak-labeled RE data from KG candidates.")
    parser.add_argument("--kg", default="data/kg/kg.json")
    parser.add_argument("--documents", default="data/processed/documents.json")
    parser.add_argument("--output-dir", default="data/re")
    parser.add_argument("--negative-ratio", type=float, default=0.35)
    parser.add_argument("--labeling-csv-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    positives = build_positive_samples(args.kg, args.documents) + build_domain_samples(args.documents)
    deduped = {}
    for sample in positives:
        key = (sample["source_url"], sample["marked_text"], sample["relation"], sample["head"]["text"], sample["tail"]["text"])
        deduped[key] = sample
    positives = list(deduped.values())
    negatives = build_negative_samples(positives, int(len(positives) * args.negative_ratio), args.seed)
    samples = positives + negatives
    splits = stratified_split(samples, args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "candidates.jsonl", samples)
    write_labeling_csv(output_dir / "labeling_sample.csv", samples, args.labeling_csv_size, args.seed)
    for split, split_samples in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_samples)

    summary = {
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "total_count": len(samples),
        "labeling_csv": str(output_dir / "labeling_sample.csv"),
        "splits": {split: len(split_samples) for split, split_samples in splits.items()},
    }
    write_json(output_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
