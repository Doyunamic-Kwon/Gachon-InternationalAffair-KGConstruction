import argparse
import html
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse, urlunparse

import requests
import urllib3
from bs4 import BeautifulSoup, Comment


BASE_URL = "https://oia.gachon.ac.kr"
SEED_URL = "https://oia.gachon.ac.kr/clientMain/a/t/internationalMain.do"
WORKSPACE = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KG_DIR = DATA_DIR / "kg"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

SKIP_EXTENSIONS = {
    ".css",
    ".js",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".ico",
    ".zip",
    ".hwp",
    ".hwpx",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}

DOCUMENT_EXTENSIONS = {".pdf", ".hwp", ".hwpx", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}


@dataclass
class CrawlConfig:
    seed_url: str = SEED_URL
    max_pages: int = 80
    max_depth: int = 2
    delay_seconds: float = 0.25
    timeout_seconds: int = 20
    include_oia_pages: bool = False


@dataclass
class CrawlState:
    documents: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    external_resources: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def normalize_url(href: str, base_url: str) -> str | None:
    if not href:
        return None

    href = href.strip()
    if href in {"#", "javascript:void(0);", "javascript:void(0)", "javascript:;"}:
        return None

    board_match = re.search(r"fnGoBoardDetail\('([^']+)'\s*,\s*'([^']+)'\)", href)
    if board_match:
        path, bor_key = board_match.groups()
        return urljoin(BASE_URL, f"{path}?borKey={quote(bor_key)}")

    menu_match = re.search(r"fnGoMoveMenu\('([^']+)'\s*,\s*'([^']*)'\)", href)
    if menu_match:
        return urljoin(BASE_URL, menu_match.group(1))

    if href.startswith("mailto:") or href.startswith("tel:"):
        return href

    if href.startswith("//"):
        href = "https:" + href

    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if not parsed.scheme:
        return None

    clean = parsed._replace(fragment="")
    return urlunparse(clean)


def is_internal_oia_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "oia.gachon.ac.kr"


def is_crawlable_internal_page(url: str, include_oia_pages: bool) -> bool:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if parsed.scheme not in {"http", "https"}:
        return False
    if not is_internal_oia_url(url):
        return False
    if suffix in DOCUMENT_EXTENSIONS or suffix in SKIP_EXTENSIONS:
        return False
    if "/files/" in parsed.path:
        return False
    if include_oia_pages:
        return True
    return (
        "/international/" in parsed.path
        or parsed.path.endswith("/internationalMain.do")
        or parsed.path.endswith("/clientMain/a/t/internationalMain.do")
    )


def classify_link(url: str) -> str:
    if url.startswith("mailto:"):
        return "email"
    if url.startswith("tel:"):
        return "phone"
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in DOCUMENT_EXTENSIONS:
        return "attachment"
    if is_internal_oia_url(url):
        return "internal"
    return "external"


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def is_useful_comment(text: str) -> bool:
    text = clean_text(text)
    if not text or text.startswith("[pp]") or text.startswith("pp"):
        return False
    if any(token in text.lower() for token in ["endif", "if lt ie", "//", "script"]):
        return False
    return bool(re.search(r"[가-힣A-Za-z0-9]", text))


def materialize_i18n_comments(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        text = clean_text(str(comment))
        if is_useful_comment(text):
            comment.replace_with(f" {text} ")
        else:
            comment.extract()


def extract_links(soup: BeautifulSoup, page_url: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for index, tag in enumerate(soup.find_all("a", href=True)):
        url = normalize_url(tag.get("href", ""), page_url)
        text = clean_text(tag.get_text(" ", strip=True))
        if not url:
            continue
        links.append(
            {
                "link_id": stable_id("link", f"{page_url}|{index}|{url}|{text}"),
                "source_url": page_url,
                "anchor_text": text or url,
                "target_url": url,
                "target_type": classify_link(url),
                "opens_new_window": tag.get("target") == "_blank",
            }
        )

    offset = len(links)
    for index, tag in enumerate(soup.find_all(attrs={"onclick": True}), start=offset):
        onclick = tag.get("onclick", "")
        match = re.search(r"window\.open\(['\"]([^'\"]+)['\"]", onclick)
        if not match:
            continue
        url = normalize_url(match.group(1), page_url)
        if not url:
            continue
        text = clean_text(tag.get_text(" ", strip=True)) or Path(urlparse(url).path).name or url
        links.append(
            {
                "link_id": stable_id("link", f"{page_url}|onclick|{index}|{url}|{text}"),
                "source_url": page_url,
                "anchor_text": text,
                "target_url": url,
                "target_type": classify_link(url),
                "opens_new_window": True,
            }
        )
    return links


def extract_sections(soup: BeautifulSoup) -> list[dict[str, Any]]:
    main = soup.find("main") or soup.body or soup
    sections: list[dict[str, Any]] = []

    candidates = main.find_all(["section", "article", "div"], recursive=True)
    for index, candidate in enumerate(candidates):
        if candidate.find_parent(["section", "article"]) and candidate.name == "div":
            continue
        text = clean_text(candidate.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        heading_tag = candidate.find(["h1", "h2", "h3", "h4", "strong"])
        heading = clean_text(heading_tag.get_text(" ", strip=True)) if heading_tag else ""
        sections.append(
            {
                "section_id": "",
                "title": heading or f"section-{index + 1}",
                "text": text,
                "text_length": len(text),
            }
        )

    if not sections:
        text = clean_text(main.get_text(" ", strip=True))
        if text:
            sections.append({"section_id": "", "title": "main", "text": text, "text_length": len(text)})

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section in sections:
        fingerprint = section["text"][:300]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(section)
    return deduped


def extract_tables(soup: BeautifulSoup) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for table_index, table in enumerate(soup.find_all("table")):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append({"table_index": table_index, "rows": rows})
    return tables


def extract_menu_items(soup: BeautifulSoup, page_url: str) -> list[dict[str, Any]]:
    menu_items: list[dict[str, Any]] = []

    def walk_li(li_tag: Any, depth: int, parent_text: str, source: str) -> None:
        direct_link = li_tag.find("a", recursive=False)
        if direct_link:
            text = clean_text(direct_link.get_text(" ", strip=True))
            url = normalize_url(direct_link.get("href", ""), page_url) if direct_link.has_attr("href") else None
            if text:
                menu_items.append(
                    {
                        "menu_id": stable_id("menu", f"{source}|{parent_text}|{depth}|{text}|{url or ''}"),
                        "source": source,
                        "text": text,
                        "url": url,
                        "depth": depth,
                        "parent_text": parent_text,
                    }
                )
                parent_text = text

        for child_ul in li_tag.find_all("ul", recursive=False):
            for child_li in child_ul.find_all("li", recursive=False):
                walk_li(child_li, depth + 1, parent_text, source)

    for source, selector in [("global_navigation", ".top_menu"), ("side_navigation", ".snb_list")]:
        for root in soup.select(selector):
            for li_tag in root.find_all("li", recursive=False):
                walk_li(li_tag, 1, "", source)

    unique: dict[str, dict[str, Any]] = {}
    for item in menu_items:
        unique[item["menu_id"]] = item
    return list(unique.values())


def extract_notices(soup: BeautifulSoup, page_url: str) -> list[dict[str, Any]]:
    notices: list[dict[str, Any]] = []
    for block in soup.select(".mainnotice_wrap, .board_view, .notice_view, .bbs_view"):
        title_tag = block.find(["h3", "h4", "strong", "p"])
        title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""
        text = clean_text(block.get_text(" ", strip=True))
        if len(text) < 40:
            continue
        dates = extract_dates(text)
        notices.append(
            {
                "notice_id": stable_id("notice", f"{page_url}|{title}|{text[:100]}"),
                "source_url": page_url,
                "title": title,
                "text": text,
                "dates": dates,
            }
        )
    return notices


def extract_dates(text: str) -> list[str]:
    patterns = [
        r"\b20\d{2}[./-]\s?\d{1,2}[./-]\s?\d{1,2}\b",
        r"\b20\d{2}\.\d{1,2}\.\d{1,2}\.\([^)]+\)",
        r"\b\d{1,2}월\s*\d{1,2}일\b",
    ]
    dates: list[str] = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    return sorted(set(clean_text(date) for date in dates))


def extract_entities(text: str) -> list[dict[str, str]]:
    specs = [
        ("Email", r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
        ("Phone", r"(?:\+82[-\s]?)?0?\d{1,2}[-\s]\d{3,4}[-\s]\d{4}"),
        ("Fee", r"\d{1,3}(?:,\d{3})*\s?원"),
        ("TopikLevel", r"TOPIK\s?[ⅠⅡI]{1,2}|TOPIK\s?\d급|토픽\s?\d급"),
        ("Location", r"(?:글로벌센터|가천관|비전타워|AI공학관|글로벌캠퍼스)[^\s,)]*"),
    ]
    entities: list[dict[str, str]] = []
    for entity_type, pattern in specs:
        for value in re.findall(pattern, text, flags=re.IGNORECASE):
            value = clean_text(value)
            if value:
                entities.append({"type": entity_type, "value": value})
    for date in extract_dates(text):
        entities.append({"type": "Date", "value": date})
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for entity in entities:
        unique[(entity["type"], entity["value"])] = entity
    return list(unique.values())


def save_raw_html(url: str, html: str) -> str:
    raw_id = stable_id("page", url).replace(":", "_")
    path = RAW_DIR / f"{raw_id}.html"
    path.write_text(html, encoding="utf-8")
    return str(path.relative_to(WORKSPACE))


def fetch_page(session: requests.Session, url: str, timeout_seconds: int) -> requests.Response:
    response = session.get(url, timeout=timeout_seconds, verify=False)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return response


def parse_document(url: str, html: str, crawled_at: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    materialize_i18n_comments(soup)

    title = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    links = extract_links(soup, url)
    sections = extract_sections(soup)
    for index, section in enumerate(sections):
        section["section_id"] = stable_id("section", f"{url}|{index}|{section['title']}|{section['text'][:80]}")

    full_text = clean_text(soup.get_text(" ", strip=True))
    return {
        "page_id": stable_id("page", url),
        "url": url,
        "title": title,
        "full_text": full_text,
        "language": "ko",
        "crawled_at": crawled_at,
        "raw_html_path": save_raw_html(url, html),
        "sections": sections,
        "tables": extract_tables(soup),
        "menu_items": extract_menu_items(soup, url),
        "links": links,
        "notices": extract_notices(soup, url),
        "entities": extract_entities(full_text),
    }


def crawl(config: CrawlConfig) -> CrawlState:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for directory in (RAW_DIR, PROCESSED_DIR, KG_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    state = CrawlState()
    queue: deque[tuple[str, int]] = deque([(config.seed_url, 0)])
    visited: set[str] = set()

    while queue and len(state.documents) < config.max_pages:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            response = fetch_page(session, url, config.timeout_seconds)
        except Exception as exc:
            state.errors.append({"url": url, "error": str(exc), "depth": depth})
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "<html" not in response.text[:500].lower():
            state.attachments.append(
                {
                    "attachment_id": stable_id("attachment", url),
                    "source_url": url,
                    "content_type": content_type,
                    "size_bytes": len(response.content),
                }
            )
            continue

        crawled_at = now_iso()
        document = parse_document(url, response.text, crawled_at)
        state.documents.append(document)
        state.links.extend(document["links"])

        for link in document["links"]:
            target = link["target_url"]
            link_type = link["target_type"]
            if link_type == "attachment":
                state.attachments.append(
                    {
                        "attachment_id": stable_id("attachment", target),
                        "source_url": document["url"],
                        "target_url": target,
                        "anchor_text": link["anchor_text"],
                    }
                )
            elif link_type == "external":
                state.external_resources.append(
                    {
                        "resource_id": stable_id("external", target),
                        "source_url": document["url"],
                        "target_url": target,
                        "anchor_text": link["anchor_text"],
                    }
                )
            elif depth < config.max_depth and is_crawlable_internal_page(target, config.include_oia_pages):
                queue.append((target, depth + 1))

        if config.delay_seconds:
            time.sleep(config.delay_seconds)

    return state


def add_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    nodes[node["id"]] = {**nodes.get(node["id"], {}), **node}


def add_relation(relations: list[dict[str, Any]], source: str, rel_type: str, target: str, **props: Any) -> None:
    relations.append(
        {
            "id": stable_id("rel", f"{source}|{rel_type}|{target}|{json.dumps(props, sort_keys=True, ensure_ascii=False)}"),
            "source": source,
            "type": rel_type,
            "target": target,
            "properties": props,
        }
    )


def infer_external_resource_type(url: str, anchor_text: str) -> str:
    text = f"{url} {anchor_text}".lower()
    if "topik" in text:
        return "ExamPortal"
    if "hikorea" in text or "immigration" in text:
        return "ImmigrationPortal"
    if "studyinkorea" in text:
        return "StudyAbroadPortal"
    if "wind.gachon" in text or "forms.gle" in text:
        return "ApplicationPortal"
    if any(name in text for name in ["youtube", "instagram", "facebook", "weibo", "youku", "bilibili"]):
        return "SocialChannel"
    return "ExternalResource"


def build_kg(state: CrawlState) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    relations: list[dict[str, Any]] = []

    root_id = stable_id("site", BASE_URL)
    add_node(nodes, {"id": root_id, "type": "Site", "name": "Gachon OIA", "url": BASE_URL})

    for document in state.documents:
        page_id = document["page_id"]
        add_node(
            nodes,
            {
                "id": page_id,
                "type": "Page",
                "name": document["title"] or document["url"],
                "url": document["url"],
                "crawled_at": document["crawled_at"],
            },
        )
        add_relation(relations, root_id, "HAS_PAGE", page_id, source_url=document["url"])

        for section in document["sections"]:
            section_id = section["section_id"]
            add_node(
                nodes,
                {
                    "id": section_id,
                    "type": "Section",
                    "name": section["title"],
                    "text": section["text"][:1200],
                    "source_url": document["url"],
                },
            )
            add_relation(relations, page_id, "HAS_SECTION", section_id, evidence_text=section["text"][:300], source_url=document["url"])

        for notice in document["notices"]:
            add_node(
                nodes,
                {
                    "id": notice["notice_id"],
                    "type": "Notice",
                    "name": notice["title"] or "Notice",
                    "text": notice["text"][:1200],
                    "source_url": document["url"],
                    "dates": notice["dates"],
                },
            )
            add_relation(relations, page_id, "HAS_NOTICE", notice["notice_id"], source_url=document["url"])

        for menu_item in document.get("menu_items", []):
            menu_id = menu_item["menu_id"]
            add_node(
                nodes,
                {
                    "id": menu_id,
                    "type": "MenuItem",
                    "name": menu_item["text"],
                    "source": menu_item["source"],
                    "depth": menu_item["depth"],
                    "url": menu_item["url"],
                },
            )
            add_relation(relations, page_id, "HAS_MENU_ITEM", menu_id, source_url=document["url"])
            if menu_item["parent_text"]:
                parent_id = stable_id("menu_parent", f"{menu_item['source']}|{menu_item['parent_text']}")
                add_node(nodes, {"id": parent_id, "type": "MenuItem", "name": menu_item["parent_text"], "source": menu_item["source"]})
                add_relation(relations, parent_id, "HAS_CHILD_MENU_ITEM", menu_id, source_url=document["url"])
            if menu_item["url"] and is_internal_oia_url(menu_item["url"]):
                target_id = stable_id("page", menu_item["url"])
                add_node(nodes, {"id": target_id, "type": "Page", "name": menu_item["text"], "url": menu_item["url"]})
                add_relation(relations, menu_id, "MENU_LINKS_TO", target_id, source_url=document["url"], target_url=menu_item["url"])

        for entity in document["entities"]:
            entity_id = stable_id(entity["type"].lower(), entity["value"])
            add_node(nodes, {"id": entity_id, "type": entity["type"], "name": entity["value"]})
            rel_type = {
                "Email": "CONTACT_EMAIL",
                "Phone": "CONTACT_PHONE",
                "Date": "MENTIONS_DATE",
                "Fee": "HAS_FEE",
                "Location": "MENTIONS_LOCATION",
                "TopikLevel": "MENTIONS_EXAM_LEVEL",
            }.get(entity["type"], "MENTIONS")
            add_relation(relations, page_id, rel_type, entity_id, source_url=document["url"], evidence_text=entity["value"])

        for link in document["links"]:
            target_url = link["target_url"]
            link_type = link["target_type"]
            if link_type == "internal":
                target_id = stable_id("page", target_url)
                add_node(nodes, {"id": target_id, "type": "Page", "name": link["anchor_text"], "url": target_url})
                add_relation(
                    relations,
                    page_id,
                    "LINKS_TO",
                    target_id,
                    anchor_text=link["anchor_text"],
                    source_url=document["url"],
                    target_url=target_url,
                )
            elif link_type == "attachment":
                attachment_id = stable_id("attachment", target_url)
                add_node(nodes, {"id": attachment_id, "type": "Attachment", "name": link["anchor_text"], "url": target_url})
                add_relation(relations, page_id, "REFERENCES_ATTACHMENT", attachment_id, anchor_text=link["anchor_text"], source_url=document["url"])
            elif link_type in {"email", "phone"}:
                contact_id = stable_id(link_type, target_url)
                add_node(nodes, {"id": contact_id, "type": "ContactPoint", "name": link["anchor_text"], "url": target_url})
                add_relation(relations, page_id, "CONTACT_POINT", contact_id, anchor_text=link["anchor_text"], source_url=document["url"])
            else:
                resource_type = infer_external_resource_type(target_url, link["anchor_text"])
                resource_id = stable_id("external", target_url)
                add_node(nodes, {"id": resource_id, "type": resource_type, "name": link["anchor_text"], "url": target_url})
                rel_type = "REFERENCES_APPLICATION" if resource_type == "ApplicationPortal" else "REFERENCES_EXTERNAL_RESOURCE"
                add_relation(relations, page_id, rel_type, resource_id, anchor_text=link["anchor_text"], source_url=document["url"])

    return {"nodes": list(nodes.values()), "relations": relations}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_records(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        fingerprint = tuple(row.get(key) for key in keys)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(row)
    return deduped


def write_outputs(state: CrawlState, kg: dict[str, Any], config: CrawlConfig) -> dict[str, Any]:
    documents = state.documents
    attachments = dedupe_records(state.attachments, ("source_url", "target_url", "anchor_text", "content_type"))
    external_resources = dedupe_records(state.external_resources, ("source_url", "target_url", "anchor_text"))
    summary = {
        "seed_url": config.seed_url,
        "document_count": len(documents),
        "link_count": len(state.links),
        "attachment_count": len(attachments),
        "external_resource_count": len(external_resources),
        "error_count": len(state.errors),
        "node_count": len(kg["nodes"]),
        "relation_count": len(kg["relations"]),
        "generated_at": now_iso(),
    }

    legacy = {
        "source_url": config.seed_url,
        "title": documents[0]["title"] if documents else "",
        "full_text": documents[0]["full_text"] if documents else "",
        "extracted_links": [
            {"text": link["anchor_text"], "href": link["target_url"]}
            for link in documents[0]["links"]
        ]
        if documents
        else [],
        "documents": documents,
        "summary": summary,
    }

    write_json(WORKSPACE / "oia_scraped_data.json", legacy)
    write_json(PROCESSED_DIR / "documents.json", documents)
    write_json(PROCESSED_DIR / "links.json", state.links)
    write_json(PROCESSED_DIR / "attachments.json", attachments)
    write_json(PROCESSED_DIR / "external_resources.json", external_resources)
    write_json(PROCESSED_DIR / "crawl_errors.json", state.errors)
    write_json(KG_DIR / "kg.json", kg)
    write_json(KG_DIR / "summary.json", summary)
    write_jsonl(KG_DIR / "nodes.jsonl", kg["nodes"])
    write_jsonl(KG_DIR / "relations.jsonl", kg["relations"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Gachon OIA international pages and build KG candidates.")
    parser.add_argument("--seed-url", default=SEED_URL)
    parser.add_argument("--max-pages", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--include-oia-pages", action="store_true", help="Also crawl general OIA pages linked from the site header.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CrawlConfig(
        seed_url=args.seed_url,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay_seconds,
        timeout_seconds=args.timeout_seconds,
        include_oia_pages=args.include_oia_pages,
    )
    state = crawl(config)
    kg = build_kg(state)
    summary = write_outputs(state, kg, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
