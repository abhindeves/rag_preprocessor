# records_extractor.py
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import urljoin, urlparse

# ---------------------
# Helpers / ID methods
# ---------------------
def _make_doc_slug(url: str) -> str:
    """
    Deterministic short slug for a document based on netloc+path and a short hash.
    """
    parsed = urlparse(url or "")
    base = (parsed.netloc + parsed.path).rstrip('/') if parsed.netloc or parsed.path else url or "doc"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    slug = (parsed.netloc.replace(".", "_") + parsed.path.replace("/", "_") + "_" + h).strip("_")
    # sanitize multiple underscores
    slug = re.sub(r'[^A-Za-z0-9_\-]', '_', slug)
    return slug or f"doc_{h}"

def _make_chunk_id(doc_slug: str, idx: int) -> str:
    return f"{doc_slug}#chunk-{idx}"

def _make_image_id(chunk_id: str, img_idx: int) -> str:
    return f"{chunk_id}#img-{img_idx}"

def _to_absolute(url: Optional[str], base: Optional[str]) -> Optional[str]:
    if not url:
        return url
    if url.startswith(("http://", "https://", "#")):
        return url
    return urljoin(base or "", url)

def _uniq_preserve_order(seq: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for text, url in seq:
        key = (text or "", url or "")
        if key not in seen:
            seen.add(key)
            out.append((text, url))
    return out

# ---------------------
# Extraction
# ---------------------
def extract_records_from_chunks(
    chunks: List[Any],
    base_url: Optional[str] = None,
    text_preview_chars: int = 256
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract two sets of records from a list of `unstructured` Element chunks:
      - text_records: one per chunk (contains text + references to image_ids)
      - image_records: one per image with a link back to chunk_id

    Args:
      chunks: list of Element objects returned by partition_html + enhancement
      base_url: optional base URL to resolve relative links (if not provided, chunk metadata url will be used)
      text_preview_chars: length of the text preview stored in the text record

    Returns:
      (text_records, image_records)
    """
    if not chunks:
        return [], []

    # derive doc slug from base_url or first chunk metadata URL
    fallback_url = base_url or getattr(getattr(chunks[0], "metadata", None), "url", "") or ""
    doc_slug = _make_doc_slug(fallback_url)

    text_records: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(doc_slug, idx)

        # chunk text and preview
        text = getattr(chunk, "text", "") or ""
        preview = text[:text_preview_chars]

        # metadata dict safe access
        meta = getattr(chunk, "metadata", None)
        meta_dict = getattr(meta, "__dict__", {}) if meta else {}

        # source_url resolution: prefer meta url then base_url fallback
        source_url = meta_dict.get("url") or base_url or fallback_url or None

        # TITLE heuristic: best heading / first orig_element that looks like header
        title = ""
        orig_elements = meta_dict.get("orig_elements") or []
        if orig_elements:
            for e in orig_elements:
                if not isinstance(e, str):
                    continue
                # prefer short lines or things with '¶' indicating a header from unstructured output
                first_line = e.splitlines()[0].strip()
                if "¶" in e or (len(first_line) > 0 and len(first_line) <= 80):
                    title = first_line.replace("¶", "").strip()
                    break
            if not title:
                # fallback to first textual orig_element
                first_text = next((str(e) for e in orig_elements if isinstance(e, str) and e.strip()), "")
                title = first_text.splitlines()[0][:120] if first_text else ""

        # enhanced_links and quick-access lists
        elinks = meta_dict.get("enhanced_links") or {}
        # anchors
        anchors: List[Tuple[str, str]] = []
        if isinstance(elinks, dict):
            anchors = [(t or "", u or "") for t, u in elinks.get("anchors", [])]

        anchor_fragment = anchors[0][1] if anchors else None
        absolute_anchor_url = _to_absolute(anchor_fragment, source_url) if anchor_fragment else None

        # important links (version, external, internal) with type
        important_links: List[Dict[str, str]] = []
        if isinstance(elinks, dict):
            for cat in ("version_links", "external_links", "internal_links"):
                for text_link, url in elinks.get(cat, []):
                    if not url:
                        continue
                    abs_url = _to_absolute(url, source_url)
                    important_links.append({"url": abs_url, "text": text_link or "", "type": cat})

        # fallback: link_texts / link_urls
        if not important_links and meta_dict.get("link_urls"):
            ltexts = meta_dict.get("link_texts") or []
            lurls = meta_dict.get("link_urls") or []
            for t, u in zip(ltexts, lurls):
                if not u:
                    continue
                important_links.append({"url": _to_absolute(u, source_url), "text": t or "", "type": "other"})

        # images: prefer elinks['images'] then all_images/all_image_alts
        raw_images: List[Tuple[str, str]] = []
        if isinstance(elinks, dict) and elinks.get("images"):
            # elinks images expected as list of [alt, absolute_url]
            for alt, u in elinks.get("images", []):
                raw_images.append((alt or "", u or ""))
        else:
            quick_imgs = meta_dict.get("all_images") or []
            quick_alts = meta_dict.get("all_image_alts") or []
            for i, u in enumerate(quick_imgs):
                alt = quick_alts[i] if (isinstance(quick_alts, list) and i < len(quick_alts)) else ""
                raw_images.append((alt or "", u or ""))

        # normalize image urls to absolute with source_url
        raw_images = [(alt, _to_absolute(u, source_url)) for alt, u in raw_images if u]

        # dedupe images preserving order
        raw_images = _uniq_preserve_order(raw_images)

        # create image records and collect image_ids
        image_ids: List[str] = []
        for img_idx, (alt, img_url) in enumerate(raw_images):
            image_id = _make_image_id(chunk_id, img_idx)
            image_ids.append(image_id)
            image_rec = {
                "image_id": image_id,
                "chunk_id": chunk_id,
                "doc_id": doc_slug,
                "source_url": source_url,
                "image_url": img_url,
                "alt": alt or "",
                "caption": None,            # fill later (captioning / summarization)
                "thumbnail_path": None,     # fill later if caching thumbnails
                "image_type": None,         # optional heuristic (screenshot/diagram/photo)
                "image_size": None,         # (width,height) if you download
                "image_format": None,       # jpeg/png etc
                "image_embedding_id": None, # placeholder for vector DB id or embedding reference
                "summary_text": None,       # placeholder for image summary produced later
                "scraped_at": None
            }
            image_records.append(image_rec)

        # token estimate simple heuristic
        token_est = int(max(1, len(text) / 4))

        # structured tags: versions and feature flags (basic regex heuristics)
        tags = set()
        for m in re.findall(r'\b\d+\.\d+\.\d+\b', text):
            tags.add(f"version:{m}")
        for m in re.findall(r'\bnavigation\.[a-zA-Z0-9_]+\b', text):
            tags.add(m)

        text_record = {
            "chunk_id": chunk_id,
            "doc_id": doc_slug,
            "source_url": source_url,
            "anchor_fragment": anchor_fragment,
            "absolute_anchor_url": absolute_anchor_url,
            "title": title or "",
            "text": text,
            "text_preview": preview,
            "text_length": len(text),
            "token_estimate": token_est,
            "image_ids": image_ids,
            "image_count": len(image_ids),
            "important_links": important_links,
            "link_stats": meta_dict.get("link_stats", {}) or {},
            "structured_tags": sorted(tags),
            "provenance": {
                "page_title": meta_dict.get("page_name") or meta_dict.get("title") or "",
                "page_url": source_url,
                "position_in_page": idx,
                "filetype": meta_dict.get("filetype")
            },
            "language": (meta_dict.get("languages") or [None])[0],
            "scraped_at": None,
            "text_embedding_id": None  # placeholder for embedding reference / vector id
        }

        text_records.append(text_record)

    return text_records, image_records

# ---------------------
# JSONL writer
# ---------------------
def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    """
    Write a list of records (dicts) to a JSONL file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------------
# Example usage (commented)
# ---------------------
if __name__ == "__main__":
    # Example:
    from rag import EnhancedHTMLChunker
    chunker = EnhancedHTMLChunker()
    chunks = chunker.chunk_html_with_enhanced_metadata("https://squidfunk.github.io/mkdocs-material/contributing/making-a-pull-request/")

    text_records, image_records = extract_records_from_chunks(chunks, base_url="https://squidfunk.github.io/mkdocs-material/contributing/making-a-pull-request/")
    write_jsonl(text_records, "./output/text_records.jsonl")
    write_jsonl(image_records, "./output/image_records.jsonl")
