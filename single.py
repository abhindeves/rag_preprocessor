from unstructured.partition.html import partition_html
from pathlib import Path
from unstructured.documents.elements import Element
import json
import re
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Tuple, Any, Optional
import requests
from PIL import Image
import io
import os
import hashlib

class EnhancedHTMLChunker:
    def __init__(self, output_path: str = "./output/"):
        self.output_path = output_path
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.ico'}
        self.base_url = None  # Will be set when processing

    def convert_to_absolute_url(self, url: str) -> str:
        """
        Convert relative URLs to absolute URLs using the base URL.
        """
        if not url or not self.base_url:
            return url
        
        # If already absolute, return as-is
        if url.startswith('http://') or url.startswith('https://'):
            return url
        
        # Keep anchors as relative
        if url.startswith('#'):
            return url
            
        # Convert relative to absolute
        return urljoin(self.base_url, url)

    def separate_links_and_images(self, metadata_dict: dict) -> Dict[str, List[Tuple[str, str]]]:
        """
        Separate images and other important links from chunked HTML metadata.
        All URLs are converted to absolute URLs.
        """
        if 'link_texts' not in metadata_dict or 'link_urls' not in metadata_dict:
            return {}
        
        link_texts = metadata_dict.get('link_texts', [])
        link_urls = metadata_dict.get('link_urls', [])
        
        # Initialize categories
        result = {
            'images': [],
            'internal_links': [],
            'external_links': [],
            'anchors': [],
            'version_links': [],
            'navigation_refs': [],
            'other_links': []
        }
        
        # Process each link
        for text, url in zip(link_texts, link_urls):
            if not url:  # Skip empty URLs
                continue
                
            # Parse the ORIGINAL URL to determine type (before conversion)
            parsed_url = urlparse(url)
            path = parsed_url.path.lower() if parsed_url.path else ''
            
            # Convert to absolute URL (except for anchors)
            if url.startswith('#'):
                absolute_url = url  # Keep anchors as relative
            else:
                absolute_url = self.convert_to_absolute_url(url)
            
            # Check if it's an image (use absolute URL in result)
            if any(path.endswith(ext) for ext in self.image_extensions):
                result['images'].append((text, absolute_url))
            
            # Check if it's an anchor link (starts with #) - keep original for anchors
            elif url.startswith('#'):
                result['anchors'].append((text, url))  # Keep anchors as relative
            
            # Check if it's a version/changelog link
            elif 'changelog' in url or re.match(r'^\d+\.\d+\.\d+$', text):
                result['version_links'].append((text, absolute_url))
            
            # Check if it's a navigation reference (internal)
            elif url.startswith('#navigation') or text in ['navigation.tabs', 'navigation.sections']:
                result['navigation_refs'].append((text, url))  # Keep navigation anchors as relative
            
            # Check if it's an internal link (relative path) - now absolute
            elif url.startswith('../') or url.startswith('./') or (not parsed_url.scheme and not url.startswith('#')):
                result['internal_links'].append((text, absolute_url))
            
            # External links (already absolute)
            elif parsed_url.scheme in ['http', 'https']:
                result['external_links'].append((text, absolute_url))
            
            # Fallback for other links
            else:
                result['other_links'].append((text, absolute_url))
        
        return result

    def chunk_html_with_enhanced_metadata(self, url: str, **kwargs) -> List[Element]:
        """
        Main chunking method that returns chunks with integrated link/image analysis.
        Perfect for RAG pipelines. All relative URLs converted to absolute.
        """
        # Store base URL for relative link conversion
        self.base_url = url
        
        # Default parameters
        default_params = {
            'image_alt_model': "to_text",
            'infer_table_structure': True,
            'strategy': "hi_res",
            'extract_image_block_types': ["Image", "Table"],
            'image_output_dir_path': self.output_path,
            'extract_image_block_to_payload': True,
            'chunking_strategy': "by_title",
            'max_characters': 2500,
            'combine_text_under_n_chars': 500,
            'new_after_n_chars': 1500,
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        # Partition the HTML
        chunks = partition_html(url=url, **default_params)
        
        # Enhance each chunk with link/image analysis
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = self.enhance_chunk_metadata(chunk)
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def enhance_chunk_metadata(self, element: Element) -> Element:
        """
        Add link and image analysis directly to the element's metadata.
        This modifies the element in place and returns it.
        """
        if not element.metadata:
            return element
            
        metadata_dict = element.metadata.__dict__
        
        # Perform link analysis
        categorized_links = self.separate_links_and_images(metadata_dict)
        
        if categorized_links:  # Only add if links were found
            # Calculate statistics
            link_stats = {}
            total_links = 0
            for category, links in categorized_links.items():
                count = len(links)
                link_stats[f'{category}_count'] = count
                total_links += count
            
            # Add enhanced data to metadata
            element.metadata.enhanced_links = categorized_links
            element.metadata.link_stats = link_stats
            element.metadata.total_links = total_links
            element.metadata.has_images = len(categorized_links.get('images', [])) > 0
            element.metadata.has_important_links = (
                len(categorized_links.get('external_links', [])) > 0 or
                len(categorized_links.get('internal_links', [])) > 0 or
                len(categorized_links.get('version_links', [])) > 0
            )
            
            # Create simplified lists for easier access in RAG pipeline
            element.metadata.all_images = [url for _, url in categorized_links.get('images', [])]
            element.metadata.all_image_alts = [text for text, _ in categorized_links.get('images', [])]
            element.metadata.important_links = []
            element.metadata.important_link_texts = []
            
            # Combine important links (excluding anchors and navigation refs)
            for category in ['external_links', 'internal_links', 'version_links']:
                for text, url in categorized_links.get(category, []):
                    element.metadata.important_links.append(url)
                    element.metadata.important_link_texts.append(text)
        
        return element

    def get_chunk_data_for_rag(self, chunk: Element) -> Dict[str, Any]:
        """
        Extract all relevant data from a chunk for RAG pipeline usage.
        This gives you a clean dictionary with all the enhanced data.
        """
        chunk_data = {
            'text': chunk.text,
            'element_type': type(chunk).__name__,
            'text_length': len(chunk.text) if chunk.text else 0,
        }
        
        # Add original metadata
        if chunk.metadata:
            original_metadata = chunk.metadata.__dict__.copy()
            
            # Extract enhanced data separately
            enhanced_data = {}
            if hasattr(chunk.metadata, 'enhanced_links'):
                enhanced_data = {
                    'enhanced_links': chunk.metadata.enhanced_links,
                    'link_stats': chunk.metadata.link_stats,
                    'total_links': chunk.metadata.total_links,
                    'has_images': chunk.metadata.has_images,
                    'has_important_links': chunk.metadata.has_important_links,
                    'all_images': chunk.metadata.all_images,
                    'all_image_alts': chunk.metadata.all_image_alts,
                    'important_links': chunk.metadata.important_links,
                    'important_link_texts': chunk.metadata.important_link_texts,
                }
            
            chunk_data['original_metadata'] = original_metadata
            chunk_data['enhanced_metadata'] = enhanced_data
        
        return chunk_data

# ---------------------
# Helper functions (moved from Code 2)
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
# Convenience functions
# ---------------------
def chunk_html_for_rag(url: str, output_path: str = "./output/", **kwargs) -> List[Element]:
    """
    Convenience function that returns ready-to-use chunks for RAG pipeline.
    Each chunk contains integrated link and image analysis in its metadata.
    All relative URLs are converted to absolute URLs for direct processing.
    """
    chunker = EnhancedHTMLChunker(output_path=output_path)
    return chunker.chunk_html_with_enhanced_metadata(url, **kwargs)

def extract_rag_data_from_chunks(chunks: List[Element]) -> List[Dict[str, Any]]:
    """
    Convert enhanced chunks to clean dictionaries for RAG pipeline.
    """
    chunker = EnhancedHTMLChunker()  # Just for the method
    return [chunker.get_chunk_data_for_rag(chunk) for chunk in chunks]

def fix_chunk_urls(chunks: List[Element], base_url: str) -> List[Element]:
    """
    Fix relative URLs in existing chunks - use this if URLs are still relative.
    """
    print(f"🔧 Fixing URLs with base: {base_url}")
    
    for chunk in chunks:
        if hasattr(chunk.metadata, 'enhanced_links'):
            # Fix enhanced_links
            for category, links in chunk.metadata.enhanced_links.items():
                if category in ['anchors', 'navigation_refs']:
                    continue  # Keep these as relative
                
                fixed_links = []
                for text, url in links:
                    if url.startswith('#'):
                        fixed_url = url  # Keep anchors
                    elif url.startswith('http'):
                        fixed_url = url  # Already absolute
                    else:
                        fixed_url = urljoin(base_url, url)
                    fixed_links.append((text, fixed_url))
                
                chunk.metadata.enhanced_links[category] = fixed_links
            
            # Fix quick access lists
            if hasattr(chunk.metadata, 'all_images'):
                chunk.metadata.all_images = [
                    urljoin(base_url, url) if not url.startswith('http') and not url.startswith('#') else url 
                    for url in chunk.metadata.all_images
                ]
            
            if hasattr(chunk.metadata, 'important_links'):
                fixed_important = []
                for url in chunk.metadata.important_links:
                    if url.startswith('#'):
                        fixed_important.append(url)
                    elif url.startswith('http'):
                        fixed_important.append(url)
                    else:
                        fixed_important.append(urljoin(base_url, url))
                chunk.metadata.important_links = fixed_important
    
    print("✅ URLs fixed!")
    return chunks

def download_and_process_images(chunks: List[Element], max_images: int = 5) -> Dict[str, Any]:
    """
    Download and process images from chunks for RAG pipeline.
    Returns information about downloaded images.
    """
    print(f"🖼️  Downloading up to {max_images} images...")
    
    downloaded_images = []
    failed_downloads = []
    total_processed = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        if total_processed >= max_images:
            break
            
        if hasattr(chunk.metadata, 'all_images') and chunk.metadata.all_images:
            for img_idx, img_url in enumerate(chunk.metadata.all_images):
                if total_processed >= max_images:
                    break
                
                try:
                    print(f"  Downloading: {img_url}")
                    response = requests.get(img_url, timeout=10)
                    
                    if response.status_code == 200:
                        image = Image.open(io.BytesIO(response.content))
                        
                        image_info = {
                            'url': img_url,
                            'chunk_index': chunk_idx,
                            'image_index': img_idx,
                            'size': image.size,
                            'format': image.format,
                            'mode': image.mode,
                            'alt_text': chunk.metadata.all_image_alts[img_idx] if img_idx < len(chunk.metadata.all_image_alts) else "",
                            'image_object': image  # You can save this or extract features
                        }
                        
                        downloaded_images.append(image_info)
                        print(f"    ✅ Success: {image.size} {image.format}")
                        total_processed += 1
                    else:
                        failed_downloads.append({'url': img_url, 'status': response.status_code})
                        print(f"    ❌ Failed: {response.status_code}")
                        
                except Exception as e:
                    failed_downloads.append({'url': img_url, 'error': str(e)})
                    print(f"    ❌ Error: {e}")
    
    return {
        'downloaded_images': downloaded_images,
        'failed_downloads': failed_downloads,
        'total_downloaded': len(downloaded_images),
        'total_failed': len(failed_downloads)
    }

def extract_chunk_summary_for_rag(chunk: Element) -> Dict[str, Any]:
    """
    Extract a concise summary of a chunk for RAG pipeline.
    """
    summary = {
        'text': chunk.text,
        'text_length': len(chunk.text) if chunk.text else 0,
        'element_type': type(chunk).__name__,
        'has_images': False,
        'has_links': False,
        'image_count': 0,
        'link_count': 0,
        'images': [],
        'important_links': []
    }
    
    if hasattr(chunk.metadata, 'enhanced_links'):
        summary['has_images'] = chunk.metadata.has_images
        summary['has_links'] = chunk.metadata.has_important_links
        summary['image_count'] = len(chunk.metadata.all_images) if hasattr(chunk.metadata, 'all_images') else 0
        summary['link_count'] = chunk.metadata.total_links
        summary['images'] = chunk.metadata.all_images if hasattr(chunk.metadata, 'all_images') else []
        summary['important_links'] = chunk.metadata.important_links if hasattr(chunk.metadata, 'important_links') else []
    
    return summary

# ---------------------
# Records extraction (from Code 2)
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
    """
    if not chunks:
        return [], []

    # derive doc slug from base_url or first chunk metadata URL
    fallback_url = base_url or getattr(getattr(chunks[0], "metadata", None), "url", "") or ""
    doc_slug = _make_doc_slug(fallback_url)

    text_records: List[Dict[str, Any]] = []
    image_records: List[Dict[str, Any]] = []

    # NEW: Track hierarchical structure for the entire document
    document_headings = []
    current_hierarchy = []
    
    for idx, chunk in enumerate(chunks):
        chunk_id = _make_chunk_id(doc_slug, idx)

        # chunk text and preview
        text = getattr(chunk, "text", "") or ""
        preview = text[:text_preview_chars]

        # metadata dict safe access
        meta = getattr(chunk, "metadata", None)
        meta_dict = getattr(meta, "__dict__", {}) if meta else {}

        # NEW: Extract hierarchical information from metadata
        hierarchy_level = meta_dict.get("header_level") or 0
        element_category = meta_dict.get("category") or "Content"
        
        # NEW: Detect if this is a heading element
        is_heading = hierarchy_level > 0 or "Title" in str(type(chunk)) or "Header" in str(type(chunk))
        
        # NEW: Update hierarchical structure tracking
        if is_heading and hierarchy_level > 0:
            # Update current hierarchy based on heading level
            if hierarchy_level <= len(current_hierarchy):
                current_hierarchy = current_hierarchy[:hierarchy_level - 1]
            current_hierarchy.append(text.strip())
            
            # Store for document structure
            document_headings.append({
                "level": hierarchy_level,
                "text": text.strip(),
                "chunk_id": chunk_id,
                "hierarchy": current_hierarchy.copy()
            })

        # NEW: Extract keywords/headings from chunk text
        keywords = extract_keywords_from_text(text)
        
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
        
        # Create direct URL that combines source URL with anchor fragment
        direct_url = None
        if source_url and anchor_fragment:
            # Remove any existing fragment from source_url and add the anchor fragment
            source_without_fragment = source_url.split('#')[0]
            direct_url = f"{source_without_fragment}{anchor_fragment}"
        elif absolute_anchor_url and absolute_anchor_url.startswith('http'):
            # If absolute_anchor_url is already a full URL, use it directly
            direct_url = absolute_anchor_url
        elif source_url:
            # Fallback to just the source URL if no anchor is available
            direct_url = source_url

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
                "caption": None,
                "thumbnail_path": None,
                "image_type": None,
                "image_size": None,
                "image_format": None,
                "image_embedding_id": None,
                "summary_text": None,
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

        # NEW: Get current hierarchy for this chunk
        current_chunk_hierarchy = current_hierarchy.copy() if current_hierarchy else ["Document Root"]
        
        # NEW: Determine parent chunk if available
        parent_chunk_id = None
        if len(current_chunk_hierarchy) > 1:
            # Find the most recent heading at one level higher
            for heading in reversed(document_headings):
                if heading["level"] == hierarchy_level - 1:
                    parent_chunk_id = heading["chunk_id"]
                    break
                
        # # Get Document Structure from Document Headings
        # document_structure = get_document_structure(document_headings)
        # with open("./output/document_structure.json", "w") as f:
        #     json.dump(document_structure, f, indent=2)

        text_record = {
            "chunk_id": chunk_id,
            "doc_id": doc_slug,
            "source_url": source_url,
            "anchor_fragment": anchor_fragment,
            "absolute_anchor_url": absolute_anchor_url,
            "direct_url": direct_url,
            
            # NEW: Hierarchical structure
            "hierarchy_level": hierarchy_level,
            "hierarchy_path": current_chunk_hierarchy,
            "parent_chunk_id": parent_chunk_id,
            "is_heading": is_heading,
            "element_category": element_category,
            
            # NEW: Keywords and content analysis
            "keywords": keywords,
            
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
            "text_embedding_id": None
        }

        text_records.append(text_record)

    return text_records, image_records

def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract potential keywords/headings from chunk text.
    This identifies text that looks like section headings or important terms.
    """
    if not text:
        return []
    
    keywords = set()
    
    # Look for heading patterns (lines ending with ¶ or containing specific patterns)
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Heading patterns
        if ('¶' in line or 
            line.isupper() or 
            (len(line) < 100 and line.endswith(':')) or
            re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line)):
            # Clean the heading
            clean_line = line.replace('¶', '').strip()
            if clean_line and len(clean_line) > 2:
                keywords.add(clean_line)
    
    # Extract version numbers and technical terms
    version_patterns = re.findall(r'\b\d+\.\d+(?:\.\d+)?\b', text)
    technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)  # CamelCase terms
    
    keywords.update(version_patterns)
    keywords.update(technical_terms)
    
    # Limit the number of keywords
    return list(keywords)[:max_keywords]


def get_document_structure(document_headings: List[Dict]) -> Dict:
    """
    Generate a hierarchical structure of the entire document.
    """
    structure = {"sections": []}
    current_level = 0
    stack = [structure]
    
    for heading in document_headings:
        level = heading["level"]
        section = {
            "title": heading["text"],
            "chunk_id": heading["chunk_id"],
            "level": level,
            "subsections": []
        }
        
        # Find the right parent level
        while len(stack) > level:
            stack.pop()
        
        # Add to current parent
        if len(stack) <= level:
            stack.append({"subsections": []})
        
        parent = stack[-1]
        if "subsections" not in parent:
            parent["subsections"] = []
        parent["subsections"].append(section)
        
        # Push new section to stack
        stack.append(section)
    
    return structure



def write_jsonl(records: List[Dict[str, Any]], out_path: str) -> None:
    """
    Write a list of records (dicts) to a JSONL file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Main execution example
if __name__ == "__main__":
    # Your URL and parameters
    url = 'https://squidfunk.github.io/mkdocs-material/contributing/requesting-a-change/'
    output_path = "./output/"
    
    print("🚀 COMPLETE RAG HTML CHUNKING SYSTEM")
    print("="*80)
    
    # Step 1: Chunk HTML with enhanced metadata
    chunks = chunk_html_for_rag(
        url=url,
        output_path=output_path,
        image_alt_model="to_text",
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image","Table"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=5000,
        combine_text_under_n_chars=1000,
        new_after_n_chars=3000,
    )
    
    # Step 2: Fix URLs if they're still relative (backup solution)
    base_url = url
    chunks = fix_chunk_urls(chunks, base_url)
    

    # Example of using the records extraction
    text_records, image_records = extract_records_from_chunks(chunks, base_url=url)

    
    write_jsonl(text_records, "./output/text_records.jsonl")
    write_jsonl(image_records, "./output/image_records.jsonl")