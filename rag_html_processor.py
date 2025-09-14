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
from datetime import datetime, timezone

class RAGHTMLProcessor:
    def __init__(self, output_path: str = "./output/"):
        self.output_path = output_path
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.ico'}
        self.base_url = None  # Will be set when processing

        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

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

    def extract_table_info(self, element: Element) -> Optional[Dict[str, Any]]:
        """
        Extract table information from an element if it's a table.
        """
        if not hasattr(element, 'metadata') or not element.metadata:
            return None
            
        element_type = type(element).__name__
        metadata_dict = element.metadata.__dict__
        
        # Extract table data
        table_info = {
            'text_as_html': getattr(element.metadata, 'text_as_html', None),
            'table_structure': None,
            'row_count': 0,
            'column_count': 0,
            'headers': [],
            'caption': None,
            'table_type': 'unknown'
        }
        
        # Try to parse table structure if available
        if 'text_as_html' in metadata_dict:
            html_content = getattr(element.metadata, 'text_as_html', None)
            table_info['table_structure'] = self._parse_table_structure(html_content)
            table_info['row_count'] = table_info['table_structure'].get('row_count', 0)
            table_info['column_count'] = table_info['table_structure'].get('column_count', 0)
            table_info['headers'] = table_info['table_structure'].get('headers', [])
            table_info['caption'] = table_info['table_structure'].get('caption')
            table_info['table_type'] = table_info['table_structure'].get('table_type', 'data_table')
        
        # Fallback: analyze plain text for basic structure
        else:
            return None
        
        return table_info

    def _parse_table_structure(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML table structure to extract meaningful information.
        """
        structure = {
            'row_count': 0,
            'column_count': 0,
            'headers': [],
            'caption': None,
            'table_type': 'data_table',
            'rows_data': []
        }
        
        if not html_content:
            return structure
            
        # Simple regex-based parsing (could be enhanced with BeautifulSoup)
        # Count rows
        row_matches = re.findall(r'<tr[^>]*>', html_content, re.IGNORECASE)
        structure['row_count'] = len(row_matches)
        
        # Extract headers
        header_pattern = r'<th[^>]*>(.*?)</th>'
        headers = re.findall(header_pattern, html_content, re.IGNORECASE | re.DOTALL)
        structure['headers'] = [re.sub(r'<[^>]+>', '', h).strip() for h in headers]
        structure['column_count'] = len(structure['headers']) if structure['headers'] else 0
        
        # Extract caption
        caption_match = re.search(r'<caption[^>]*>(.*?)</caption>', html_content, re.IGNORECASE | re.DOTALL)
        if caption_match:
            structure['caption'] = re.sub(r'<[^>]+>', '', caption_match.group(1)).strip()
        
        # If no headers found, try to count columns from first row
        if structure['column_count'] == 0:
            first_row = re.search(r'<tr[^>]*>(.*?)</tr>', html_content, re.IGNORECASE | re.DOTALL)
            if first_row:
                cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
                cells = re.findall(cell_pattern, first_row.group(1), re.IGNORECASE | re.DOTALL)
                structure['column_count'] = len(cells)
        
        return structure

    def _parse_text_table(self, text: str) -> Dict[str, Any]:
        """
        Parse plain text to detect table-like structure.
        """
        structure = {
            'row_count': 0,
            'column_count': 0,
            'headers': [],
            'caption': None,
            'table_type': 'text_table'
        }
        
        if not text:
            return structure
            
        lines = text.strip().split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        structure['row_count'] = len(non_empty_lines)
        
        # Try to detect column separators (|, \t, multiple spaces)
        if non_empty_lines:
            first_line = non_empty_lines[0]
            # Count potential column separators
            pipe_count = first_line.count('|')
            tab_count = first_line.count('\t')
            
            if pipe_count > 0:
                structure['column_count'] = pipe_count + 1
                # First line might be headers
                structure['headers'] = [col.strip() for col in first_line.split('|')]
            elif tab_count > 0:
                structure['column_count'] = tab_count + 1
                structure['headers'] = [col.strip() for col in first_line.split('\t')]
            else:
                # Try to detect multiple spaces as separators
                parts = re.split(r'\s{2,}', first_line)
                if len(parts) > 1:
                    structure['column_count'] = len(parts)
                    structure['headers'] = [col.strip() for col in parts]
        
        return structure

    def chunk_html(self, url: str, **kwargs) -> List[Element]:
        """
        Main chunking method that returns chunks with integrated link/image/table analysis.
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
        
        # Enhance each chunk with link/image/table analysis
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = self.enhance_chunk_metadata(chunk)
            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def enhance_chunk_metadata(self, element: Element) -> Element:
        """
        Add link, image, and table analysis directly to the element's metadata.
        This modifies the element in place and returns it.
        """
        if not element.metadata:
            return element
            
        metadata_dict = element.metadata.__dict__
        
        # Perform link analysis
        categorized_links = self.separate_links_and_images(metadata_dict)
        
        # Perform table analysis
        table_info = self.extract_table_info(element)
        
        
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
        
        # Add table information if this is a table
        if table_info:
            element.metadata.table_info = table_info
            element.metadata.is_table = True
            element.metadata.has_table = True
        else:
            element.metadata.is_table = False
            element.metadata.has_table = False
        
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
            
            # Add table data if available
            if hasattr(chunk.metadata, 'table_info'):
                enhanced_data.update({
                    'table_info': chunk.metadata.table_info,
                    'is_table': chunk.metadata.is_table,
                    'has_table': chunk.metadata.has_table,
                })
            
            chunk_data['original_metadata'] = original_metadata
            chunk_data['enhanced_metadata'] = enhanced_data
        
        return chunk_data

    # ---------------------
    # Helper functions (converted to methods)
    # ---------------------
    def _make_doc_slug(self, url: str) -> str:
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

    def _make_chunk_id(self, doc_slug: str, idx: int) -> str:
        return f"{doc_slug}#chunk-{idx}"

    def _make_image_id(self, chunk_id: str, img_idx: int) -> str:
        return f"{chunk_id}#img-{img_idx}"

    def _make_table_id(self, chunk_id: str, table_idx: int = 0) -> str:
        return f"{chunk_id}#table-{table_idx}"

    def _to_absolute(self, url: Optional[str], base: Optional[str]) -> Optional[str]:
        if not url:
            return url
        if url.startswith(("http://", "https://", "#")):
            return url
        return urljoin(base or "", url)

    def _uniq_preserve_order(self, seq: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        seen = set()
        out = []
        for text, url in seq:
            key = (text or "", url or "")
            if key not in seen:
                seen.add(key)
                out.append((text, url))
        return out

    def extract_keywords_from_text(self, text: str, max_keywords: int = 10) -> List[str]:
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

    def get_document_structure(self, document_headings: List[Dict]) -> Dict:
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

    def write_jsonl(self, records: List[Dict[str, Any]], out_path: str) -> None:
        """
        Write a list of records (dicts) to a JSONL file.
        """
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def read_jsonl(self, in_path: str) -> List[Dict[str, Any]]:
        """
        Read a list of records (dicts) from a JSONL file.
        """
        records = []
        if not os.path.exists(in_path):
            print(f"Warning: File not found at {in_path}")
            return records
        
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {line.strip()} - {e}")
        return records

    def process_table_records(self, table_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process table records to extract additional insights and statistics.
        """
        print(f"📊 Processing {len(table_records)} table records...")
        
        processed_tables = []
        table_stats = {
            'total_tables': len(table_records),
            'tables_with_headers': 0,
            'tables_with_captions': 0,
            'average_rows': 0,
            'average_columns': 0,
            'table_types': {}
        }
        
        total_rows = 0
        total_columns = 0
        
        for table in table_records:
            # Basic statistics
            row_count = table.get('row_count', 0)
            col_count = table.get('column_count', 0)
            table_type = table.get('table_type', 'unknown')
            
            total_rows += row_count
            total_columns += col_count
            
            if table.get('headers'):
                table_stats['tables_with_headers'] += 1
                
            if table.get('caption'):
                table_stats['tables_with_captions'] += 1
                
            # Count table types
            if table_type not in table_stats['table_types']:
                table_stats['table_types'][table_type] = 0
            table_stats['table_types'][table_type] += 1
            
            # Enhanced table record with additional processing
            enhanced_table = table.copy()
            
            # Generate table summary
            summary_parts = []
            if table.get('caption'):
                summary_parts.append(f"Caption: {table['caption']}")
            if table.get('headers'):
                summary_parts.append(f"Headers: {', '.join(table['headers'][:5])}")  # First 5 headers
            if row_count > 0:
                summary_parts.append(f"{row_count} rows")
            if col_count > 0:
                summary_parts.append(f"{col_count} columns")
                
            enhanced_table['summary_text'] = '; '.join(summary_parts)
            
            # Extract searchable content from table
            searchable_content = []
            if table.get('caption'):
                searchable_content.append(table['caption'])
            if table.get('headers'):
                searchable_content.extend(table['headers'])
            # Add some of the table text content
            if table.get('table_text'):
                # Take first 200 chars of table text
                searchable_content.append(table['table_text'][:200])
                
            enhanced_table['searchable_content'] = ' '.join(searchable_content)
            
            processed_tables.append(enhanced_table)
        
        # Calculate averages
        if len(table_records) > 0:
            table_stats['average_rows'] = round(total_rows / len(table_records), 2)
            table_stats['average_columns'] = round(total_columns / len(table_records), 2)
        
        return {
            'processed_tables': processed_tables,
            'statistics': table_stats
        }

    def extract_table_summary_for_rag(self, table_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a concise summary of a table record for RAG pipeline.
        """
        return {
            'table_id': table_record.get('table_id'),
            'chunk_id': table_record.get('chunk_id'),
            'row_count': table_record.get('row_count', 0),
            'column_count': table_record.get('column_count', 0),
            'has_headers': bool(table_record.get('headers')),
            'has_caption': bool(table_record.get('caption')),
            'table_type': table_record.get('table_type', 'unknown'),
            'summary': table_record.get('summary_text', ''),
            'headers': table_record.get('headers', [])[:5],  # First 5 headers only
            'searchable_content': table_record.get('searchable_content', '')
        }

    # ---------------------
    # Convenience functions (converted to methods)
    # ---------------------
    def fix_chunk_urls(self, chunks: List[Element], base_url: str) -> List[Element]:
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

    def download_images(self, chunks: List[Element], max_images: int = 5) -> Dict[str, Any]:
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

    def extract_chunk_summary_for_rag(self, chunk: Element) -> Dict[str, Any]:
        """
        Extract a concise summary of a chunk for RAG pipeline.
        """
        summary = {
            'text': chunk.text,
            'text_length': len(chunk.text) if chunk.text else 0,
            'element_type': type(chunk).__name__,
            'has_images': False,
            'has_links': False,
            'has_table': False,
            'is_table': False,
            'image_count': 0,
            'link_count': 0,
            'images': [],
            'important_links': [],
            'table_info': None
        }
        
        if hasattr(chunk.metadata, 'enhanced_links'):
            summary['has_images'] = chunk.metadata.has_images
            summary['has_links'] = chunk.metadata.has_important_links
            summary['image_count'] = len(chunk.metadata.all_images) if hasattr(chunk.metadata, 'all_images') else 0
            summary['link_count'] = chunk.metadata.total_links
            summary['images'] = chunk.metadata.all_images if hasattr(chunk.metadata, 'all_images') else []
            summary['important_links'] = chunk.metadata.important_links if hasattr(chunk.metadata, 'important_links') else []
        
        if hasattr(chunk.metadata, 'is_table'):
            summary['is_table'] = chunk.metadata.is_table
            summary['has_table'] = chunk.metadata.has_table
            if hasattr(chunk.metadata, 'table_info'):
                summary['table_info'] = chunk.metadata.table_info
        
        return summary

    def extract_records(
        self,
        chunks: List[Any],
        base_url: Optional[str] = None,
        text_preview_chars: int = 256
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract three sets of records from a list of `unstructured` Element chunks:
          - text_records: one per chunk (contains text + references to image_ids and table_ids)
          - image_records: one per image with a link back to chunk_id
          - table_records: one per table with a link back to chunk_id
        """
        scraped_at_time = datetime.now(timezone.utc).isoformat()
        if not chunks:
            return [], [], []

        # derive doc slug from base_url or first chunk metadata URL
        fallback_url = base_url or getattr(getattr(chunks[0], "metadata", None), "url", "") or ""
        doc_slug = self._make_doc_slug(fallback_url)

        text_records: List[Dict[str, Any]] = []
        image_records: List[Dict[str, Any]] = []
        table_records: List[Dict[str, Any]] = []

        # NEW: Track hierarchical structure for the entire document
        document_headings = []
        current_hierarchy = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = self._make_chunk_id(doc_slug, idx)

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
            keywords = self.extract_keywords_from_text(text)
            
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
            absolute_anchor_url = self._to_absolute(anchor_fragment, source_url) if anchor_fragment else None
            
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
                        abs_url = self._to_absolute(url, source_url)
                        important_links.append({"url": abs_url, "text": text_link or "", "type": cat})

            # fallback: link_texts / link_urls
            if not important_links and meta_dict.get("link_urls"):
                ltexts = meta_dict.get("link_texts") or []
                lurls = meta_dict.get("link_urls") or []
                for t, u in zip(ltexts, lurls):
                    if not u:
                        continue
                    important_links.append({"url": self._to_absolute(u, source_url), "text": t or "", "type": "other"})

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
            raw_images = [(alt, self._to_absolute(u, source_url)) for alt, u in raw_images if u]

            # dedupe images preserving order
            raw_images = self._uniq_preserve_order(raw_images)

            # create image records and collect image_ids
            image_ids: List[str] = []
            for img_idx, (alt, img_url) in enumerate(raw_images):
                image_id = self._make_image_id(chunk_id, img_idx)
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
                    "scraped_at": scraped_at_time
                }
                image_records.append(image_rec)

            # NEW: Handle table records
            table_ids: List[str] = []
            if meta_dict.get("is_table") or meta_dict.get("has_table"):
                table_info = meta_dict.get("table_info", {})
                table_id = self._make_table_id(chunk_id, 0)
                table_ids.append(table_id)
                
                # Extract table content
                table_html = table_info.get("text_as_html") if table_info else None
                table_structure = table_info.get("table_structure", {}) if table_info else {}
                
                table_rec = {
                    "table_id": table_id,
                    "chunk_id": chunk_id,
                    "doc_id": doc_slug,
                    "source_url": source_url,
                    "table_html": table_html,
                    "table_text": text,  # Plain text representation
                    "table_structure": table_structure,
                    "row_count": table_structure.get("row_count", 0),
                    "column_count": table_structure.get("column_count", 0),
                    "headers": table_structure.get("headers", []),
                    "caption": table_structure.get("caption"),
                    "table_type": table_structure.get("table_type", "unknown"),
                    "table_embedding_id": None,
                    "summary_text": None,
                    "scraped_at": scraped_at_time
                }
                table_records.append(table_rec)

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
                "table_ids": table_ids,  # NEW: Table IDs
                "table_count": len(table_ids),  # NEW: Table count
                "is_table": meta_dict.get("is_table", False),  # NEW: Is this chunk a table?
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
                "scraped_at": scraped_at_time,
                "text_embedding_id": None
            }

            text_records.append(text_record)

        return text_records, image_records, table_records

    def display_chunks_overview(self, text_records: List[Dict[str, Any]], 
                                image_records: List[Dict[str, Any]], 
                                table_records: List[Dict[str, Any]]) -> None:
        """
        Displays a comprehensive overview of the processed chunks, images, and tables.
        """
        print("\n" + "="*80)
        print("✨ CHUNKS OVERVIEW ✨")
        print("="*80)

        if not text_records:
            print("No text chunks to display.")
            return

        print(f"Total Text Chunks: {len(text_records)}")
        print(f"Total Image Records: {len(image_records)}")
        print(f"Total Table Records: {len(table_records)}")
        print("-" * 80)

        # Create mappings for quick lookup
        image_map = {img['chunk_id']: [] for img in image_records}
        for img in image_records:
            image_map[img['chunk_id']].append(img)

        table_map = {tbl['chunk_id']: [] for tbl in table_records}
        for tbl in table_records:
            table_map[tbl['chunk_id']].append(tbl)

        for i, chunk in enumerate(text_records):
            print(f"\nCHUNK {i+1}: {chunk['chunk_id']}")
            print(f"  Type: {chunk['element_category']} (Is Heading: {chunk['is_heading']})")
            print(f"  Hierarchy: {' -> '.join(chunk['hierarchy_path'])}")
            print(f"  Title: {chunk['title']}")
            print(f"  Text Preview: {chunk['text_preview']}...")
            print(f"  Text Length: {chunk['text_length']} chars (Est. Tokens: {chunk['token_estimate']})")
            print(f"  Source URL: {chunk['direct_url']}")
            
            if chunk['keywords']:
                print(f"  Keywords: {', '.join(chunk['keywords'])}")

            # Display images associated with this chunk
            chunk_images = image_map.get(chunk['chunk_id'], [])
            if chunk_images:
                print(f"  Images ({len(chunk_images)}):")
                for img_idx, img in enumerate(chunk_images):
                    print(f"    - Image {img_idx+1}: {img['image_url']} (Alt: '{img['alt']}')")
            
            # Display tables associated with this chunk
            chunk_tables = table_map.get(chunk['chunk_id'], [])
            if chunk_tables:
                print(f"  Tables ({len(chunk_tables)}):")
                for tbl_idx, tbl in enumerate(chunk_tables):
                    print(f"    - Table {tbl_idx+1}: Rows={tbl['row_count']}, Cols={tbl['column_count']}, Caption='{tbl['caption'] or 'N/A'}'")
            
            # Display important links
            if chunk['important_links']:
                print(f"  Important Links ({len(chunk['important_links'])}):")
                for link_idx, link in enumerate(chunk['important_links']):
                    print(f"    - Link {link_idx+1}: {link['text']} ({link['url']}) [{link['type']}]")
            
            print("-" * 40)

        print("\n" + "="*80)
        print("OVERVIEW COMPLETE")
        print("="*80)

# Main execution example
if __name__ == "__main__":
    # Your URL and parameters
    url = 'https://squidfunk.github.io/mkdocs-material/contributing/reporting-a-bug/'
    output_path = "./output/"
    
    print("COMPLETE RAG HTML CHUNKING SYSTEM")
    print()
    
    processor = RAGHTMLProcessor(output_path=output_path)

    # Step 1: Chunk HTML with enhanced metadata (including tables)
    chunks = processor.chunk_html(
        url=url,
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
    chunks = processor.fix_chunk_urls(chunks, base_url)
    
    # Step 3: Extract records (now includes tables)
    text_records, image_records, table_records = processor.extract_records(chunks, base_url=url)
    
    print(f"Extracted {len(text_records)} text records")
    print(f"Extracted {len(image_records)} image records")
    print(f"Extracted {len(table_records)} table records")
    
    # Step 4: Process table records for additional insights
    if table_records:
        table_processing_result = processor.process_table_records(table_records)
        processed_table_records = table_processing_result['processed_tables']
        table_statistics = table_processing_result['statistics']
        
        # Write processed table records
        processor.write_jsonl(processed_table_records, os.path.join(output_path, "table_records.jsonl"))
    else:
        print("No tables found in the document")
    
    # Step 5: Write all records to files
    processor.write_jsonl(text_records, os.path.join(output_path, "text_records.jsonl"))
    processor.write_jsonl(image_records, os.path.join(output_path, "image_records.jsonl"))

    # Step 6: Display a comprehensive overview of the chunks
    processor.display_chunks_overview(text_records, image_records, table_records)
