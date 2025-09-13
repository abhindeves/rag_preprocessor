from unstructured.partition.html import partition_html
from pathlib import Path
from unstructured.documents.elements import Element
import json
import re
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Tuple, Any
import requests
from PIL import Image
import io

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

    def print_enhanced_partitioned_elements(self, elements: list[Element]) -> None:
        """
        Print partitioned elements with enhanced link analysis integrated.
        """
        print(f"=== ENHANCED HTML CHUNKING WITH INTEGRATED ANALYSIS ===")
        print(f"Total partitions: {len(elements)}\n")
        
        for i, element in enumerate(elements, start=1):
            print(f"Partition {i}:")
            print(f"  Type: {type(element).__name__}")
            print(f"  Text Length: {len(element.text) if element.text else 0} characters")
            
            # Check if enhanced metadata exists
            if hasattr(element.metadata, 'enhanced_links'):
                print(f"  🔗 Enhanced Links Analysis:")
                print(f"    Total links: {element.metadata.total_links}")
                print(f"    Has images: {element.metadata.has_images}")
                print(f"    Has important links: {element.metadata.has_important_links}")
                
                # Show link statistics
                stats = element.metadata.link_stats
                if stats.get('images_count', 0) > 0:
                    print(f"    📸 Images: {stats['images_count']}")
                    for j, (alt, url) in enumerate(zip(element.metadata.all_image_alts, element.metadata.all_images)):
                        clean_alt = alt.strip() if alt.strip() else "[No alt text]"
                        print(f"      {j+1}. {clean_alt} -> {url}")
                
                if stats.get('external_links_count', 0) > 0:
                    print(f"    🌐 External links: {stats['external_links_count']}")
                
                if stats.get('internal_links_count', 0) > 0:
                    print(f"    📁 Internal links: {stats['internal_links_count']}")
                
                if stats.get('version_links_count', 0) > 0:
                    print(f"    📋 Version links: {stats['version_links_count']}")
                
                # Show important links (first few)
                if element.metadata.important_links:
                    print(f"    🎯 Important links (first 3):")
                    for j, (text, url) in enumerate(zip(
                        element.metadata.important_link_texts[:3], 
                        element.metadata.important_links[:3]
                    )):
                        clean_text = text.strip() if text.strip() else "[No text]"
                        print(f"      {j+1}. {clean_text} -> {url}")
            else:
                print(f"  📊 No enhanced link analysis (no links found)")
            
            # Show text preview
            if element.text:
                preview = element.text[:150].replace('\n', ' ')
                if len(element.text) > 150:
                    preview += "..."
                print(f"  📝 Preview: {preview}")
            
            print("-" * 80)

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
            'max_characters': 5000,
            'combine_text_under_n_chars': 1000,
            'new_after_n_chars': 3000,
        }
        
        # Update with user parameters
        default_params.update(kwargs)
        
        print(f"🚀 Starting enhanced HTML chunking for: {url}")
        print(f"📁 Output directory: {self.output_path}")
        print(f"🔗 Base URL for absolute conversion: {self.base_url}")
        
        # Partition the HTML
        chunks = partition_html(url=url, **default_params)
        
        # Enhance each chunk with link/image analysis
        enhanced_chunks = []
        for chunk in chunks:
            enhanced_chunk = self.enhance_chunk_metadata(chunk)
            enhanced_chunks.append(enhanced_chunk)
        
        # Print analysis (optional)
        self.print_enhanced_partitioned_elements(enhanced_chunks)
        
        return enhanced_chunks

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

# Convenience functions
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

# Main execution example
if __name__ == "__main__":
    # Your URL and parameters
    url = 'https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/'
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
    
    # Step 3: Verify the results
    print(f"\n🎯 RAG PIPELINE READY!")
    print(f"📦 You now have {len(chunks)} enhanced chunks")
    
    # Step 4: Show examples with ABSOLUTE URLs
    for i, chunk in enumerate(chunks):
        if hasattr(chunk.metadata, 'enhanced_links') and (chunk.metadata.has_images or chunk.metadata.has_important_links):
            print(f"\n📋 Chunk {i+1} - ABSOLUTE URLS:")
            if chunk.metadata.has_images:
                print(f"   🖼️  Images: {chunk.metadata.all_images}...")
            if chunk.metadata.has_important_links:
                print(f"   🔗 Links: {chunk.metadata.important_links}...")
            
            # Verify first image URL is absolute
            if chunk.metadata.all_images:
                first_img = chunk.metadata.all_images[0]
                is_absolute = first_img.startswith('https://') or first_img.startswith('http://')
                print(f"   ✅ First image URL is absolute: {is_absolute}")
            break
    
    # Step 5: Optional - Download and process images
    print(f"\n🖼️  Image Processing Demo:")
    image_results = download_and_process_images(chunks, max_images=3)
    print(f"   Downloaded: {image_results['total_downloaded']} images")
    print(f"   Failed: {image_results['total_failed']} images")
    
    # Step 6: Extract clean data for your RAG system
    print(f"\n📄 Converting to RAG-ready format...")
    rag_summaries = [extract_chunk_summary_for_rag(chunk) for chunk in chunks]
    
    print(f"✅ COMPLETE! You now have:")
    print(f"   • {len(chunks)} enhanced chunks with absolute URLs")
    print(f"   • {len(rag_summaries)} RAG-ready summaries")
    print(f"   • {sum(1 for s in rag_summaries if s['has_images'])} chunks with images")
    print(f"   • {sum(1 for s in rag_summaries if s['has_links'])} chunks with important links")
    
    print(f"\n💡 Usage in your RAG pipeline:")
    print(f"   chunks[i].metadata.all_images - Direct image URLs")
    print(f"   chunks[i].metadata.important_links - Direct link URLs")
    print(f"   chunks[i].text - Text content")
    print(f"   All URLs are absolute and ready to process!")
    
    