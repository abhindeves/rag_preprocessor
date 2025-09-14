import json
from typing import Dict, List, Any, Optional
import re
from urllib.parse import urlparse
import markdownify
import os
from bs4 import BeautifulSoup

from rag_html_processor import RAGHTMLProcessor
from model import TechnicalDocumentImageSummarizer


summarizer = TechnicalDocumentImageSummarizer()


class MultimodalEnhancer:
    """Enhances extracted content with multimodal summaries and structured representations."""
    
    def __init__(self, output_path: str = "./enhanced_output/"):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
    
    def enhance_image_record(self, image_record: Dict, associated_text: Optional[str] = None) -> Dict:
        """
        Enhance an image record with a multimodal summary and structured representation.
        
        Args:
            image_record: The original image record from extraction
            associated_text: Text from the parent chunk for context
        
        Returns:
            Enhanced image record with multimodal summary
        """
        enhanced = {
            "type": "multimodal_chunk",
            "content_type": "image_with_summary",
            "id": image_record.get("image_id"),
            "doc_id": image_record.get("doc_id"),
            "source_url": image_record.get("source_url"),
            "image_url": image_record.get("image_url"),
            "alt_text": image_record.get("alt"),
            "associated_text_chunk_id": image_record.get("chunk_id"),
            "multimodal_summary": self._generate_image_summary(
                image_record.get("alt"), 
                image_record.get("image_url"),
                associated_text
            ),
            "provenance": {
                "scraped_at": image_record.get("scraped_at"),
                "original_data_type": "image"
            }
        }
        
        # Keep original fields
        for key in ["caption", "thumbnail_path", "image_type", "image_size", "image_format"]:
            if key in image_record:
                enhanced[key] = image_record[key]
                
        return enhanced
    
    def _generate_image_summary(self, alt_text: str, image_url: str, context_text: Optional[str] = None) -> str:
        """
        Generate a descriptive summary for an image.
        In a real implementation, this would call a multimodal model.
        """
        
        
        result = summarizer.summarize_technical_image(
            image_url=image_url,
            text_context=context_text,
            custom_instructions="",
            model="gpt-4o",  # Use gpt-4o for better multimodal performance
            temperature=0.1,  # Lower temperature for technical accuracy
            max_tokens=800    # More tokens for detailed technical analysis
        )
        
        if result.get('summary'):
            return result['summary']
        else:
            return alt_text or "No description available"
    
    
    def enhance_table_record(self, table_record: Dict, associated_text: Optional[str] = None) -> Dict:
        """
        Enhance a table record with markdown representation and summary.
        
        Args:
            table_record: The original table record from extraction
            associated_text: Text from the parent chunk for context
        
        Returns:
            Enhanced table record with markdown and summary
        """
        # getting the html content of the table
        table_html = table_record.get("table_html", "")

        
        enhanced = {
            "type": "multimodal_chunk",
            "content_type": "table_with_summary",
            "id": table_record.get("table_id"),
            "doc_id": table_record.get("doc_id"),
            "source_url": table_record.get("source_url"),
            "table_html": table_html,  # Keep original for reference
            "associated_text_context": associated_text or "",
            "multimodal_summary": self._generate_table_summary(
                table_html, 
                table_record.get("table_structure", {}),
                associated_text
            ),
            "provenance": {
                "original_data_type": "table",
                "summary_text": table_record.get("summary_text", "")
            }
        }
        
        # Keep original structure information
        for key in ["row_count", "column_count", "headers", "caption", "table_type"]:
            if key in table_record:
                enhanced[key] = table_record[key]
                
        return enhanced

    
    def _simple_table_to_markdown(self, html_content: str) -> str:
        """
        Simple regex-based HTML table to markdown conversion.
        """
        # Remove HTML tags except table structure
        clean_html = re.sub(r'<(/)?(div|span|font)[^>]*>', '', html_content)
        
        # Replace table headers
        clean_html = re.sub(r'<th[^>]*>(.*?)</th>', r'**\1**', clean_html, flags=re.DOTALL)
        
        # Replace table cells
        clean_html = re.sub(r'<td[^>]*>(.*?)</td>', r'\1', clean_html, flags=re.DOTALL)
        
        # Replace table rows
        clean_html = re.sub(r'<tr[^>]*>(.*?)</tr>', r'\1\n', clean_html, flags=re.DOTALL)
        
        # Remove remaining HTML tags
        clean_text = re.sub(r'<[^>]+>', '', clean_html)
        
        # Format as markdown table
        lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
        if not lines:
            return ""
            
        # Simple formatting - assumes first line is header
        markdown_table = "| " + " | ".join(lines[0].split()) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(lines[0].split())) + " |\n"
        
        for line in lines[1:]:
            markdown_table += "| " + " | ".join(line.split()) + " |\n"
            
        return markdown_table
    
    def _generate_table_summary(self, table_html: str, table_structure: Dict, context_text: Optional[str] = None) -> str:
        """
        Generate a summary of the table's content and purpose.
        """

        # Analyze table structure
        row_count = table_structure.get("row_count", 0)
        col_count = table_structure.get("column_count", 0)
        headers = table_structure.get("headers", [])
        
        #Join context with the structure
        table_context = f"Table with {row_count} rows and {col_count} columns. "
        #joining the context_text with table_context
        context_text = "Table Context\n\n" + (table_context or "") + "Text Around Table\n\n" + (context_text or "")
        
        # Basic summary based on structure
        
        table_result = summarizer.summarize_technical_table(
                table_html=table_html,
                text_context=context_text,
                custom_instructions="",
                model="gpt-4o-mini",
                max_tokens=1000
            )
        if table_result.get('description'):
            summary = table_result['description']
        else:
            summary = None
            
        return summary
    
    # def enhance_text_record(self, text_record: Dict) -> Dict:
    #     """
    #     Enhance a text record with summary and key concepts.
        
    #     Args:
    #         text_record: The original text record from extraction
        
    #     Returns:
    #         Enhanced text record with summary and key concepts
    #     """
    #     enhanced = {
    #         "type": "text_chunk",
    #         "content_type": "enhanced_text",
    #         "chunk_id": text_record.get("chunk_id"),
    #         "doc_id": text_record.get("doc_id"),
    #         "source_url": text_record.get("source_url"),
    #         "title": text_record.get("title", ""),
    #         "text": text_record.get("text", ""),
    #         "text_summary": self._generate_text_summary(text_record.get("text", "")),
    #         "key_concepts": self._extract_key_concepts(text_record.get("text", "")),
    #         "keywords": text_record.get("keywords", []),
    #         "provenance": {
    #             "scraped_at": text_record.get("scraped_at"),
    #             "original_data_type": "text"
    #         }
    #     }
        
    #     # Keep important metadata
    #     for key in ["text_length", "token_estimate", "image_ids", "image_count", 
    #                "table_ids", "table_count", "important_links", "link_stats"]:
    #         if key in text_record:
    #             enhanced[key] = text_record[key]
                
    #     return enhanced
    
    def _generate_text_summary(self, text: str, max_sentences: int = 2) -> str:
        """
        Generate a concise summary of the text.
        """
        if not text:
            return ""
            
        # Simple extraction of first few sentences
        sentences = re.split(r'[.!?]', text)
        summary = ". ".join([s.strip() for s in sentences[:max_sentences] if s.strip()])
        
        if len(sentences) > max_sentences:
            summary += "..."  # Indicate truncation
            
        return summary
    
    def _extract_key_concepts(self, text: str, max_concepts: int = 5) -> List[str]:
        """
        Extract key concepts from text.
        """
        if not text:
            return []
            
        # Find capitalized phrases (potential concepts)
        concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Find technical terms (words with numbers or special patterns)
        technical_terms = re.findall(r'\b\w+\d+\w*\b', text)
        
        # Combine and deduplicate
        all_concepts = list(set(concepts + technical_terms))
        
        # Return most relevant ones (prioritize longer concepts)
        all_concepts.sort(key=len, reverse=True)
        return all_concepts[:max_concepts]
    
    def enhance_all_records(self, text_records: List[Dict], image_records: List[Dict], 
                           table_records: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Enhance all extracted records with multimodal summaries.
        
        Args:
            text_records: List of text records from extraction
            image_records: List of image records from extraction
            table_records: List of table records from extraction
        
        Returns:
            Dictionary with enhanced text, image, and table records
        """
        # Create a mapping from chunk_id to text for context
        text_by_chunk_id = {record["chunk_id"]: record["text"] for record in text_records}
        
        # Enhance image records
        enhanced_images = []
        for image_record in image_records:
            chunk_id = image_record.get("chunk_id")
            associated_text = text_by_chunk_id.get(chunk_id, "")

            enhanced_images.append(self.enhance_image_record(image_record, associated_text))
        
        # Enhance table records
        enhanced_tables = []
        for table_record in table_records:
            chunk_id = table_record.get("chunk_id")
            associated_text = text_by_chunk_id.get(chunk_id, "")
            enhanced_tables.append(self.enhance_table_record(table_record, associated_text))
        
        # Enhance text records
        # enhanced_texts = [self.enhance_text_record(record) for record in text_records]
        enhanced_texts = text_records
        
        return {
            "enhanced_text_records": enhanced_texts,
            "enhanced_image_records": enhanced_images,
            "enhanced_table_records": enhanced_tables
        }
    
    def write_enhanced_records(self, enhanced_records: Dict[str, List[Dict]]):
        """
        Write enhanced records to JSONL files.
        """
        for record_type, records in enhanced_records.items():
            filename = f"{record_type}.jsonl"
            filepath = os.path.join(self.output_path, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            print(f"✓ Wrote {len(records)} {record_type} to {filepath}")

# Integration with your existing RAGHTMLProcessor
def enhance_extraction_pipeline(processor: RAGHTMLProcessor, url: str):
    """
    Complete pipeline from URL to enhanced multimodal records.
    """
    print("Starting multimodal enhancement pipeline...")
    
    # Step 1: Extract content using existing processor
    #read back the records seperately in the same line
    text_records = processor.read_jsonl(os.path.join(output_path, "text_records.jsonl"))
    image_records = processor.read_jsonl(os.path.join(output_path, "image_records.jsonl"))
    table_records = processor.read_jsonl(os.path.join(output_path, "table_records.jsonl"))
    
    # Step 2: Enhance with multimodal summaries
    enhancer = MultimodalEnhancer()
    enhanced_records = enhancer.enhance_all_records(text_records, image_records, table_records)
    
    # Step 3: Write enhanced records
    enhancer.write_enhanced_records(enhanced_records)
    
    print("Multimodal enhancement complete!")
    return enhanced_records

# Example usage
if __name__ == "__main__":
    
    # Initialize your processor
    processor = RAGHTMLProcessor(output_path="./output/")
    
    # URL to process
    url = "https://squidfunk.github.io/mkdocs-material/setup/setting-up-tags/"
    output_path = "./output/"
    
    
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
        
    base_url = url
    chunks = processor.fix_chunk_urls(chunks, base_url)
    
    text_records, image_records, table_records = processor.extract_records(chunks, base_url=url)
    
        
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
    processor.write_jsonl(table_records, os.path.join(output_path, "table_records.jsonl"))


    # Run the enhanced pipeline
    enhanced_records = enhance_extraction_pipeline(processor, url)    
        
    # Display sample enhanced records
    print("\nSample enhanced text record:")
    print(json.dumps(enhanced_records["enhanced_text_records"][0], indent=2))
    
    if enhanced_records["enhanced_image_records"]:
        print("\nSample enhanced image record:")
        print(json.dumps(enhanced_records["enhanced_image_records"][0], indent=2))
    
    if enhanced_records["enhanced_table_records"]:
        print("\nSample enhanced table record:")
        print(json.dumps(enhanced_records["enhanced_table_records"][0], indent=2))