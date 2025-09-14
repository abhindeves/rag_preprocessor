from rag_html_processor import RAGHTMLProcessor
import os

# Your URL and parameters
url = 'https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/'
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


