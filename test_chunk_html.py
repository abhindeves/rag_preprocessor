from unstructured.partition.html import partition_html
from pathlib import Path
from unstructured.documents.elements import Element
import json



def print_partitioned_elements(elements: list[Element]) -> None:
    print(f"Total partitions: {len(elements)}\n")
    for i, element in enumerate(elements, start=1):
        print(f"Partition {i}:")
        print(f"  Type: {type(element).__name__}")
        print(f"  Text: {element.text}")  # Print the full text
        
        # Print metadata in a structured way
        if element.metadata:
            metadata_dict = element.metadata.__dict__
            print("  Metadata:")
            print(json.dumps(metadata_dict, indent=4, default=str))  # Pretty print as JSON
        else:
            print("  Metadata: None")
        
        print("-" * 60)
        
        
output_path = "./output/"
url = 'https://squidfunk.github.io/mkdocs-material/setup/setting-up-social-cards/'

# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_html(
    url=url,
    image_alt_model="to_text",
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables

    extract_image_block_types=["Image","Table"],   # Add 'Table' to list to extract image of tables
    image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=5000,                  # defaults to 500
    combine_text_under_n_chars=1000,       # defaults to 0
    new_after_n_chars=3000,

    # extract_images_in_pdf=True,          # deprecated
)

print_partitioned_elements(chunks)  
