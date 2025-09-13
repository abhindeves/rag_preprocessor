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

def test_partition_html(url: str) -> None:
    print(f"Reading HTML from: {url}\n")
    elements = partition_html(url=url)
    print_partitioned_elements(elements)

if __name__ == "__main__":
    url = "https://squidfunk.github.io/mkdocs-material/getting-started/"
    test_partition_html(url)
