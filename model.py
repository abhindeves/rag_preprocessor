# Add a function for just text AI

# pip install openai
import os
from openai import OpenAI
from typing import Optional, Dict, Any, List
import re

class TechnicalDocumentImageSummarizer:
    def __init__(self):
        # Set OPENAI_API_KEY in env
        self.client = OpenAI()
    
    def _build_technical_prompt(self, text_context: str = "", custom_instructions: str = "") -> str:
        """Build a comprehensive prompt for technical documentation image analysis."""
        base_prompt = """
You are analyzing an image from technical documentation. Please provide a summary on what this image depicts, focusing on technical aspects such as diagrams, charts, code snippets, or architecture illustrations or Screenshots of software interfaces.

Format your response as a structured summary that would be useful for:
- Technical search and retrieval
- Understanding the content without viewing the image
- Integration with surrounding documentation context

You should start with What the document is about ALWAYS
"""
        
        if text_context.strip():
            context_prompt = f"\n\n**Surrounding Text Context:**\n{text_context.strip()}\n\nUse this context to better understand the image's purpose and provide more relevant technical details."
            base_prompt += context_prompt
        
        if custom_instructions.strip():
            base_prompt += f"\n\n**Additional Instructions:**\n{custom_instructions.strip()}"
        
        return base_prompt

    def summarize_technical_image(
        self, 
        image_url: str, 
        text_context: str = "", 
        custom_instructions: str = "",
        model: str = "gpt-4o", 
        temperature: float = 0.1, 
        max_tokens: int = 800,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Summarizes a technical documentation image with enhanced prompting for RAG systems.

        Args:
            image_url (str): The URL of the image or a data: URI (base64) for local files.
            text_context (str): Surrounding text content from the documentation.
            custom_instructions (str): Additional specific instructions for this image.
            model (str): The OpenAI model to use (recommended: "gpt-4o" for better multimodal).
            temperature (float): Controls randomness (lower for technical accuracy).
            max_tokens (int): Maximum tokens for the summary (increased for detailed analysis).
            include_metadata (bool): Whether to include metadata about the analysis.

        Returns:
            Dict[str, Any]: Comprehensive analysis including summary, extracted elements, and metadata.
        """
        
        prompt = self._build_technical_prompt(text_context, custom_instructions)
        
        content_parts = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
        ]

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content_parts,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            summary = resp.choices[0].message.content
            
            result = {
                "summary": summary,
                "model_used": model,
                "has_context": bool(text_context.strip()),
                "context_length": len(text_context) if text_context else 0,
            }
            
            if include_metadata:
                result.update({
                    "tokens_used": resp.usage.total_tokens if resp.usage else None,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "summary": None,
                "model_used": model,
                "has_context": bool(text_context.strip())
            }

    def summarize_technical_text(
        self,
        text_content: str,
        custom_instructions: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 800,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Summarizes technical text content for RAG systems.
        
        Args:
            text_content (str): The technical text to summarize.
            custom_instructions (str): Additional instructions for the summarization.
            model (str): The OpenAI model to use.
            temperature (float): Controls randomness (lower for technical accuracy).
            max_tokens (int): Maximum tokens for the summary.
            include_metadata (bool): Whether to include metadata about the analysis.
            
        Returns:
            Dict[str, Any]: Summary and metadata.
        """
        
        base_prompt = """
You are analyzing technical documentation. Please provide a clear, structured summary of the following text.
Focus on technical accuracy and clarity.
You should start with What the document is about ALWAYS
"""
        
        if custom_instructions.strip():
            base_prompt += f"\n\n**Additional Instructions:**\n{custom_instructions.strip()}"
        
        prompt = f"{base_prompt}\n\n**Text to analyze:**\n{text_content}"

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            summary = resp.choices[0].message.content
            
            result = {
                "summary": summary,
                "model_used": model,
                "text_length": len(text_content),
            }
            
            if include_metadata:
                result.update({
                    "tokens_used": resp.usage.total_tokens if resp.usage else None,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "summary": None,
                "model_used": model
            }

    def summarize_technical_table(
        self,
        table_html: str,
        text_context: str = "",
        custom_instructions: str = "",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Analyzes an HTML table along with associated text to generate a comprehensive description
        useful for RAG systems.
        
        Args:
            table_html (str): HTML representation of the table.
            text_context (str): Surrounding text that provides context for the table.
            custom_instructions (str): Additional instructions for table analysis.
            model (str): The OpenAI model to use.
            temperature (float): Controls randomness (lower for technical accuracy).
            max_tokens (int): Maximum tokens for the description.
            include_metadata (bool): Whether to include metadata about the analysis.
            
        Returns:
            Dict[str, Any]: Table description and metadata.
        """
        
        # Clean up the HTML table for better prompting
        cleaned_table = self._clean_html_table(table_html)
        
        base_prompt = """
You are analyzing a technical table from documentation. Alwasys start with WHAT IS THE TABLE USED FOR

Format your response in a structured way that would help someone understand the table without seeing it.
"""
        
        if text_context.strip():
            base_prompt += f"\n\n**Surrounding Text Context:**\n{text_context.strip()}"
        
        if custom_instructions.strip():
            base_prompt += f"\n\n**Additional Instructions:**\n{custom_instructions.strip()}"
        
        prompt = f"{base_prompt}\n\n**HTML Table:**\n{cleaned_table}"

        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            description = resp.choices[0].message.content
            
            result = {
                "description": description,
                "model_used": model,
                "has_context": bool(text_context.strip()),
                "context_length": len(text_context) if text_context else 0,
                "table_size": self._estimate_table_size(cleaned_table)
            }
            
            if include_metadata:
                result.update({
                    "tokens_used": resp.usage.total_tokens if resp.usage else None,
                    "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                    "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                })
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "description": None,
                "model_used": model,
                "has_context": bool(text_context.strip())
            }

    def _clean_html_table(self, table_html: str) -> str:
        """
        Clean and simplify HTML table for better prompting.
        
        Args:
            table_html (str): Raw HTML table string.
            
        Returns:
            str: Cleaned table representation.
        """
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', table_html)
        
        # Basic extraction of table structure
        # Remove style attributes
        cleaned = re.sub(r' style="[^"]*"', '', cleaned)
        # Remove class attributes
        cleaned = re.sub(r' class="[^"]*"', '', cleaned)
        # Remove other common attributes but keep colspan/rowspan
        cleaned = re.sub(r' (id|width|height|border|cellpadding|cellspacing)="[^"]*"', '', cleaned)
        
        return cleaned.strip()

    def _estimate_table_size(self, table_html: str) -> Dict[str, int]:
        """
        Estimate the size of the table for metadata.
        
        Args:
            table_html (str): HTML table string.
            
        Returns:
            Dict[str, int]: Estimated row and column counts.
        """
        # Count rows
        row_matches = re.findall(r'<tr', table_html, re.IGNORECASE)
        row_count = len(row_matches)
        
        # Count columns in first row
        col_matches = re.findall(r'<(td|th)', table_html, re.IGNORECASE)
        col_count = 0
        if row_count > 0:
            # Estimate columns per row
            col_count = len(col_matches) // max(1, row_count)
        
        return {
            "estimated_rows": row_count,
            "estimated_columns": col_count
        }

    def extract_searchable_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract searchable keywords from text for RAG indexing.
        
        Args:
            text (str): The text to extract keywords from.
            max_keywords (int): Maximum number of keywords to extract.
            
        Returns:
            List[str]: List of extracted keywords.
        """
        prompt = f"""
Extract the most important technical keywords from the following text for search and retrieval purposes.
Return only a comma-separated list of {max_keywords} keywords, without any additional explanation.

Text:
{text}
"""
        
        try:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a cheaper model for this task
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.1,
                max_tokens=100,
            )
            
            keywords_text = resp.choices[0].message.content
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            return keywords[:max_keywords]
            
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []


if __name__ == "__main__":
    # Example usage for technical documentation RAG:
    summarizer = TechnicalDocumentImageSummarizer()
    
    # Example with surrounding context from documentation
    image_url = "https://example.com/technical-diagram.jpg"  # Replace with your image
    
    surrounding_text = """
    The following diagram illustrates the microservices architecture implementation. 
    This setup includes API Gateway configuration, service discovery mechanisms, 
    and load balancing strategies. Each service communicates through REST APIs 
    and shares data through the central message queue system.
    """
    
    custom_instructions = "Focus on the API endpoints and data flow patterns shown in the diagram."
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            # Single image analysis
            result = summarizer.summarize_technical_image(
                image_url=image_url,
                text_context=surrounding_text,
                custom_instructions=custom_instructions,
                model="gpt-4o",  # Use gpt-4o for better multimodal performance
                temperature=0.1,  # Lower temperature for technical accuracy
                max_tokens=800    # More tokens for detailed technical analysis
            )
            
            if result.get('summary'):
                print("=== TECHNICAL IMAGE ANALYSIS ===")
                print(result['summary'])
                print(f"\n=== METADATA ===")
                print(f"Model: {result['model_used']}")
                print(f"Tokens used: {result.get('tokens_used', 'N/A')}")
                print(f"Has context: {result['has_context']}")
                
                # Extract keywords for RAG indexing
                keywords = summarizer.extract_searchable_keywords(result['summary'])
                print(f"\n=== EXTRACTED KEYWORDS FOR RAG ===")
                print(", ".join(keywords))
            else:
                print(f"Error occurred: {result.get('error')}")
            
            # Example of text-only summarization
            technical_text = """
            Kubernetes is an open-source container orchestration platform that automates the deployment, 
            scaling, and management of containerized applications. It groups containers that make up an 
            application into logical units for easy management and discovery. Kubernetes builds upon 
            15 years of experience of running production workloads at Google, combined with best-of-breed 
            ideas and practices from the community.
            """
            
            print("\n\n=== TEXT-ONLY SUMMARY EXAMPLE ===")
            text_result = summarizer.summarize_technical_text(
                text_content=technical_text,
                custom_instructions="Focus on the core architecture concepts and benefits",
                model="gpt-4o-mini",  # Use a cheaper model for text
                max_tokens=300
            )
            
            if text_result.get('summary'):
                print(text_result['summary'])
                print(f"\nText length: {text_result['text_length']} characters")
            else:
                print(f"Error in text summarization: {text_result.get('error')}")
            
            # Example of table analysis
            table_html = """
            <table border="1">
                <tr>
                    <th>Service</th>
                    <th>CPU Allocation</th>
                    <th>Memory Allocation</th>
                    <th>Replicas</th>
                </tr>
                <tr>
                    <td>API Gateway</td>
                    <td>2 cores</td>
                    <td>4GB</td>
                    <td>3</td>
                </tr>
                <tr>
                    <td>User Service</td>
                    <td>1 core</td>
                    <td>2GB</td>
                    <td>4</td>
                </tr>
                <tr>
                    <td>Product Service</td>
                    <td>1.5 cores</td>
                    <td>3GB</td>
                    <td>3</td>
                </tr>
            </table>
            """
            
            table_context = """
            The following table shows the resource allocation for our microservices deployment.
            These values represent the minimum resources required for each service in production.
            """
            
            print("\n\n=== TABLE ANALYSIS EXAMPLE ===")
            table_result = summarizer.summarize_technical_table(
                table_html=table_html,
                text_context=table_context,
                custom_instructions="Focus on resource allocation patterns and potential bottlenecks",
                model="gpt-4o-mini",
                max_tokens=500
            )
            
            if table_result.get('description'):
                print(table_result['description'])
                print(f"\nTable size: {table_result['table_size']['estimated_rows']} rows, {table_result['table_size']['estimated_columns']} columns")
            else:
                print(f"Error in table analysis: {table_result.get('error')}")
                
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure OPENAI_API_KEY is set and the image_url is valid.")
    else:
        print("OPENAI_API_KEY environment variable not set. Please set it to use the OpenAI API.")