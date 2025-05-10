import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import PyPDF2
from bertopic import BERTopic

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file
    """
    if pdf_path.suffix.lower() == ".pdf":
        text = ""
        with pdf_path.open("rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text



def clean_text(text: str) -> str:
    """
    Clean the text by removing unnecesary sections
    """
    match = re.search(r'Bibliography|Acknowledgements|Index|Contents|Carbon', text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    return text




def chunk_text(text: str, max_chunk_size: int = 2500) -> list[str]:
    """
    Chunk the text into smaller chunks of max_chunk_size
    """
    paragraphs = text.split(".\n")
    chunks=[]
    current_chunk=""
    print(f"The length of the text is {len(paragraphs)}")
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
        else:
            current_chunk += paragraph + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks




def main():
    input_folder = Path("ESG_reports/Siemens")
    document_text = []
    
    for file in input_folder.glob("*.pdf"):
        text = extract_text_from_pdf(file)
        cleaned_text = clean_text(text)
        chunks = chunk_text(cleaned_text)
        for chunk in chunks:
            document_text.append(chunk)

    print(len(document_text))
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(document_text)
    print(topic_model.get_topic_info())

if __name__ == "__main__":
    main()