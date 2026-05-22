import os
import glob
import logging
import re
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_text(text: str) -> str:
    """Preprocess text by removing extra spaces and repeated newlines."""
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def load_pdfs_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all PDF files from a given directory, extract text, and return a list of dictionaries.
    
    Args:
        directory_path (str): The path to the directory containing PDF files.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing document ID, source, text, and metadata.
    """
    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist.")
        return []

    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {directory_path}.")
        return []

    documents = []
    
    for doc_idx, file_path in enumerate(pdf_files, start=1):
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}...")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            full_text = ""
            for page in pages:
                page_text = page.page_content.strip()
                if not page_text:
                    continue
                
                full_text += page_text + "\n"
            
            cleaned_text = preprocess_text(full_text)
            
            if cleaned_text:
                doc_id = f"doc_{doc_idx}"
                documents.append({
                    "doc_id": doc_id,
                    "source": filename,
                    "text": cleaned_text,
                    "metadata": {
                        "num_pages_extracted": len(pages),
                        "file_path": file_path
                    }
                })
            else:
                logger.warning(f"No valid text found in {filename}.")
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    logger.info(f"Successfully loaded {len(documents)} documents.")
    return documents

if __name__ == "__main__":
    docs = load_pdfs_from_directory("../data/raw_docs")
    if docs:
        print(f"Sample output: {docs[0]}")
