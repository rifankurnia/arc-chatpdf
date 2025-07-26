import os
import argparse
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> List[Document]:
    """Load a single PDF file"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return []


def process_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks


def ingest_pdfs(pdf_directory: str, persist_directory: str = "./chroma_langchain_db"):
    """
    Ingest all PDFs from a directory into Chroma vector store
    
    Args:
        pdf_directory: Path to directory containing PDF files
        persist_directory: Path where Chroma DB will be persisted
    """
    # Validate directory
    pdf_path = Path(pdf_directory)
    if not pdf_path.exists():
        raise ValueError(f"Directory {pdf_directory} does not exist")
    
    # Find all PDF files
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Load all PDFs
    all_documents = []
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        docs = load_pdf(str(pdf_file))
        # Add metadata
        for doc in docs:
            doc.metadata["source"] = pdf_file.name
            doc.metadata["file_path"] = str(pdf_file)
        all_documents.extend(docs)
    
    if not all_documents:
        logger.error("No documents were loaded")
        return
    
    logger.info(f"Loaded {len(all_documents)} pages total")
    
    # Process documents
    chunks = process_documents(all_documents)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create or update vector store
    logger.info("Creating vector store...")
    
    # Check if vector store already exists
    if os.path.exists(persist_directory):
        logger.info(f"Updating existing vector store at {persist_directory}")
        vector_store = Chroma(
            collection_name="pdf_collection",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        # Add new documents
        vector_store.add_documents(chunks)
    else:
        logger.info(f"Creating new vector store at {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="pdf_collection",
            persist_directory=persist_directory,
        )
    
    logger.info(f"Successfully ingested {len(chunks)} chunks into vector store")
    
    # Test retrieval
    test_query = "What is generative AI?"
    test_results = vector_store.similarity_search(test_query, k=3)
    logger.info(f"\nTest query: '{test_query}'")
    logger.info(f"Found {len(test_results)} relevant chunks")
    
    return vector_store


def clear_vector_store(persist_directory: str = "./chroma_langchain_db"):
    """Clear the existing vector store"""
    import shutil
    
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        logger.info(f"Cleared vector store at {persist_directory}")
    else:
        logger.info("No existing vector store to clear")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description="Ingest PDF files into Chroma vector store")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="/Users/rifan/Documents/GitHub/boilerplate-langgraph/data",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_langchain_db",
        help="Directory to persist Chroma database"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingesting"
    )
    
    args = parser.parse_args()
    
    try:
        if args.clear:
            clear_vector_store(args.persist_dir)
        
        ingest_pdfs(args.pdf_dir, args.persist_dir)
        logger.info("PDF ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise


if __name__ == "__main__":
    main()