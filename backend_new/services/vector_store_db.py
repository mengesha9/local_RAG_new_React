from langchain_community.document_loaders import  (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    UnstructuredCSVLoader,
    UnstructuredExcelLoader

)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_ollama import OllamaEmbeddings
# import ollama


from langchain_chroma import Chroma
from typing import List, Dict
from langchain_core.documents import Document
import os


from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
from langchain.document_loaders import PyMuPDFLoader
from chromadb.config import Settings
import chromadb
from services.database import get_db_connection  # Add this import
import logging
from chromadb.utils import embedding_functions
import shutil

# Load environment variables from .env file
load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

print("OPENAI_API_KEY:",OPENAI_API_KEY)

# EMBEDDING_MODEL = "nomic-embed-text"


# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings()
# embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Initialize a single, shared instance of the vectorstore
_vectorstore = None

def get_vectorstore():
    """Get or create the vectorstore as a singleton."""
    global _vectorstore
    
    try:
        if _vectorstore is not None:
            return _vectorstore
            
        # Initialize embeddings
        embeddings = OpenAIEmbeddings()
        
        # Initialize Chroma client with consistent settings
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Create vectorstore
        _vectorstore = Chroma(
            client=client,
            collection_name="documents",
            embedding_function=embeddings
        )
        
        return _vectorstore
        
    except Exception as e:
        logging.error(f"Error initializing vectorstore: {str(e)}")
        raise

def index_document_to_chroma(file_id: int, chunks: List[Dict]) -> bool:
    """Index document chunks to ChromaDB with batching."""
    try:
        logging.info(f"Indexing document {file_id} to ChromaDB")
        vectorstore = get_vectorstore()
        
        # Process in batches of 5000 chunks
        BATCH_SIZE = 5000
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            logging.info(f"Processing batch {i//BATCH_SIZE + 1} of {(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            # Prepare batch data
            texts = []
            metadatas = []
            
            for chunk in batch:
                texts.append(chunk['chunk_text'])
                metadatas.append({
                    'file_id': file_id,
                    'page_number': chunk['page_number'],
                    'x1': chunk['x1'],
                    'y1': chunk['y1'],
                    'x2': chunk['x2'],
                    'y2': chunk['y2'],
                    'width': chunk['width'],
                    'height': chunk['height']
                })
            
            # Add batch to vectorstore
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logging.info(f"Successfully indexed batch of {len(batch)} chunks")
        
        logging.info(f"Successfully indexed all {total_chunks} chunks for document {file_id}")
        return True
        
    except Exception as e:
        logging.error(f"Error indexing document to ChromaDB: {str(e)}")
        return False

def delete_doc_from_chroma(file_id: int, user_id: int):
    try:
        # Get the vectorstore instance
        vectorstore = get_vectorstore()
        
        # Fix the logical operator in the where clause
        where_clause = {"$and": [{"file_id": file_id}, {"user_id": user_id}]}
        
        # Delete documents from Chroma
        vectorstore._collection.delete(where=where_clause)
        logging.info(f"Deleted all documents with file_id {file_id}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        logging.error("Full error details:", exc_info=True)
        return False

def process_pdf(file_path: str) -> List[Dict]:
    """Process PDF file and extract text chunks with metadata."""
    try:
        logging.info(f"Processing PDF: {file_path}")
        chunks = []
        
        # Open PDF with PyMuPDF
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with their positions
            blocks = page.get_text("dict")["blocks"]
            
            for b in blocks:
                if "lines" in b:
                    current_text = []
                    current_bbox = None
                    
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text or len(text) <= 1:  # Skip empty or single-char text
                                continue
                                
                            current_text.append(text)
                            if current_bbox is None:
                                current_bbox = list(span["bbox"])
                            else:
                                current_bbox[2] = max(current_bbox[2], span["bbox"][2])
                                current_bbox[3] = max(current_bbox[3], span["bbox"][3])
                    
                    if current_text:  # Only add if we have text
                        x1, y1, x2, y2 = current_bbox
                        chunks.append({
                            "chunk_text": " ".join(current_text),
                            "page_number": page_num + 1,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "width": x2 - x1,
                            "height": y2 - y1
                        })
        
        doc.close()
        logging.info(f"Extracted {len(chunks)} text chunks from PDF")
        return chunks
        
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        raise

def clear_vectorstore():
    """Clear all data from the vector store."""
    try:
        # Get vectorstore instance
        vectorstore = get_vectorstore()
        
        # Delete all documents
        vectorstore._collection.delete(where={})
        
        # Reset the singleton instance
        global _vectorstore
        _vectorstore = None
        
        # Delete the persistent storage
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            
        logging.info("Successfully cleared vector store")
        return True
        
    except Exception as e:
        logging.error(f"Error clearing vector store: {str(e)}")
        return False
