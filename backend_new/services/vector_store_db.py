from typing import List, Dict
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import logging

# Import necessary packages
import PyPDF2

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

import openai

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



def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='UTF-8')
    elif file_path.endswith('.csv'):
        loader = UnstructuredCSVLoader(file_path,mode="elements")
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        loader = UnstructuredExcelLoader(file_path,mode="elements")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)




def index_document_to_chroma(file_path: str, file_id: int,user_id:int) -> bool:
    """Index document chunks to ChromaDB with batching."""
    try:
        
        file_path = "temp_final_summary.txt"

        
        splits = load_and_split_document(file_path)

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['user_id'] = user_id
            split.metadata['source_file'] = file_path
            
        logging.info(f"Indexing document {file_id} to ChromaDB")
        
        # vectorstore = get_vectorstore()
        
        # Process in batches of 5000 chunks
        BATCH_SIZE = 5000
        # total_chunks = len(chunks)
        
        # for i in range(0, total_chunks, BATCH_SIZE):
        #     batch = chunks[i:i + BATCH_SIZE]
        #     logging.info(f"Processing batch {i//BATCH_SIZE + 1} of {(total_chunks + BATCH_SIZE - 1)//BATCH_SIZE}")
            
        #     # Prepare batch data
        #     texts = []
        #     metadatas = []
            
        #     for chunk in batch:
        #         texts.append(chunk['chunk_text'])
        #         metadatas.append({
        #             "file_id": file_id,
        #             "chunk_id": chunk["chunk_id"],
        #             "source_file": chunk["source_file"]
        #         })
            
            # Add batch to vectorstore
        _vectorstore.add_documents(splits)

        # vectorstore.add_texts(
        #     texts=texts,
        #     metadatas=metadatas
        # )
            
        
        logging.info(f"Successfully indexed all  chunks for document {file_id}")
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

# def process_pdf(file_path: str) -> List[Dict]:
#     """Process PDF file and extract text chunks with metadata."""
#     try:
#         logging.info(f"Processing PDF: {file_path}")
#         chunks = []
        
#         # Open PDF with PyMuPDF
#         doc = fitz.open(file_path)
        
#         for page_num in range(len(doc)):
#             page = doc[page_num]
            
#             # Extract text blocks with their positions
#             blocks = page.get_text("dict")["blocks"]
            
#             for b in blocks:
#                 if "lines" in b:
#                     current_text = []
#                     current_bbox = None
                    
#                     for line in b["lines"]:
#                         for span in line["spans"]:
#                             text = span["text"].strip()
#                             if not text or len(text) <= 1:  # Skip empty or single-char text
#                                 continue
                                
#                             current_text.append(text)
#                             if current_bbox is None:
#                                 current_bbox = list(span["bbox"])
#                             else:
#                                 current_bbox[2] = max(current_bbox[2], span["bbox"][2])
#                                 current_bbox[3] = max(current_bbox[3], span["bbox"][3])
                    
#                     if current_text:  # Only add if we have text
#                         x1, y1, x2, y2 = current_bbox
#                         chunks.append({
#                             "chunk_text": " ".join(current_text),
#                             "page_number": page_num + 1,
#                             "x1": x1,
#                             "y1": y1,
#                             "x2": x2,
#                             "y2": y2,
#                             "width": x2 - x1,
#                             "height": y2 - y1
#                         })
        
#         doc.close()
#         logging.info(f"Extracted {len(chunks)} text chunks from PDF")
#         logging.info(f"Chunks: {chunks}")
#         return chunks
        
#     except Exception as e:
#         logging.error(f"Error processing PDF: {str(e)}")
#         raise



def process_pdf(file_path: str) -> List[Dict]:
    """Process a PDF file using OCR and return extracted text chunks with basic metadata."""
    try:
        logging.info(f"Processing PDF with OCR: {file_path}")
        
        pdf_path = file_path
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total_pages = len(reader.pages)
        print("Total pages in PDF:", total_pages)
        
        # Define number of pages per chunk (for example, 600/10 = 60 pages each)
        chunk_pages = total_pages // 10

        # Create directory to store PDF chunks
        if not os.path.exists("chunks"):
            os.makedirs("chunks")
        
        
        # Function to split PDF into a new file given a page range
        def split_pdf(input_pdf, output_pdf, start, end):
            with open(input_pdf, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                writer = PyPDF2.PdfWriter()
                for i in range(start, end):
                    writer.add_page(reader.pages[i])
                with open(output_pdf, "wb") as f_out:
                    writer.write(f_out)
        
        chunk_files = []
        for i in range(10):
            start_page = i * chunk_pages
            # Ensure the last chunk takes any remainder pages
            end_page = (i + 1) * chunk_pages if i < 9 else total_pages
            chunk_filename = f"chunks/chunk_{i+1}.pdf"
            split_pdf(pdf_path, chunk_filename, start_page, end_page)
            chunk_files.append(chunk_filename)
            print(f"Created {chunk_filename} (pages {start_page+1} to {end_page})")
            
        # Set up Docling OCR pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions()
        pipeline_options.ocr_options.lang = ["en", "de", "es", "fr"]

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        
        api_key = OPENAI_API_KEY  # Replace this with your fresh key

        client = openai.OpenAI(api_key=api_key)

        def openai_summarize(text):
            prompt = (
                "Please provide a detailed and comprehensive summary of the following text. "
                "This should not be a brief or surface-level overview — I’m looking for a thorough, in-depth summary "
                "that captures the key ideas, reasoning, and structure of the content. "
                "The summary should be extensive — ideally at least half as long as the original — and should span multiple paragraphs (or even pages) if needed.\n\n"
                "When you provide the summary, please output only the core content without any additional sections such as an introduction, conclusion, or any other extraneous blocks.\n\n"

                f"{text}"
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            return response.choices[0].message.content

        
        

        # Convert PDF and extract markdown
        result = converter.convert(file_path)
        markdown_text = result.document.export_to_markdown()
        
    
        
        summaries = []
        for chunk_file in chunk_files:
            print(f"Processing {chunk_file}...")
            # Convert the chunk using the converter that was already initialized with pipeline_options
            result = converter.convert(chunk_file)

            # Export the processed document as markdown (or JSON if preferred)
            chunk_text = result.document.export_to_markdown()

            print(f"Extracted text from {chunk_file}. Generating summary...")
            summary = openai_summarize(chunk_text)
            summaries.append(summary)
            print(f"Summary for {chunk_file}:\n{chunk_text}\n{'-'*40}")


        
        final_summary = "\n\n".join(summaries)
        summary_filename = "temp_final_summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(final_summary)
        logging.info(f"Final Aggregated Summary saved to: {summary_filename}")

        # Basic chunking based on headers or page breaks
        chunks = []
        for i, section in enumerate(markdown_text.split("\n\n")):
            cleaned_text = section.strip()
            if cleaned_text:
                chunks.append({
                    "chunk_id": i + 1,
                    "chunk_text": cleaned_text,
                    "source_file": file_path
                })

        logging.info(f"Extracted {len(chunks)} chunks from PDF: {file_path}")
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
