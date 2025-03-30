from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Response, status
from models.pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest, User, ChatNameUpdate, HighlightResponse, ChatRequest, DocumentHighlightsRequest, DocumentHighlightsResponse, ChatResponse
from fastapi.security import OAuth2PasswordBearer
from models.user import UserRegister
from services.langchain_utils import get_rag_chain, get_source_chunks
from services.database import (
      insert_application_logs,
      get_chat_history, get_all_documents, 
      insert_document_record, 
      delete_document_record,
      get_user_by_email,
      insert_user,
      get_user_chat_history,
      delete_chat_session,
      reset_password_db,
      store_pdf,
      get_pdf,
      get_db_connection,
      create_pdf_store,
      create_application_logs,
      get_all_user_pdfs,
      delete_pdf,
      get_user_by_id,
      update_chat_name,
      get_pdf_highlights,
      store_highlight,
      get_document_highlights,
      create_or_get_chat,
      store_chat_message,
      print_database_info,
      clear_all_data_except_users,
)
from services.vector_store_db import (
    index_document_to_chroma, 
    delete_doc_from_chroma,
    process_pdf,
    clear_vectorstore
)
from services.auth import decode_token, hash_password, create_access_token,verify_password, oauth2_scheme
import os
import uuid
import logging
import shutil
from typing import Dict, List
from PIL import Image
import pytesseract
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import sys
from datetime import datetime
import openai
from pydantic import BaseModel

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/5/tessdata"

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with full access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Update OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Document formats
allowed_extensions = [
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.csv', '.txt',
    '.jpeg', '.jpg', '.png',
    '.html'
]

# Initialize database tables
@app.on_event("startup")
async def startup_event():
    try:
        # Just create necessary tables
        create_pdf_store()
        create_application_logs()
        logging.info("Database tables created successfully")
    except Exception as e:
        logging.error(f"Error initializing database: {str(e)}")
        raise

@app.get("/chat-history")
async def get_chat_history_endpoint(user_id: int):
    return get_user_chat_history(user_id)

# Add this class for request validation
class ChatRequest(BaseModel):
    question: str
    session_id: str
    model: str
    user_id: int

# Update get_current_user dependency
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        user_id = decode_token(token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = get_user_by_id(int(user_id))
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
            
        return User(id=user["id"], email=user["email"])
        
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Update chat endpoint
@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    try:
        # Validate user_id matches the authenticated user
        if current_user.id != request.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User ID mismatch"
            )
        
        logging.info(f"Starting chat request for user {current_user.id}")
        print(request)
        # Create or get chat session
        chat_data = create_or_get_chat(
            session_id=request.session_id,
            user_id=request.user_id,
            model=str(request.model),
            name=None  # Make it None by default
        )
        
        # Get chat history for context
        chat_history = get_chat_history(
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Initialize RAG chain
        rag_chain = get_rag_chain(model_name=request.model)
        
        # Log the documents being processed
        logging.info(f"Processing chat for session {request.session_id}")
        
        response = rag_chain.invoke({
            "question": request.question,
            "chat_history": chat_history
        })
        
        # Extract source documents and log them
        source_docs = response.get("source_documents", [])
        logging.info(f"Retrieved {len(source_docs)} source documents")
        
        # Store highlights and collect document references
        documents = {}  # pdf_id -> list of highlight_ids
        for doc in source_docs:
            pdf_id = doc.metadata.get('file_id')
            logging.info(f"Processing source document with PDF ID: {pdf_id}")
            
            if not pdf_id:
                logging.warning("Source document missing file_id in metadata")
                continue
            
            # Verify PDF exists before creating highlight
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM pdf_store WHERE id = ?', (pdf_id,))
            if not cursor.fetchone():
                logging.warning(f"PDF with ID {pdf_id} not found in database")
                continue
            
            # Get the filename from the database
            cursor.execute('SELECT filename FROM pdf_store WHERE id = ?', (pdf_id,))
            pdf_data = cursor.fetchone()
            filename = pdf_data['filename'] if pdf_data else 'unknown.pdf'
            
            if pdf_id not in documents:
                documents[pdf_id] = []
            
            # Create highlight with logging
            highlight_data = {
                "highlight_id": str(uuid.uuid4()),
                "chat_id": request.session_id,
                "pdf_id": pdf_id,
                "content_text": doc.page_content.strip(),
                "position": {
                    "boundingRect": {
                        "x1": float(doc.metadata.get('x1', 0)),
                        "y1": float(doc.metadata.get('y1', 0)),
                        "x2": float(doc.metadata.get('x2', 0)),
                        "y2": float(doc.metadata.get('y2', 0)),
                        "width": float(doc.metadata.get('width', 0)),
                        "height": float(doc.metadata.get('height', 0))
                    },
                    "pageNumber": int(doc.metadata.get('page_number', 1))
                },
                "comment": {
                    "text": "Source text for the answer",
                    "emoji": "💡"
                },
                "filename": filename  # Use the retrieved filename
            }
            
            logging.info(f"Creating highlight for PDF {pdf_id}: {highlight_data}")
            highlight_id = store_highlight(highlight_data)
            documents[pdf_id].append(highlight_id)
        
        logging.info(f"Final documents and highlights mapping: {documents}")
        
        # Store the chat message
        store_chat_message(
            session_id=request.session_id,
            user_query=request.question,
            gpt_response=response.get("answer", "")
        )
        
        # Format the response according to new structure
        return ChatResponse(
            answer=response.get("answer", ""),
            session_id=request.session_id,
            model=request.model,
            name=chat_data.get("name"),
            user_id=request.user_id,
            documents=documents
        )
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        logging.exception("Full traceback:")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.delete("/delete-chat-history")
def delete_chat_history(user_id:int,session_id:str):
    return delete_chat_session(user_id,session_id)

@app.post("/upload-pdf")
async def upload_document(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    current_user: User = Depends(get_current_user)
):
    try:
        # Log initial request details
        logging.info(f"Starting PDF upload process...")
        logging.info(f"File name: {file.filename}")
        logging.info(f"User ID: {user_id}")
        logging.info(f"Content type: {file.content_type}")

        # Validate file type
        if not file.filename.endswith('.pdf'):
            logging.warning(f"Invalid file type attempted: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        try:
            # Read file content as bytes
            logging.info("Reading file content...")
            file_content = await file.read()  # This returns bytes
            file_size = len(file_content)
            logging.info(f"File size: {file_size} bytes")

            if file_size == 0:
                logging.error("Empty file detected")
                raise HTTPException(
                    status_code=400,
                    detail="Empty file detected"
                )

            # Create temporary file for processing
            temp_file_path = f"temp_{uuid.uuid4()}.pdf"
            try:
                with open(temp_file_path, "wb") as temp_file:
                    temp_file.write(file_content)

                # Store PDF in database
                logging.info("Storing PDF in database...")
                pdf_id = store_pdf(file.filename, user_id, file_content)
                logging.info(f"PDF stored successfully with ID: {pdf_id}")

                # Process the PDF and get chunks
                chunks = process_pdf(temp_file_path)

                # Index in vector database
                logging.info("Indexing PDF in vector database...")
                indexing_success = index_document_to_chroma(
                    file_id=pdf_id,
                    chunks=chunks
                )

                if not indexing_success:
                    raise Exception("Failed to index document in vector store")

                return {"message": "Document uploaded successfully", "pdf_id": pdf_id}

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    logging.info(f"Cleaned up temporary file: {temp_file_path}")

        except HTTPException as he:
            raise he
        except Exception as inner_e:
            logging.error(f"Inner error during upload: {str(inner_e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process upload: {str(inner_e)}"
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logging.error(f"Error during upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents(user_id:int):
    return get_all_documents(user_id)

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    logging.info(f"Starting document deletion process for file_id: {request.file_id}, user_id: {request.user_id}")
    
    try:
        # Log all documents before deletion
        all_docs_before = get_all_documents(request.user_id)
        logging.info("Documents in database BEFORE deletion:")
        for doc in all_docs_before:
            logging.info(f"Document: {doc}")
        
        # First delete from Chroma (vector store)
        logging.info(f"Attempting to delete document and chunks from Chroma for file_id: {request.file_id}")
        chroma_delete_success = delete_doc_from_chroma(request.file_id, request.user_id)
        logging.info(f"Chroma deletion status for file_id {request.file_id}: {chroma_delete_success}")

        if chroma_delete_success:
            # Then delete the PDF file
            logging.info(f"Attempting to delete PDF file for file_id: {request.file_id}")
            pdf_delete_success = delete_pdf(request.file_id, request.user_id)
            logging.info(f"PDF deletion status for file_id {request.file_id}: {pdf_delete_success}")

            # Finally delete the document record
            logging.info(f"Attempting to delete document record from database for file_id: {request.file_id}")
            db_delete_success = delete_document_record(request.file_id, request.user_id)
            logging.info(f"Database deletion status for file_id {request.file_id}: {db_delete_success}")
            
            # Log all documents after deletion
            all_docs_after = get_all_documents(request.user_id)
            logging.info("Documents in database AFTER deletion:")
            for doc in all_docs_after:
                logging.info(f"Document: {doc}")
            
            if db_delete_success and pdf_delete_success:
                logging.info(f"Successfully deleted document with file_id {request.file_id} from Chroma, PDF store, and database")
                return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
            else:
                error_msg = []
                if not pdf_delete_success:
                    error_msg.append("PDF file deletion failed")
                if not db_delete_success:
                    error_msg.append("Database record deletion failed")
                error_str = " and ".join(error_msg)
                logging.error(f"Partial deletion for file_id {request.file_id}: {error_str}")
                return {"error": f"Deleted from Chroma but {error_str}."}
        else:
            logging.error(f"Failed to delete document with file_id {request.file_id} from Chroma")
            return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
            
    except Exception as e:
        logging.error(f"Error during document deletion for file_id {request.file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during document deletion: {str(e)}"
        )

@app.post("/register")
async def register(user: UserRegister):
    try:
        # Check if user already exists
        existing_user = get_user_by_email(user.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )

        # Hash password and create user
        hashed_password = hash_password(user.password)
        user_id = insert_user(user.email, hashed_password)
        
        # Create access token
        token = create_access_token(data={"sub": str(user_id)})
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_id
        }
        
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during registration"
        )

@app.post("/login")
def login(userlogin: UserRegister):
    try:
        user = get_user_by_email(userlogin.email)
        if not user or not verify_password(userlogin.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid credentials"
            )

        # Ensure user_id is converted to string when creating token
        token = create_access_token(data={"sub": str(user["id"])})
        
        logging.info(f"User {user['id']} logged in successfully")
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user["id"]
        }
    except Exception as e:
        logging.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

def process_image(image_path: str) -> Dict:
    try:
        # Load the image using Pillow
        img = Image.open(image_path)
        custom_config = r'-l deu --psm 6 -c preserve_interword_spaces=1'
        # Extract text using OCR
        ocr_text = pytesseract.image_to_string(img, config=custom_config)

        # Extract image metadata
        metadata = {
            "format": img.format,
            "size": img.size,  # (width, height)
            "mode": img.mode
        }

        return {
            "text": ocr_text,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

def process_document(file_path: str, user_id: int) -> Dict:
    file_id = insert_document_record(file_path, user_id)
    print("before the chroma db", file_id)
    success = index_document_to_chroma(file_path, file_id, user_id)

    if success:
        return {"file_id": file_id, "status": "indexed"}
    else:
        delete_document_record(file_id, user_id)
        raise HTTPException(status_code=500, detail=f"Failed to index {file_path}.")

@app.post("/reset")
async def reset_password(request: UserRegister):
    user = get_user_by_email(request.email)
    if user:
        hashed_password = hash_password(request.password)
            
        return reset_password_db(request.email, hashed_password)
    else:
        raise HTTPException(status_code=404, detail="User not found")
    
@app.get("/sources/{filename}")
async def get_source(filename: str):
    """Serve source files (PDF or HTML) with appropriate content type.

    Args:
        filename: Name of the file to serve

    Returns:
        FileResponse or HTMLResponse depending on file type

    Raises:
        HTTPException: If file not found or invalid type
    """
    try:
        # Assuming files are stored in a 'data' directory relative to the backend
        base_path = os.path.abspath("data")
        file_path = os.path.normpath(os.path.join(base_path, filename))

        if not file_path.startswith(base_path):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Determine content type from file extension
        content_type, _ = mimetypes.guess_type(file_path)

        if content_type == "application/pdf":
            return FileResponse(
                file_path,
                media_type="application/pdf",
                filename=filename
            )
        elif content_type in ["text/html", "text/plain"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return HTMLResponse(content=content)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type"
            )

    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdfs/{user_id}")
async def list_user_pdfs(user_id: int):
    try:
        logging.info(f"Fetching PDFs for user_id: {user_id}")
        pdfs = get_all_user_pdfs(user_id)
        logging.info(f"Found PDFs: {pdfs}")
        return pdfs
    except Exception as e:
        logging.error(f"Error listing PDFs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/pdf/{pdf_id}")
async def get_pdf_file(pdf_id: str, user_id: str):
    try:
        # Convert string IDs to integers
        pdf_id_int = int(pdf_id)
        user_id_int = int(user_id)
        
        logging.info(f"Retrieving PDF {pdf_id_int} for user_id: {user_id_int}")
        file_data, filename = get_pdf(pdf_id_int, user_id_int)
        
        if not file_data:
            raise HTTPException(
                status_code=404,
                detail=f"PDF {pdf_id} not found or access denied"
            )
            
        return Response(
            content=file_data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except ValueError as ve:
        logging.error(f"Invalid ID format - pdf_id: {pdf_id}, user_id: {user_id}")
        raise HTTPException(
            status_code=400,
            detail="Invalid ID format"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.delete("/pdf/{pdf_id}")
async def delete_pdf_file(pdf_id: int, user_id: int):
    return delete_pdf(pdf_id, user_id)

@app.patch("/chat-name")
async def update_chat_name_endpoint(request: ChatNameUpdate):
    try:
        success = update_chat_name(
            request.session_id,
            request.user_id,
            request.name
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
            
        return {"message": "Chat name updated successfully"}
        
    except Exception as e:
        logging.error(f"Error updating chat name: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/pdf-highlights")
async def get_pdf_highlights_endpoint(
    pdf_id: int,
    current_user: User = Depends(get_current_user)
):
    try:
        highlights = get_pdf_highlights(pdf_id, current_user.id)
        return {
            "pdf_id": pdf_id,
            "highlights": highlights
        }
    except Exception as e:
        logging.error(f"Error fetching highlights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/document-highlights")
async def get_document_highlights_endpoint(request: DocumentHighlightsRequest):
    try:
        logging.info(f"Document highlights request: {request.dict()}")
        doc_data = get_document_highlights(
            pdf_id=int(request.pdf_id),  # Convert string to int
            highlight_ids=request.highlight_ids
        )
        
        if not doc_data['documentData']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
            
        response = DocumentHighlightsResponse(**doc_data)
        logging.info(f"Returning response: {response.dict()}")
        return response
        
    except ValueError as ve:
        logging.error(f"Invalid PDF ID format: {request.pdf_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF ID format"
        )
    except Exception as e:
        logging.error(f"Error fetching document highlights: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/clear-data")
async def clear_all_data(current_user: User = Depends(get_current_user)):
    """Clear all application data while preserving user accounts."""
    try:
        clear_all_data_except_users()
        return {"message": "Successfully cleared all data"}
    except Exception as e:
        logging.error(f"Error clearing data: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

