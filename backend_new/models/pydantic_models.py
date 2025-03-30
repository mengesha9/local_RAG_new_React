from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

class ModelName(str, Enum):
    GPT4_O = "gpt-4o"
    GPT4_O_MINI = "gpt-4o-mini"
    LLAMA3_1 = "llama3.1"
    LLAMA3_2 = "llama3.2"

# Define DocumentChunk first since it's used in QueryResponse
class DocumentChunk(BaseModel):
    chunk_text: str
    page_number: int
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    doc_id: int          # Add document ID
    filename: str        # Add filename

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    user_id: int
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)

class BoundingRect(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float

class HighlightRect(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    pageNumber: Optional[int] = None

class HighlightPosition(BaseModel):
    boundingRect: BoundingRect
    rects: List[HighlightRect]
    pageNumber: int

class HighlightContent(BaseModel):
    text: Optional[str] = None

class HighlightComment(BaseModel):
    text: str
    emoji: str

class Highlight(BaseModel):
    id: str
    content: HighlightContent
    position: HighlightPosition
    comment: HighlightComment

class DocumentHighlight(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: HighlightContent
    position: HighlightPosition
    comment: HighlightComment
    doc_id: int
    filename: str

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    user_id: int
    model: ModelName
    highlights: List[Dict] = []

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    user_id: int
    file_size: int  = None
    content_type: str = None

class DeleteFileRequest(BaseModel):
    file_id: int
    user_id: int

class User(BaseModel):
    id: int
    email: str
    
    class Config:
        from_attributes = True

# Add these new models
class ChatNameUpdate(BaseModel):
    user_id: int
    session_id: str
    name: str

class ChatRequest(BaseModel):
    question: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))  # Generate UUID if not provided
    user_id: int
    model: ModelName = Field(default=ModelName.GPT4_O_MINI)
    name: Optional[str] = None  # Optional chat name

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    model: str
    name: Optional[str]
    user_id: int
    documents: Dict[int, List[str]]  # pdf_id -> list of highlight_ids

class DocumentHighlightsRequest(BaseModel):
    pdf_id: int
    highlight_ids: List[str]

class DocumentHighlightsResponse(BaseModel):
    documentData: Dict[str, Any]
    highlights: List[Dict[str, Any]]

class ChatHistoryResponse(BaseModel):
    model: str
    timestamp: datetime
    name: Optional[str]
    queries: List[Dict[str, Any]]

# Add this new model
class HighlightResponse(BaseModel):
    pdf_id: int
    filename: str
    highlights: List[Dict[str, Any]] = []


