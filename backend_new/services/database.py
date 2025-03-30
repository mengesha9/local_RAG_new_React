from fastapi import HTTPException
import sqlite3
from datetime import datetime
import logging
import json
from typing import List, Dict, Optional, Any
import os
import shutil

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS chats
                    (session_id TEXT PRIMARY KEY,
                     name TEXT,
                     user_id INTEGER,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)''')
                     
    conn.execute('''CREATE TABLE IF NOT EXISTS chat_messages
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (session_id) REFERENCES chats (session_id) ON DELETE CASCADE)''')

    # Updated highlights table
    conn.execute('''CREATE TABLE IF NOT EXISTS highlights
                    (highlight_id TEXT PRIMARY KEY,
                     chat_id TEXT,
                     pdf_id INTEGER,
                     content_text TEXT,
                     position_json TEXT,
                     comment_text TEXT DEFAULT 'Source text for the answer',
                     comment_emoji TEXT DEFAULT '💡',
                     filename TEXT,
                     page_number INTEGER,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY (chat_id) REFERENCES chats (session_id) ON DELETE CASCADE,
                     FOREIGN KEY (pdf_id) REFERENCES pdf_store (id) ON DELETE CASCADE)''')
    
    # Create indices for better performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_highlights_pdf_id ON highlights(pdf_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_highlights_chat_id ON highlights(chat_id)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id)')
    conn.close()

def create_document_store():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     user_id INTEGER,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     pdf_url TEXT,
                     FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)''')
    
    # Create new table for document chunks with metadata
    conn.execute('''CREATE TABLE IF NOT EXISTS document_chunks
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     doc_id INTEGER,
                     chunk_text TEXT,
                     page_number INTEGER,
                     x1 FLOAT,
                     y1 FLOAT,
                     x2 FLOAT,
                     y2 FLOAT,
                     width FLOAT,
                     height FLOAT,
                     FOREIGN KEY (doc_id) REFERENCES document_store (id) ON DELETE CASCADE)''')
    conn.close()

def create_users_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     email TEXT UNIQUE NOT NULL,
                     hashed_password TEXT NOT NULL)''')
    conn.close()


def insert_application_logs(session_id,user_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?, ?)',
                 (session_id,user_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

# def get_user_chat_history(user_id):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute('SELECT session_id, user_query, gpt_response FROM application_logs WHERE user_id = ? ORDER BY created_at', (user_id,))
#     chat_history = {}
#     for row in cursor.fetchall():

#         session_id = row['session_id']
        
#         # Initialize chat_history for session_id if it doesn't exist
#         if session_id not in chat_history:
#             chat_history[session_id] = []
        
#         chat_history[session_id].extend([
#             {"role": "human", "content": row['user_query']},
#             {"role": "ai", "content": row['gpt_response']}
#         ])
#     conn.close()
#     return chat_history

def get_chat_history(session_id: str, user_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_query, gpt_response 
            FROM application_logs 
            WHERE session_id = ? AND user_id = ?
            ORDER BY created_at ASC
        ''', (session_id, user_id))
        
        history = cursor.fetchall()
        conn.close()
        
        # Format history for the chat context
        return [(row[0], row[1]) for row in history]
        
    except Exception as e:
        logging.error(f"Error getting chat history: {str(e)}")
        return []

def get_user_chat_history(user_id: int) -> Dict[str, Any]:
    """Get all chat history with highlights grouped by document."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all chats for user from the chats table instead of application_logs
        cursor.execute('''
            SELECT session_id, model, name, created_at
            FROM chats 
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        sessions = {}
        for chat in cursor.fetchall():
            session_id = chat['session_id']
            
            # Get messages for this chat from chat_messages table
            cursor.execute('''
                SELECT user_query, gpt_response, created_at
                FROM chat_messages 
                WHERE session_id = ?
                ORDER BY created_at
            ''', (session_id,))
            
            queries = []
            for msg in cursor.fetchall():
                # Get highlight IDs grouped by document
                cursor.execute('''
                    SELECT pdf_id, highlight_id
                    FROM highlights
                    WHERE chat_id = ?
                ''', (session_id,))
                
                documents = {}
                for h in cursor.fetchall():
                    pdf_id = h['pdf_id']
                    if pdf_id not in documents:
                        documents[pdf_id] = []
                    documents[pdf_id].append(h['highlight_id'])
                
                queries.append({
                    'query': msg['user_query'],
                    'response': msg['gpt_response'],
                    'timestamp': msg['created_at'],
                    'documents': documents
                })
            
            sessions[session_id] = {
                'model': chat['model'],
                'name': chat['name'],  # This will now come from the chats table
                'timestamp': chat['created_at'],
                'queries': queries
            }
        
        conn.close()
        return sessions
        
    except Exception as e:
        logging.error(f"Error getting chat history: {str(e)}")
        raise

def delete_chat_session(user_id,session_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM application_logs WHERE user_id = ? AND session_id = ?', (user_id,session_id))
    conn.commit()
    conn.close()
    return True

def insert_document_record(filename,user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, user_id) VALUES (?,?)', (filename,user_id))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id,user_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ? AND user_id = ?', (file_id,user_id))
    conn.commit()
    conn.close()
    return True

def get_all_documents(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, user_id, filename, upload_timestamp FROM document_store WHERE user_id = ? ORDER BY upload_timestamp DESC', (user_id,))
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]





def insert_user(email: str, hashed_password: str) -> int:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO users (email, hashed_password) VALUES (?, ?)',
            (email, hashed_password)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
        
    except Exception as e:
        logging.error(f"Database error during user insertion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create user"
        )

def get_user_by_email(email: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, email, hashed_password FROM users WHERE email = ?',
            (email,)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        return user
        
    except Exception as e:
        logging.error(f"Database error during user lookup: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to lookup user"
        )

def reset_password_db(email: str, hashed_password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET hashed_password = ? WHERE email = ?', (hashed_password, email))
    conn.commit()
    conn.close()
    return {"message": "Password reset successfully!"}

def create_pdf_store():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS pdf_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT NOT NULL,
                         file_data BLOB NOT NULL,
                         user_id INTEGER,
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                         FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE)''')
        
        conn.commit()
        conn.close()
        logging.info("PDF store table created successfully")
        
    except Exception as e:
        logging.error(f"Error creating PDF store table: {str(e)}")
        raise

def store_pdf(filename: str, user_id: int, file_content: bytes) -> int:
    """Store PDF file in database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        logging.info(f"Storing PDF for user_id: {user_id}")
        
        # Make sure file_content is bytes
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
        
        cursor.execute('''
            INSERT INTO pdf_store (filename, user_id, file_data) 
            VALUES (?, ?, ?)
        ''', (filename, user_id, file_content))
        
        pdf_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logging.info(f"PDF {filename} stored successfully with ID: {pdf_id} for user_id: {user_id}")
        return pdf_id
        
    except Exception as e:
        logging.error(f"Error storing PDF: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store PDF: {str(e)}"
        )

def get_all_user_pdfs(user_id: int) -> List[Dict]:
    """Get all PDFs for a user."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First get all PDFs
        cursor.execute('''
            SELECT 
                p.id,
                p.filename,
                p.upload_timestamp
            FROM pdf_store p
            WHERE p.user_id = ?
            ORDER BY p.upload_timestamp DESC
        ''', (user_id,))
        
        pdfs = []
        for row in cursor.fetchall():
            pdf = dict(row)
            
            # Get highlights for this PDF
            cursor.execute('''
                SELECT highlight_id, chat_id
                FROM highlights
                WHERE pdf_id = ?
                ORDER BY created_at DESC
            ''', (pdf['id'],))
            
            highlights = cursor.fetchall()
            pdf['highlight_ids'] = [h['highlight_id'] for h in highlights]
            
            # Get latest chat_id if there are any highlights
            pdf['latest_chat_id'] = highlights[0]['chat_id'] if highlights else None
            
            pdfs.append(pdf)
            
        conn.close()
        logging.info(f"Retrieved {len(pdfs)} PDFs for user {user_id}")
        return pdfs
        
    except Exception as e:
        logging.error(f"Error getting user PDFs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch PDFs: {str(e)}"
        )

def get_pdf(pdf_id: int, user_id: int) -> tuple[bytes, str]:
    """Get PDF file data and filename."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if the PDF exists and belongs to the user
        cursor.execute('''
            SELECT filename, file_data 
            FROM pdf_store 
            WHERE id = ? AND user_id = ?
        ''', (pdf_id, user_id))
        
        result = cursor.fetchone()
        if not result:
            return None, None
            
        conn.close()
        return result['file_data'], result['filename']
        
    except Exception as e:
        logging.error(f"Error retrieving PDF {pdf_id}: {str(e)}")
        raise

def delete_pdf(pdf_id: int, user_id: int) -> bool:
    """Delete PDF and all associated data (chunks, highlights) from the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Start transaction
        conn.execute('BEGIN')
        
        try:
            # First verify the PDF exists and belongs to the user
            cursor.execute(
                'SELECT id FROM pdf_store WHERE id = ? AND user_id = ?',
                (pdf_id, user_id)
            )
            if not cursor.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF {pdf_id} not found or access denied"
                )
            
            # Delete associated document chunks
            cursor.execute('DELETE FROM document_chunks WHERE doc_id = ?', (pdf_id,))
            chunks_deleted = cursor.rowcount
            logging.info(f"Deleted {chunks_deleted} chunks for PDF {pdf_id}")
            
            # Delete associated highlights
            cursor.execute('DELETE FROM highlights WHERE pdf_id = ?', (pdf_id,))
            highlights_deleted = cursor.rowcount
            logging.info(f"Deleted {highlights_deleted} highlights for PDF {pdf_id}")
            
            # Finally delete the PDF
            cursor.execute(
                'DELETE FROM pdf_store WHERE id = ? AND user_id = ?',
                (pdf_id, user_id)
            )
            
            # Commit transaction
            conn.commit()
            logging.info(f"Successfully deleted PDF {pdf_id} and all associated data")
            return True
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            raise e
            
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error deleting PDF {pdf_id}: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete PDF: {str(e)}"
        )

def get_user_by_id(user_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        return {
            "id": user[0],
            "email": user[1],
            "hashed_password": user[2]
        }
    return None

# Initialize the database tables
create_application_logs()
create_document_store()
create_users_table()

def store_highlight(highlight_data: dict) -> str:
    """Store a highlight with its text content."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # If filename is missing, get it from the pdf_store table
        if not highlight_data.get('filename'):
            cursor.execute('SELECT filename FROM pdf_store WHERE id = ?', (highlight_data['pdf_id'],))
            result = cursor.fetchone()
            highlight_data['filename'] = result['filename'] if result else 'unknown.pdf'
        
        position_json = {
            'boundingRect': highlight_data['position']['boundingRect'],
            'rects': [highlight_data['position']['boundingRect']],
            'pageNumber': highlight_data['position']['pageNumber']
        }
        
        cursor.execute('''
            INSERT INTO highlights (
                highlight_id, chat_id, pdf_id, content_text, 
                position_json, comment_text, comment_emoji, 
                filename, page_number
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            highlight_data['highlight_id'],
            highlight_data['chat_id'],
            highlight_data['pdf_id'],
            highlight_data['content_text'],
            json.dumps(position_json),
            highlight_data['comment']['text'],
            highlight_data['comment']['emoji'],
            highlight_data['filename'],
            highlight_data['position']['pageNumber']
        ))
        
        conn.commit()
        conn.close()
        return highlight_data['highlight_id']
        
    except Exception as e:
        logging.error(f"Error storing highlight: {str(e)}")
        raise

def update_chat_name(session_id: str, user_id: int, name: str) -> bool:
    """Update the name of a chat session."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First get the current name
        cursor.execute('''
            SELECT name 
            FROM chats 
            WHERE session_id = ?
        ''', (session_id,))
        
        current = cursor.fetchone()
        old_name = current['name'] if current else None
        
        # Update the chat name in the chats table
        cursor.execute('''
            UPDATE chats 
            SET name = ? 
            WHERE session_id = ?
        ''', (name, session_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        # Log the change
        logging.info(f"Chat name update for session {session_id}:")
        logging.info(f"Old name: {old_name}")
        logging.info(f"New name: {name}")
        
        return success
        
    except Exception as e:
        logging.error(f"Error updating chat name: {str(e)}")
        raise

def get_pdf_highlights(pdf_id: int, user_id: int) -> List[Dict]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT h.* 
            FROM highlights h
            JOIN application_logs a ON h.chat_id = a.session_id
            WHERE h.pdf_id = ? AND a.user_id = ?
            ORDER BY h.created_at DESC
        ''', (pdf_id, user_id))
        
        highlights = cursor.fetchall()
        conn.close()
        
        return [{
            'highlight_id': h['highlight_id'],
            'chat_id': h['chat_id'],
            'content_text': h['content_text'],
            'position': json.loads(h['position_json']),
            'comment': {
                'text': h['comment_text'],
                'emoji': h['comment_emoji']
            },
            'filename': h['filename'],
            'page_number': h['page_number']
        } for h in highlights]
        
    except Exception as e:
        logging.error(f"Error fetching PDF highlights: {str(e)}")
        raise

def create_or_get_chat(session_id: str, user_id: int, model: str, name: Optional[str] = None) -> Dict:
    """Create a new chat session or get existing one."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if chat exists
        cursor.execute('''
            SELECT session_id, name, model, created_at 
            FROM chats 
            WHERE session_id = ? AND user_id = ?
        ''', (session_id, user_id))
        
        existing_chat = cursor.fetchone()
        
        if existing_chat:
            chat_data = dict(existing_chat)
        else:
            # Create new chat
            cursor.execute('''
                INSERT INTO chats (session_id, user_id, model, name)
                VALUES (?, ?, ?, ?)
            ''', (session_id, user_id, model, name))
            
            chat_data = {
                "session_id": session_id,
                "name": name,
                "model": model,
                "created_at": datetime.now().isoformat()
            }
            
        conn.commit()
        conn.close()
        return chat_data
        
    except Exception as e:
        logging.error(f"Error in create_or_get_chat: {str(e)}")
        raise

def store_chat_message(session_id: str, user_query: str, gpt_response: str) -> int:
    """Store a new chat message."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_messages (session_id, user_query, gpt_response)
            VALUES (?, ?, ?)
        ''', (session_id, user_query, gpt_response))
        
        message_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return message_id
        
    except Exception as e:
        logging.error(f"Error storing chat message: {str(e)}")
        raise

def check_pdf_exists(pdf_id: int) -> bool:
    """Check if a PDF exists in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM pdf_store WHERE id = ?', (pdf_id,))
        result = cursor.fetchone() is not None
        conn.close()
        return result
    except Exception as e:
        logging.error(f"Error checking PDF existence: {str(e)}")
        return False

def get_document_highlights(pdf_id: int, highlight_ids: List[str]) -> Dict:
    """Get document data and specified highlights."""
    try:
        logging.info(f"Fetching highlights for PDF {pdf_id} with highlight_ids: {highlight_ids}")
        
        # First check if PDF exists
        if not check_pdf_exists(pdf_id):
            logging.error(f"PDF {pdf_id} not found in database")
            raise HTTPException(
                status_code=404,
                detail=f"Document {pdf_id} not found"
            )
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get document data with more detailed logging
        cursor.execute('''
            SELECT id as pdf_id, filename, user_id as uploaded_by, upload_timestamp
            FROM pdf_store 
            WHERE id = ?
        ''', (pdf_id,))
        
        doc_data = cursor.fetchone()
        logging.info(f"Document data found: {dict(doc_data) if doc_data else None}")
        
        if not doc_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {pdf_id} not found"
            )
            
        doc_data = dict(doc_data)
        
        # Get specified highlights
        placeholders = ','.join(['?' for _ in highlight_ids])
        query = f'''
            SELECT h.*, c.chunk_text 
            FROM highlights h
            LEFT JOIN document_chunks c ON 
                c.doc_id = h.pdf_id AND 
                c.page_number = h.page_number
            WHERE h.pdf_id = ? AND h.highlight_id IN ({placeholders})
            ORDER BY h.created_at
        '''
        logging.info(f"Executing query: {query}")
        logging.info(f"With parameters: {[pdf_id] + highlight_ids}")
        
        cursor.execute(query, [pdf_id] + highlight_ids)
        
        highlights = []
        for h in cursor.fetchall():
            logging.info(f"Processing highlight: {dict(h)}")
            position_data = json.loads(h['position_json'])
            highlights.append({
                'id': h['highlight_id'],
                'content': {
                    'text': h['chunk_text'] or h['content_text'] or ''
                },
                'position': {
                    'boundingRect': position_data['boundingRect'],
                    'rects': [position_data['boundingRect']],
                    'pageNumber': h['page_number']
                },
                'comment': {
                    'text': h['comment_text'] or '',
                    'emoji': h['comment_emoji'] or '💡'
                }
            })
        
        conn.close()
        result = {
            'documentData': doc_data,
            'highlights': highlights
        }
        logging.info(f"Returning result: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Error fetching document highlights: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def print_database_info():
    """Print database schema and contents for debugging."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("\n=== DATABASE SCHEMA AND CONTENTS ===\n")
        
        for table in tables:
            table_name = table[0]
            print(f"\n=== TABLE: {table_name} ===")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("\nSchema:")
            for col in columns:
                print(f"  {col['name']} ({col['type']})")
            
            # Get table contents
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
            rows = cursor.fetchall()
            print(f"\nContents (up to 5 rows):")
            for row in rows:
                print(f"  {dict(row)}")
                
        conn.close()
        
    except Exception as e:
        logging.error(f"Error printing database info: {str(e)}")
        raise

def clear_all_data_except_users():
    """Clear all data from the database except user login information."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Start transaction
        conn.execute('BEGIN')
        
        try:
            # Delete data from all tables except users
            # Order matters due to foreign key constraints
            tables_to_clear = [
                'highlights',
                'document_chunks',
                'chat_messages',
                'chats',
                'application_logs',
                'pdf_store',
                'document_store'
            ]
            
            for table in tables_to_clear:
                cursor.execute(f'DELETE FROM {table}')
                rows_deleted = cursor.rowcount
                logging.info(f"Deleted {rows_deleted} rows from {table}")
            
            # Also delete the vector store
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                logging.info("Cleared vector store database")
            
            # Commit transaction
            conn.commit()
            logging.info("Successfully cleared all data while preserving user information")
            
        except Exception as e:
            # Rollback on error
            conn.rollback()
            raise e
            
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error clearing database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear database: {str(e)}"
        )
