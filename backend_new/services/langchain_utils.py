from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from uuid import uuid4

import os
from services.vector_store_db import get_vectorstore
from dotenv import load_dotenv
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
import chromadb

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY: inside the langchain",OPENAI_API_KEY)

retriever = get_vectorstore().as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,  # Number of documents to retrieve
        "fetch_k": 4,  # Fetch more documents than k for better diversity
        "filter": None,  # Will be set per user in the chain
    }
)

output_parser = StrOutputParser()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

prompt_file_path = "./prompt.txt"
try:
    with open(prompt_file_path, "r") as file:
        dynamic_prompt_text = file.read().strip()
except FileNotFoundError:
    raise Exception(f"Prompt file not found at: {prompt_file_path}")


print("Dynamic Prompt Text: ",dynamic_prompt_text)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",dynamic_prompt_text),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_source_chunks(query: str, k: int = 4) -> List[Document]:
    """Get relevant chunks from the vector store."""
    try:
        vectorstore = get_vectorstore()
        
        # Get relevant documents with their scores
        docs_and_scores = vectorstore.similarity_search_with_score(
            query,
            k=k * 2,  # Get more docs initially to filter
            search_kwargs={
                "k": k * 2,
                "fetch_k": k * 4,  # Fetch more for better diversity
            }
        )
        
        # Log the retrieved documents for debugging
        logging.info(f"Query: {query}")
        for doc, score in docs_and_scores:
            logging.info(f"Retrieved document with score {score}:")
            logging.info(f"Content: {doc.page_content}")
            logging.info(f"Metadata: {doc.metadata}")

        # Filter and sort documents
        documents = [doc for doc, _ in docs_and_scores][:k]
        
        if not documents:
            logging.warning(f"No relevant documents found for query: {query}")
            return []
            
        logging.info(f"Retrieved {len(documents)} relevant chunks")
        return documents
        
    except Exception as e:
        logging.error(f"Error getting source chunks: {str(e)}")
        logging.error("Full error details:", exc_info=True)
        return []

def get_rag_chain(model_name: str = "gpt-4"):
    """Create a RAG chain with the specified model."""
    try:
        # Get the singleton vectorstore instance
        vectorstore = get_vectorstore()
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        llm_instance = ChatOpenAI(model_name=model_name)
        
        if model_name == ("llama3.1"):
            llm_instance = ChatOllama(model="llama3.1:latest")
        elif model_name == ("llama3.2"):
            llm_instance = ChatOllama(model="llama3.2:3b")
        # If you don't know the answer, just say that you don't know, don't try to make up an answer.

        # Create prompt template
        prompt = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            
            
            Context: {context}
            
            Chat History: {chat_history}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create memory with specified input/output keys
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",  # Specify input key
            output_key="answer",   # Specify output key
            return_messages=True
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_instance,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        
        return chain
        
    except Exception as e:
        logging.error(f"Error creating RAG chain: {str(e)}")
        raise