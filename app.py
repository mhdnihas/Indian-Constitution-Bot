from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document
from huggingface_hub import hf_hub_download
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import faiss
import json
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import os
from typing import List
import sys
from typing import List, Optional, Any, Dict
from langchain.llms.base import LLM




app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

class Document:
    def __init__(self, page_content: str, metadata: dict,doc_id: int):
        self.page_content = page_content
        self.metadata = metadata
        self.id = doc_id 

class GeminiLLM(LLM):
    model: Any
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": "gemini-pro"}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

class GeminiLLM(LLM):
    model: Any
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": "gemini-pro"}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

def setup_models():
    """Initialize the language and embedding models"""
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-pro")
    llm = GeminiLLM(model=gemini_model)
    
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    return llm, embeddings

def load_metadata():
    """Load data from JSON file"""
    with open("faiss_metadata.json", "r", encoding='utf-8') as f:
        texts = json.load(f)
    return texts

def create_documents(texts: List[str]) -> List[Document]:
    """Create Document objects from text strings"""
    documents = []
    for i, text in enumerate(texts):
        try:
            documents.append(
                Document(
                    page_content=text,  # Use the correct attribute name for the content
                    metadata={"source": f"document_{i}"},
                    doc_id=i                )
            )
        except Exception as e:
            raise ValueError(f"Error creating document {i}: {str(e)}")
    
    return documents


def setup_vectorstore(documents: List[Document], embeddings) -> FAISS:
    """Create and save FAISS vectorstore"""
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore
    except Exception as e:
        raise ValueError(f"Error creating vectorstore: {str(e)}")

def setup_qa_chain(vectorstore, llm):
    """Set up the question-answering chain"""
    template = """
    You are a helpful assistant trained to answer questions about the Indian Constitution. 
    Below are some relevant sections from the Constitution:

    {documents}

    Please answer the following question based on the above context:
    {question}

    If the answer cannot be directly found in the provided context, please say so.
    """

    prompt = PromptTemplate(
        input_variables=["documents", "question"],
        template=template
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

# Initialize everything
try:
    print("Starting initialization...")
    llm, embeddings = setup_models()
    print("Models initialized successfully")
    
    data = load_metadata()
    print(f"Loaded {len(data)} documents from metadata")
    
    documents = create_documents(data)
    print(f"Created {len(documents)} Document objects")


    
    vectorstore = setup_vectorstore(documents, embeddings)
    print("Vectorstore created successfully")
    
    qa_chain = setup_qa_chain(vectorstore, llm)
    print("QA chain initialized successfully")
    
except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise




class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.post("/chatbot", response_model=ChatResponse)
async def chatbot(request: ChatRequest):
    user_message = request.message

    if GEMINI_API_KEY is None:

        raise HTTPException(status_code=500, detail="Gemini API key not found in environment variables")
    
    
    response = llm.generate_content(user_message)

    print("hai")
    return JSONResponse(content={"reply": response.text})
