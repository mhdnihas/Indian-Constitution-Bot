from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import json
from langchain_community.vectorstores import FAISS
import os
from typing import List
from typing import List, Optional, Any, Dict
from langchain.llms.base import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



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

llm, embeddings, vectorstore = None, None, None

def setup_models():
    """Initialize the language and embedding models"""

    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)

    gemini_model = genai.GenerativeModel("gemini-pro")
    
    llm = GeminiLLM(model=gemini_model)
    

    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
         model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs 

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
                    doc_id=i)
            )
        except Exception as e:
            raise ValueError(f"Error creating document {i}: {str(e)}")
    
    return documents


def setup_vectorstore(documents: List[Document], embeddings) -> FAISS:
    """Create and save FAISS vectorstore"""
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")


    return vectorstore
    
@app.on_event("startup")
async def startup_event():
    global llm, embeddings, vectorstore
    llm, embeddings = setup_models()
    data = load_metadata()
    documents = create_documents(data)
    vectorstore = setup_vectorstore(documents, embeddings)




def retrieval_chain_llm_response(vectorstore, llm,embeddings,input):
    """Set up the question-answering chain"""

    template = """
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    You are a helpful assistant trained to answer questions about the Indian Constitution. 
    Below are some relevant sections from the Constitution:

    <context>
    {context}
    </context>

    Please answer the following question based on the above context:
    {input}

    If the answer cannot be directly found in the provided context, please say so.
    """

    
    prompt = ChatPromptTemplate.from_template(template)

   

    document_chain=create_stuff_documents_chain(llm,prompt)


    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})

    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    response = retrieval_chain.invoke({"input": input})


    print(response['answer']) 

    return response['answer']


def gemini_response(user_message):
    pass


llm, embeddings = setup_models()
data = load_metadata()
documents = create_documents(data)
vectorstore = setup_vectorstore(documents, embeddings)

    



    



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

    response=retrieval_chain_llm_response(vectorstore, llm,embeddings,user_message)
    

    print("hai")
    return JSONResponse(content={"reply": response})
