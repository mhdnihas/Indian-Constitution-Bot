from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
from typing import List, Optional, Any, Dict
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


load_dotenv()


# Define custom document structure
class ConstitutionDocument:
    def __init__(self, content: str, metadata: dict, doc_id: int):
        self.page_content = content
        self.metadata = metadata
        self.id = doc_id


# Define custom LLM class
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


def initialize_models():
    """Initialize language model and embedding model with required configuration."""
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-pro")
    
    llm_instance = GeminiLLM(model=gemini_model)

    model_settings = {'device': 'cpu'}
    embedding_settings = {'normalize_embeddings': False}

    embeddings_instance = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs=model_settings,
        encode_kwargs=embedding_settings
    )
    
    return llm_instance, embeddings_instance


def load_json_metadata(filepath: str = "faiss_metadata.json") -> List[str]:
    """Load document metadata from JSON file."""
    with open(filepath, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

def build_documents(data: List[str]) -> List[ConstitutionDocument]:
    """Convert loaded metadata into a list of ConstitutionDocument objects."""
    documents = []
    for i, text in enumerate(data):
        try:
            documents.append(
                ConstitutionDocument(
                    content=text,
                    metadata={"source": f"document_{i}"},
                    doc_id=i
                )
            )
        except Exception as e:
            raise ValueError(f"Error creating document {i}: {str(e)}")
    
    return documents


def configure_vectorstore(documents: List[ConstitutionDocument], embeddings) -> FAISS:
    """Create and save FAISS vectorstore."""
    vectorstore_instance = FAISS.from_documents(documents, embeddings)
    vectorstore_instance.save_local("faiss_index")
    return vectorstore_instance


@app.on_event("startup")
async def initialize_app():
    """Initialize the application, loading models and data."""
    global llm, embeddings, vectorstore
    llm, embeddings = initialize_models()
    data = load_json_metadata()
    documents = build_documents(data)
    vectorstore = configure_vectorstore(documents, embeddings)


def generate_response_with_retrieval_chain(vectorstore, llm, embeddings, user_input: str) -> str:
    """Generate response using the retrieval chain."""

    prompt_template = """
    You are a knowledgeable assistant, well-versed in the Indian Constitution and able to answer user queries accurately and helpfully. Use the provided context when it is directly relevant to the question. If a question is not directly answered in the context but is commonly known (such as general information about the Constitution), answer it informatively.

    Below are some relevant sections from the Constitution:

    <context>
    {context}
    </context>

    Follow these response guidelines:

    1. **If the user greets you (e.g., 'Hi', 'Hello')**:
       - Respond with a friendly message such as: 
         "Hello! I'm here to help answer questions about the Indian Constitution. Feel free to ask about articles, amendments, or topics within the Constitution."

    2. **If the user asks for assistance (e.g., 'Can you help me?')**:
       - Reply supportively, like this: 
         "Certainly! I’m here to assist with questions about the Indian Constitution. You can ask about articles, rights, duties, or amendments. How can I help?"

    3. **If the user asks a general question about the Indian Constitution (e.g., '1.What is the Indian Constitution?','2.What does Article 31A of the Indian Constitution cover?')**:
       - Provide a brief, informative answer. For example: 
         "1.The Indian Constitution is the supreme law of India, laying out the framework for the organization, functions, and duties of government institutions, as well as the fundamental rights and duties of citizens.",
         "2.The Indian Constitution is the supreme legal document of India, and Article 31A protects land reform laws and state acquisitions from being challenged on the grounds of violating fundamental rights."

    4. **If the user asks a specific question about the Indian Constitution that relates to the provided context**:
       - Answer based on the relevant sections in the provided context, giving a detailed response. If the answer cannot be found in the provided context, let the user know by saying, "I'm sorry, but the answer cannot be directly found in the provided context."

    **User Input**: {input}

    **Response**:
    """


    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": user_input})
    return response['answer']




class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


@app.get("/")
async def get_homepage():
    """Serve the chatbot interface HTML page."""
    return FileResponse("static/index.html")


@app.post("/chatbot", response_model=ChatResponse)
async def respond_to_chat_request(request: ChatRequest):
    """Handle incoming chatbot messages."""
    user_message = request.message
    response_text = generate_response_with_retrieval_chain(vectorstore, llm, embeddings, user_message)
    return JSONResponse(content={"reply": response_text})
