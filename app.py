from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from huggingface_hub import hf_hub_download
import faiss


import requests
import os


import sys
import site

print("sys path:",sys.path)

load_dotenv()


     

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

HuggingFace_Token_API = os.getenv('HuggingFace_Token_API')


model = genai.GenerativeModel("gemini-pro")
prompt = "What is Article 19 of the Indian Constitution?"

# Generate a response
response = model.generate_content("Write a story about a magic backpack.")


index_file = hf_hub_download(
    repo_id="your-username/your-dataset-repo",
    filename="faiss_index_file.index",
    repo_type="dataset"
)


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")






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
    
    
    response = model.generate_content(user_message)

    print("hai")
    return JSONResponse(content={"reply": response.text})
