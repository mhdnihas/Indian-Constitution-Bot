from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI()

# Serve static files (e.g., index.html, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the request and response models
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
    bot_reply = f"You said: {user_message}"
    return JSONResponse(content={"reply": bot_reply})
