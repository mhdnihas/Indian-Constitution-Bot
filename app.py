from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFDirectoryLoader

app = FastAPI()

@app.post("/load_data")
def load_data():
    try:
        directory_path = "Indian_Constitution.pdf"
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()
        return {"message": f"Loaded {len(docs)} documents successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
