import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Globals ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
INDEX = None
DOC_CHUNKS = []
CHUNK_MAP = {}

# --- Utils ---
def load_and_split(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "html":
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.split_documents(docs)

def build_faiss_index(chunks):
    embeddings = EMBEDDING_MODEL.encode([c.page_content for c in chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def search(query, k=5, mode="semantic"):
    if INDEX is None or not DOC_CHUNKS:
        return []
    if mode == "semantic":
        q_emb = EMBEDDING_MODEL.encode([query])
        D, I = INDEX.search(np.array(q_emb).astype("float32"), k)
        return [DOC_CHUNKS[i].page_content for i in I[0]]
    elif mode == "keyword":
        results = []
        for c in DOC_CHUNKS:
            if query.lower() in c.page_content.lower():
                results.append(c.page_content)
        return results[:k]
    else:
        return []

# --- API Models ---
class QueryRequest(BaseModel):
    query: str
    mode: str = "semantic"

# --- Endpoints ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["pdf", "html", "htm"]:
        return JSONResponse({"error": "Only PDF and HTML supported"}, status_code=400)
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    global DOC_CHUNKS, INDEX, CHUNK_MAP
    DOC_CHUNKS = load_and_split(tmp_path, "pdf" if ext == "pdf" else "html")
    INDEX, _ = build_faiss_index(DOC_CHUNKS)
    CHUNK_MAP = {i: c.page_content for i, c in enumerate(DOC_CHUNKS)}
    os.unlink(tmp_path)
    return {"status": "uploaded", "chunks": len(DOC_CHUNKS)}

@app.post("/query")
async def query(req: QueryRequest):
    results = search(req.query, k=5, mode=req.mode)
    async def streamer():
        for chunk in results:
            yield chunk + "\n---\n"
    return StreamingResponse(streamer(), media_type="text/plain")

@app.get("/")
def root():
    return HTMLResponse("""
    <html>
    <head>
    <title>RAG Demo</title>
    </head>
    <body>
    <h2>RAG Demo Backend Running</h2>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
