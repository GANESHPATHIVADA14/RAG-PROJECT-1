import os
import logging
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Check for API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")

# Constants
COLLECTION_NAME = "attention-pdf-index"

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FastAPI App Setup ---
app = FastAPI(
    title="Attention PDF RAG API",
    description="A backend for querying attention.pdf using LlamaIndex, Gemini, and Qdrant.",
    version="1.0.0",
)

# CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str

# Global variable to hold the query engine
query_engine = None

@app.on_event("startup")
def startup_event():
    """
    Initialize the RAG system on server startup.
    """
    global query_engine
    logging.info("--- Server starting up: Initializing Query Engine ---")

    # --- 1. Configure LlamaIndex Global Settings ---
    logging.info("Configuring LlamaIndex settings with Gemini models...")
    Settings.llm = Gemini(model_name="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
    logging.info("Settings configured.")

    # --- 2. Initialize Qdrant and connect to the existing collection ---
    logging.info(f"Connecting to Qdrant collection: '{COLLECTION_NAME}'")
    try:
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION_NAME
        )
        
        # --- 3. Load the index from the vector store ---
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        logging.info("Successfully loaded index from Qdrant.")
        
        # --- 4. Create the Query Engine ---
        query_engine = index.as_query_engine(similarity_top_k=5)
        logging.info("✅ Query engine is ready!")

    except Exception as e:
        logging.error(f"Failed to initialize query engine: {e}")
        raise RuntimeError("Could not initialize the query engine. Please check logs.")

@app.get("/query")
async def query_endpoint(q: str):
    """
    GET endpoint to query the attention.pdf document.
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="Query engine is not available.")
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty.")

    logging.info(f"Received query: {q}")
    
    try:
        response = query_engine.query(q)
        return {"query": q, "answer": str(response)}
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Error processing query.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
