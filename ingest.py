# ingest.py
import os
import logging
import sys
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # Optional: for Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional: for Qdrant Cloud

# Check for required API keys
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")

# PDF and Qdrant settings
PDF_PATH = "/home/ganesh-pathivada/Downloads/attention.pdf"  # IMPORTANT: Make sure this path is correct
COLLECTION_NAME = "attention-pdf-index"
EMBEDDING_DIM = 768  # Correct dimension for "models/embedding-001"

def main():
    """
    Main function to process PDF, create embeddings, and store them in Qdrant.
    """
    logging.info("--- Starting Ingestion Process ---")

    # --- 1. Configure LlamaIndex Global Settings ---
    logging.info("Configuring LlamaIndex settings with Gemini models...")
    Settings.llm = Gemini(
        model="models/gemini-1.5-flash",
        api_key=GOOGLE_API_KEY
    )
    Settings.embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=GOOGLE_API_KEY
    )
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 200
    logging.info("Settings configured.")

    # --- 2. Load Documents ---
    try:
        logging.info(f"Loading documents from: {PDF_PATH}")
        documents = SimpleDirectoryReader(input_files=[PDF_PATH]).load_data()
        if not documents:
            logging.error("No documents were loaded. Check the PDF path and file content.")
            return
        logging.info(f"Successfully loaded {len(documents)} document chunk(s).")
    except Exception as e:
        logging.error(f"Error loading the PDF file: {e}")
        logging.error(f"Please check that the path is correct: {PDF_PATH}")
        return

    # --- 3. Initialize Qdrant Client ---
    logging.info("Initializing Qdrant client...")
    
    # Option 1: Local Qdrant (default)
    if not QDRANT_URL:
        logging.info("Using local Qdrant instance...")
        client = QdrantClient(host="localhost", port=6333)
    # Option 2: Qdrant Cloud
    else:
        logging.info("Using Qdrant Cloud...")
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    # --- 4. Create Qdrant Collection if it doesn't exist ---
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            logging.info(f"Creating new Qdrant collection: {COLLECTION_NAME}")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            logging.info("Collection created successfully.")
        else:
            logging.info(f"Qdrant collection '{COLLECTION_NAME}' already exists. Will add/update vectors.")
    except Exception as e:
        logging.error(f"Error creating/checking collection: {e}")
        return

    # --- 5. Setup LlamaIndex Vector Store and Storage Context ---
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- 6. Create Index and Store Embeddings ---
    logging.info("Creating index and storing embeddings in Qdrant... This may take a moment.")
    try:
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        logging.info("--- Finished Ingestion Process ---")
    except Exception as e:
        logging.error(f"Error during indexing: {e}")
        return

if __name__ == "__main__":
    main()