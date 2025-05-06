import os
import pickle
import logging
from pymongo import MongoClient
from urllib.parse import quote_plus
from logger import logger  # Importing your logger setup

# MongoDB connection details
def ingest_from_mongodb():
    logger.info("🔌 Connecting to MongoDB...")

    # MongoDB connection credentials (you can adjust this to be more secure)
    username = quote_plus("saishashankbhiram")
    password = quote_plus("Admin123")
    uri = f"mongodb+srv://{username}:{password}@cluster0.o0y1c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    try:
        client = MongoClient(uri)
        db = client["rag_app_db"]
        collection = db["pdf_documents"]

        logger.info("📥 Fetching documents from MongoDB...")
        docs_cursor = collection.find({})
        documents = []

        # Iterate over MongoDB documents and collect valid ones
        for i, doc in enumerate(docs_cursor, start=1):
            text = doc.get("content", "")
            metadata = doc.get("metadata", {})
            if text.strip():
                logger.info(f"✅ Document #{i} retrieved, length: {len(text)} chars")
                documents.append({
                    "content": text,
                    "metadata": metadata
                })
            else:
                logger.warning(f"⚠️ Skipped empty document #{i}")

        logger.info(f"📄 Total valid documents retrieved: {len(documents)}")

        # Save the documents to a pickle file
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        DATA_DIR = os.path.join(BASE_DIR, "data")
        os.makedirs(DATA_DIR, exist_ok=True)

        with open(os.path.join(DATA_DIR, "E:/ML projects/ragec2/data/documents.pkl"), "wb") as f:
            pickle.dump(documents, f)
        logger.info("📦 Documents saved to 'data/documents.pkl'")

    except Exception as e:
        logger.error(f"❌ Error during MongoDB ingestion: {e}")

if __name__ == "__main__":
    ingest_from_mongodb()
