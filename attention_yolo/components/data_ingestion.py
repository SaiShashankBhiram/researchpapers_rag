import os
import pickle
import sys
import logging
from pymongo import MongoClient
from urllib.parse import quote_plus
from logger import logger  # Importing your logger setup
from attention_yolo.exception import CustomException  # Import CustomException

# MongoDB connection details
def ingest_from_mongodb():
    logger.info("🔌 Connecting to MongoDB...")

    # MongoDB connection credentials (consider storing securely in env variables)
    try:
        username = quote_plus("saishashankbhiram")
        password = quote_plus("Admin123")
        uri = f"mongodb+srv://{username}:{password}@cluster0.o0y1c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        
        client = MongoClient(uri)
        db = client["rag_app_db"]
        collection = db["pdf_documents"]

        logger.info("📥 Fetching documents from MongoDB...")
        docs_cursor = collection.find({})
        documents = []

        # Iterate over MongoDB documents and collect valid ones
        for i, doc in enumerate(docs_cursor, start=1):
            try:
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
            except Exception as doc_error:
                logger.error(f"❌ Error processing document #{i}: {doc_error}")
                raise CustomException(doc_error, sys)

        logger.info(f"📄 Total valid documents retrieved: {len(documents)}")

        # Save the documents to a pickle file
        try:
            BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            DATA_DIR = os.path.join(BASE_DIR, "data")
            os.makedirs(DATA_DIR, exist_ok=True)

            with open(os.path.join(DATA_DIR, "documents.pkl"), "wb") as f:
                pickle.dump(documents, f)
            
            logger.info("📦 Documents saved to 'data/documents.pkl'")
        
        except Exception as file_error:
            logger.error(f"❌ Error saving documents: {file_error}")
            raise CustomException(file_error, sys)

    except Exception as db_error:
        logger.error(f"❌ Error during MongoDB ingestion: {db_error}")
        raise CustomException(db_error, sys)

if __name__ == "__main__":
    try:
        ingest_from_mongodb()
    except CustomException as ce:
        logger.error(f"🚨 Critical Failure: {ce}")