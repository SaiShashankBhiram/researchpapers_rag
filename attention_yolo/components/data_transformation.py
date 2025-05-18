import os
import pickle
import sys
import json
from uuid import uuid4
import boto3
from pymongo import MongoClient
from attention_yolo.logger import logger  # Using the logger module
from urllib.parse import quote_plus
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter  # New splitter
from attention_yolo.exception import CustomException  # Import CustomException

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "../data/documents.pkl")

# Step 1: Load documents from the pickle file
try:
    with open(file_path, "rb") as f:
        documents = pickle.load(f)
    logger.info(f"📂 Loaded {len(documents)} documents from pickle file.")
except Exception as e:
    logger.error(f"❌ Error loading pickle file: {e}")
    raise CustomException(e, sys)

# Step 2: Initialize Bedrock client
try:
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = "amazon.titan-embed-text-v2:0"
except Exception as e:
    logger.error(f"❌ Error initializing Bedrock client: {e}")
    raise CustomException(e, sys)

# Step 3: Function to generate embeddings (retry mechanism added)
def generate_embedding(text, retries=3):
    try:
        if len(text) > 50000:
            logger.warning(f"⚠️ Skipping overly long chunk ({len(text)} characters).")
            return None
        
        for attempt in range(retries):
            try:
                response = bedrock_runtime.invoke_model(
                    body=json.dumps({"inputText": text}),
                    modelId=model_id,
                    accept="application/json",
                    contentType="application/json"
                )
                response_body = json.loads(response["body"].read())
                return response_body["embedding"]
            except Exception as e:
                logger.error(f"Error generating embedding (Attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    raise CustomException(e, sys)
    except Exception as e:
        logger.error(f"❌ Unexpected error in embedding generation: {e}")
        raise CustomException(e, sys)

# Step 4: Initialize RecursiveCharacterTextSplitter (New Chunking Strategy)
try:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", ".", " "],  # Preferred split order
        chunk_size=1000,  # Max characters per chunk
        chunk_overlap=100  # Preserve context by overlapping chunks
    )
except Exception as e:
    logger.error(f"❌ Error initializing text splitter: {e}")
    raise CustomException(e, sys)

# Step 5: Generate chunks for each document using RecursiveCharacterTextSplitter
chunked_documents = []
try:
    for doc in documents:
        text = doc["content"]
        doc_chunks = text_splitter.split_text(text)

        for chunk in doc_chunks:
            embedding = generate_embedding(chunk)  # Generate vector embeddings
            if embedding:
                chunked_documents.append({
                    "id": str(uuid4()),
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": doc["metadata"]
                })

    logger.info(f"🧩 Created {len(chunked_documents)} text chunks using RecursiveCharacterTextSplitter.")
except Exception as e:
    logger.error(f"❌ Error in chunking documents: {e}")
    raise CustomException(e, sys)

# Step 6: Connect to MongoDB
try:
    username = quote_plus("saishashankbhiram")
    password = quote_plus("Admin123")
    mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.o0y1c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    client = MongoClient(mongo_uri)
    db = client["rag_app_db"]
    collection = db["vector_embeddings"]
except Exception as e:
    logger.error(f"❌ Error connecting to MongoDB: {e}")
    raise CustomException(e, sys)

# Step 7: Insert chunks with embeddings into MongoDB
try:
    collection.insert_many(chunked_documents, ordered=False)
    logger.info("✅ Vector embeddings saved to MongoDB.")
except Exception as e:
    logger.error(f"❌ MongoDB insertion failed: {e}")
    raise CustomException(e, sys)