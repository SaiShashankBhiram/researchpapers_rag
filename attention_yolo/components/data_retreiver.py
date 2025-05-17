import json
import sys
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
from attention_yolo.logger import logger
from attention_yolo.exception import CustomException  # Import CustomException

from langchain.prompts import PromptTemplate
from attention_yolo.components.data_transformation import bedrock_runtime, generate_embedding

# Step 1: MongoDB Connection
try:
    username = quote_plus("saishashankbhiram")
    password = quote_plus("Admin123")
    mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.o0y1c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    client = MongoClient(mongo_uri)
    db = client["rag_app_db"]
    collection = db["vector_embeddings"]
    logger.info("✅ Connected to MongoDB successfully.")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")
    raise CustomException(e, sys)

# Step 2: Prompt Template
prompt_template = """
Human: Use the following context to answer the question in 250 words with a detailed explanation.
If you don't know the answer, just say you don't know — don't guess.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Step 3: Call LLaMA-3 on Bedrock
def get_llama_llm(prompt: str):
    try:
        formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
        }

        response = bedrock_runtime.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(native_request),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["generation"]
    
    except Exception as e:
        logger.error(f"❌ Error during model invocation: {e}")
        raise CustomException(e, sys)

# Step 4: Main RAG Function
def query_rag(question: str, top_k: int = 3):
    try:
        # Step 4.1: Generate embedding for the input question
        question_embedding = generate_embedding(question)
        if question_embedding is None:
            logger.error("❌ Failed to generate embedding for the query.")
            return "❌ Failed to generate embedding for the query."

        question_embedding_np = np.array(question_embedding).reshape(1, -1)

        # Step 4.2: Retrieve all documents from MongoDB
        all_docs = list(collection.find({}, {"content": 1, "embedding": 1}))

        # Step 4.3: Compute cosine similarity
        similarities = []
        for doc in all_docs:
            try:
                doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
                similarity = cosine_similarity(question_embedding_np, doc_embedding)[0][0]
                similarities.append((similarity, doc["content"]))
            except Exception as e:
                logger.error(f"❌ Error computing similarity for document: {e}")
                raise CustomException(e, sys)

        # Step 4.4: Sort by similarity and get top_k documents
        top_docs = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
        context = "\n\n".join([doc[1] for doc in top_docs])

        # Step 4.5: Format and query the LLM
        full_prompt = PROMPT.format(context=context, question=question)
        return get_llama_llm(full_prompt)
    
    except Exception as e:
        logger.error(f"❌ Error in querying MongoDB or Bedrock: {e}")
        raise CustomException(e, sys)