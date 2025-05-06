import json
import numpy as np
from pymongo import MongoClient
from urllib.parse import quote_plus
from sklearn.metrics.pairwise import cosine_similarity
from attention_yolo.logger import logger

from langchain.prompts import PromptTemplate
from attention_yolo.data_transformation import bedrock_runtime, generate_embedding
# MongoDB connection
username = quote_plus("saishashankbhiram")
password = quote_plus("Admin123")
mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.o0y1c.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

try:
    client = MongoClient(mongo_uri)
    db = client["rag_app_db"]
    collection = db["vector_embeddings"]
    logger.info("✅ Connected to MongoDB successfully.")
except Exception as e:
    logger.error(f"MongoDB connection failed: {e}")
    raise RuntimeError("Could not connect to MongoDB")

# Prompt Template
prompt_template = """
Human: Use the following context to answer the question in 250 words with a detailed explanation. 
If you don't know the answer, just say you don't know — don't guess.

<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


# Call LLaMA-3 on Bedrock
def get_llama_llm(prompt: str):
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

    try:
        response = bedrock_runtime.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=json.dumps(native_request),
            contentType="application/json",
            accept="application/json"
        )
        result = json.loads(response["body"].read())
        return result["generation"]
    except Exception as e:
        return f"Error during model invocation: {e}"


# Main RAG function using MongoDB instead of FAISS
def query_rag(question: str, top_k: int = 3):
    try:
        # Step 1: Generate embedding for the input question
        question_embedding = generate_embedding(question)
        if question_embedding is None:
            return "❌ Failed to generate embedding for the query."

        question_embedding_np = np.array(question_embedding).reshape(1, -1)

        # Step 2: Retrieve all documents from MongoDB
        all_docs = list(collection.find({}, {"content": 1, "embedding": 1}))

        # Step 3: Compute cosine similarity
        similarities = []
        for doc in all_docs:
            doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
            similarity = cosine_similarity(question_embedding_np, doc_embedding)[0][0]
            similarities.append((similarity, doc["content"]))

        # Step 4: Sort by similarity and get top_k
        top_docs = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]
        context = "\n\n".join([doc[1] for doc in top_docs])

        # Step 5: Format and query the LLM
        full_prompt = PROMPT.format(context=context, question=question)
        return get_llama_llm(full_prompt)

    except Exception as e:
        return f"Error in querying MongoDB or Bedrock: {e}"
