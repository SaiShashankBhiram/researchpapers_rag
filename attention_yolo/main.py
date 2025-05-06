from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from attention_yolo.data_retreiver import query_rag

app = FastAPI()

# Default top_k value
TOP_K = 3  # 👈 You can change this globally

# Input schema
class QueryRequest(BaseModel):
    question: str

# Health check
@app.get("/")
def health_check():
    return {"status": "✅ API is up and running"}

# Main query endpoint
@app.post("/query")
def run_query(request: QueryRequest):
    try:
        result = query_rag(request.question, top_k=TOP_K)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
