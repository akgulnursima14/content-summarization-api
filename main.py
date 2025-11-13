from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from groq import Groq

app = FastAPI(
    title="content summarization API",
    description="I need an API that summarizes comment content for my e-commerce platform.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Request Model
class AnalyzeRequest(BaseModel):
    text: str
    
# Response Model
class AnalyzeResponse(BaseModel):
    result: str
    confidence: float

@app.get("/")
async def root():
    return {
        "name": "content summarization API",
        "status": "active",
        "service": "content-summarization"
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    content-summarization endpoint
    """
    try:
        # Groq API call
        completion = client.chat.completions.create(
            model="mixtral-8x7b",
            messages=[
                {
                    "role": "system",
                    "content": "Sen bir content-summarization uzmanısın. Kısa ve net cevap ver."
                },
                {
                    "role": "user",
                    "content": f"Metni analiz et: {request.text}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        result = completion.choices[0].message.content
        
        return AnalyzeResponse(
            result=result,
            confidence=0.95
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
