from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.genai import types
from agent import app as agent_app

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request shape
class ResearchRequest(BaseModel):
    question: str

# Response shape
class ResearchResponse(BaseModel):
    answer: str
    sources: list[str]
    search_count: int

@api.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest):
    result = agent_app.invoke({
        "question": request.question,
        "messages": [types.Content(role="user", parts=[types.Part(text=request.question)])],
        "search_count": 0,
        "sources": [],
        "final_answer": ""
    })

    return ResearchResponse(
        answer=result["final_answer"],
        sources=result["sources"],
        search_count=result["search_count"]
    )

@api.get("/health")
async def health():
    return {"status": "ok"}