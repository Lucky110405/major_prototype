from fastapi import APIRouter, Query, Depends, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

class VisualModel(BaseModel):
    insights: List[str]
    chunks: List[Dict[str, Any]]

class ChunksModel(BaseModel):
    chunks: List[Dict[str, Any]]
    intent: str

def get_orchestrator(request: Request):
    return request.app.state.orchestrator

def get_text_embedder(request: Request):
    return request.app.state.text_embedder

def get_intent_agent(request: Request):
    return request.app.state.orchestrator.intent_agent

def get_retriever_agent(request: Request):
    return request.app.state.orchestrator.retriever_agent

def get_analyzer_agent(request: Request):
    return request.app.state.orchestrator.analyzer_agent

def get_visual_agent(request: Request):
    return request.app.state.orchestrator.visual_agent

def get_modality_agent(request: Request):
    return request.app.state.modality_agent

@router.get("/run")
def run_agents_endpoint(q: str = Query(...),
                        orchestrator=Depends(get_orchestrator),
                        text_embedder=Depends(get_text_embedder)):
    """
    Run the full agentic workflow.
    """
    # Generate query vector
    query_vector = text_embedder.embed([q])[0] if text_embedder else []
    try:
        result = orchestrator.run_workflow(q, query_vector)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/intent")
def classify_intent_endpoint(q: str = Query(...),
                             intent_agent=Depends(get_intent_agent)):
    """
    Classify the intent of the query.
    """
    try:
        result = intent_agent.run(q)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/retrieve")
def retrieve_chunks_endpoint(q: str = Query(...), intent: str = Query("descriptive"),
                             retriever_agent=Depends(get_retriever_agent),
                             text_embedder=Depends(get_text_embedder)):
    """
    Retrieve relevant chunks for the query.
    """
    try:
        query_vector = text_embedder.embed([q])[0]
        result = retriever_agent.run(q, query_vector, intent=intent)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/analyze")
def analyze_chunks_endpoint(data: ChunksModel,
                            analyzer_agent=Depends(get_analyzer_agent)):
    """
    Analyze retrieved chunks and generate insights.
    """
    try:
        result = analyzer_agent.run(data.chunks, data.intent)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/visual")
def generate_visualizations_endpoint(data: VisualModel,
                                     visual_agent=Depends(get_visual_agent)):
    """
    Generate visualizations from insights and chunks.
    """
    try:
        result = visual_agent.run(data.insights, data.chunks)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/select_model")
def select_model_endpoint(task: str = Query(...), modality: str = Query(...), constraints: Optional[str] = Query(None),
                          modality_agent=Depends(get_modality_agent)):
    """
    Select a model for a given task and modality.
    """
    try:
        cons = eval(constraints) if constraints else None  # Simple eval, in prod use proper parsing
        model = modality_agent.select_model(task, modality, cons)
        return {"selected_model": model}
    except Exception as e:
        return {"status": "error", "message": str(e)}