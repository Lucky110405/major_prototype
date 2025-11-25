from fastapi import APIRouter, Query, Depends

router = APIRouter()

def get_orchestrator(request):
    return request.app.state.orchestrator

def get_text_embedder(request):
    return request.app.state.text_embedder

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