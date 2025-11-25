from fastapi import FastAPI
from api.routes import ingest, query, agents

app = FastAPI(title="Collaborative Multi-Modal Agentic BI Framework", version="1.0.0")

# Include routers
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the BI Framework API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)