from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import ingest, query, agents, documents

app = FastAPI(title="Collaborative Multi-Modal Agentic BI Framework", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(documents.router, tags=["documents"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the BI Framework API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)