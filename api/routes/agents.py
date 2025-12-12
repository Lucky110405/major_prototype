from fastapi import APIRouter, Query, Depends, Request
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class VisualModel(BaseModel):
    insights: List[str]
    chunks: List[Dict[str, Any]]

class ChunksModel(BaseModel):
    chunks: List[Dict[str, Any]]
    intent: str


class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str


class ConversationCreate(BaseModel):
    title: Optional[str] = None


class RunRequest(BaseModel):
    q: str
    conversation_id: Optional[str] = None


class MessageGenerate(BaseModel):
    conversation_id: Optional[str] = None
    content: str

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

def get_chat_agent(request: Request):
    return request.app.state.chat_agent

@router.get("/run")
def run_agents_endpoint(q: str = Query(...), conversation_id: Optional[str] = Query(None),
                        orchestrator=Depends(get_orchestrator),
                        text_embedder=Depends(get_text_embedder)):
    """
    Run the full agentic workflow.
    """
    # Generate query vector
    query_vector = text_embedder.embed([q])[0] if text_embedder else []
    try:
        result = orchestrator.run_workflow(q, query_vector, conversation_id=conversation_id)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/run")
def run_agents_endpoint_post(payload: RunRequest,
                             orchestrator=Depends(get_orchestrator),
                             text_embedder=Depends(get_text_embedder)):
    q = payload.q
    # For now we don't use conversation_id in the workflow, but accept it for compatibility
    try:
        query_vector = text_embedder.embed([q])[0] if text_embedder else []
        result = orchestrator.run_workflow(q, query_vector, conversation_id=payload.conversation_id)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post('/run/stream')
def run_agents_stream(payload: RunRequest,
                      orchestrator=Depends(get_orchestrator),
                      text_embedder=Depends(get_text_embedder)):
    q = payload.q
    conversation_id = payload.conversation_id
    query_vector = text_embedder.embed([q])[0] if text_embedder else []

    def event_generator():
        for event in orchestrator.run_workflow_stream(q, query_vector, conversation_id=conversation_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')

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
                            conversation_id: Optional[str] = Query(None),
                            orchestrator=Depends(get_orchestrator),
                            analyzer_agent=Depends(get_analyzer_agent)):
    """
    Analyze retrieved chunks and generate insights.
    """
    try:
        conv_msgs = []
        if conversation_id:
            try:
                conv_msgs = orchestrator.get_messages(conversation_id)
            except Exception:
                conv_msgs = []
        result = analyzer_agent.run(data.chunks, data.intent, conversation_messages=conv_msgs, conversation_id=conversation_id)
        # Attach conversation id in response
        if conversation_id and isinstance(result, dict):
            result['conversation_id'] = conversation_id
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


@router.get("/conversations")
def list_conversations(orchestrator=Depends(get_orchestrator)):
    try:
        convs = orchestrator.get_all_conversations()
        return {"conversations": convs}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/conversations")
def create_conversation(payload: Optional[ConversationCreate] = None, title: Optional[str] = Query(None), orchestrator=Depends(get_orchestrator)):
    try:
        conv_title = (payload.title if payload and payload.title else title) if payload or title else None
        conv = orchestrator.create_conversation(conv_title)
        # Normalize conv shape for consistent client use
        def try_get_id(obj):
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj
            if isinstance(obj, dict):
                if obj.get('id'):
                    return obj.get('id')
                if obj.get('conversation_id'):
                    return obj.get('conversation_id')
                if obj.get('data'):
                    return try_get_id(obj.get('data'))
                for v in obj.values():
                    try:
                        found = try_get_id(v)
                        if found:
                            return found
                    except Exception:
                        continue
            return None

        conv_id = try_get_id(conv)
        if isinstance(conv, dict) and not conv.get('id') and conv_id:
            conv['id'] = conv_id
        if not conv_id:
            # Strong fallback: generate id string based on title and return.
            # This only happens in rare cases where Supabase/SDK returns an unexpected object.
            # Avoid raising NameError (e.g., missing logger) — we have logger above.
            try:
                logger.warning('create_conversation: No id found in result from orchestrator', extra={'raw': conv})
            except Exception:
                # If logging with 'extra' triggers an issue (e.g., conv is not serializable), log basic info
                logger.warning('create_conversation: No id found in result from orchestrator (conv not serializable)')
            # Always ensure the route returns a plain dict with 'id'
            conv = {'id': f'local-fallback-{len(orchestrator._conversations)+1}', 'title': conv_title or 'New Conversation', 'created_at': __import__('datetime').datetime.utcnow().isoformat()}
        return {"conversation": conv}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/messages")
def list_messages(conversation_id: Optional[str] = Query(None), orchestrator=Depends(get_orchestrator)):
    try:
        if not conversation_id:
            return {"messages": []}
        msgs = orchestrator.get_messages(conversation_id)
        return {"messages": msgs}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/messages")
def create_message(payload: MessageCreate, orchestrator=Depends(get_orchestrator)):
    try:
        msg = orchestrator.create_message(payload.conversation_id, payload.role, payload.content)
        return {"message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/messages/generate")
def generate_message(payload: MessageGenerate, orchestrator=Depends(get_orchestrator), chat_agent=Depends(get_chat_agent)):
    try:
        result = chat_agent.run(payload.conversation_id, payload.content)
        return {
            'conversation_id': result.get('conversation_id'),
            'assistant_message': result.get('assistant_message'),
            'workflow_result': result.get('workflow_result')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post('/messages/generate/stream')
def generate_message_stream(payload: MessageGenerate, orchestrator=Depends(get_orchestrator), chat_agent=Depends(get_chat_agent), text_embedder=Depends(get_text_embedder)):
    try:
        # Ensure conversation exists
        conversation_id = payload.conversation_id
        if not conversation_id:
            conv = orchestrator.create_conversation(None)
            conversation_id = conv.get('id') if isinstance(conv, dict) else conv

        # Create user message
        try:
            user_msg = orchestrator.create_message(conversation_id, 'user', payload.content)
        except Exception:
            user_msg = None

        # Embed and start streaming events
        query_vector = text_embedder.embed([payload.content])[0] if text_embedder else []

        def event_gen():
            for event in orchestrator.run_workflow_stream(payload.content, query_vector, conversation_id=conversation_id):
                # If this is the final event, persist assistant message and include it in the final payload
                if event.get('type') == 'final':
                    final_output = event.get('result', {}).get('final_output') or ''
                    try:
                        assistant_msg = orchestrator.create_message(conversation_id, 'assistant', final_output)
                    except Exception:
                        assistant_msg = None
                    # Attach assistant_msg and conversation_id
                    event['assistant_message'] = assistant_msg
                    event['conversation_id'] = conversation_id
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_gen(), media_type='text/event-stream')
    except Exception as e:
        return {"status": "error", "message": str(e)}