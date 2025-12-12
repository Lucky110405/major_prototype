import logging
import os
from typing import Dict, Any, List, Optional, Generator
import json
from supabase import create_client, Client
from agents.intent_agent import IntentAgent
from agents.retriever_agent import RetrieverAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.visual_agent import VisualAgent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, intent_agent: IntentAgent, retriever_agent: RetrieverAgent, 
                 analyzer_agent: AnalyzerAgent, visual_agent: VisualAgent):
        """
        Initialize the Orchestrator.
        
        Args:
            intent_agent: Instance of IntentAgent.
            retriever_agent: Instance of RetrieverAgent.
            analyzer_agent: Instance of AnalyzerAgent.
            visual_agent: Instance of VisualAgent.
        """
        self.intent_agent = intent_agent
        self.retriever_agent = retriever_agent
        self.analyzer_agent = analyzer_agent
        self.visual_agent = visual_agent
        
        # Initialize Supabase for conversations and messages
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        if self.supabase_url and self.supabase_key:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        else:
            self.supabase = None
        # Health flag for Supabase to avoid repeated failed requests
        self.supabase_ready = False if not self.supabase else True
        if self.supabase:
            try:
                # Try a simple select to ensure the client and tables exist
                _ = self.supabase.table('conversations').select('*').limit(1).execute()
                self.supabase_ready = True
            except Exception as e:
                logger.warning(f"Supabase health check failed: {e}. Falling back to in-memory.")
                self.supabase_ready = False
        # In-memory fallback store for conversations and messages when Supabase is not configured
        self._conversations = []
        self._messages = []
        # Track whether we've already emitted supabase warnings per table to avoid log spam
        self._supabase_warned = {'conversations': False, 'messages': False}

    def _maybe_warn(self, table_name: str, message: str):
        if not self._supabase_warned.get(table_name, False):
            logger.warning(message)
            self._supabase_warned[table_name] = True

    def _safe_insert(self, table_name: str, data: Dict[str, Any], single: bool = True) -> Dict[str, Any]:
        """
        Attempt to insert into Supabase with compatibility across client versions.
        Falls back to plain insert or to in-memory store on failure.
        Returns a dict representing the inserted row.
        """
        # No supabase configured or not ready -> fallback in-memory
        if not self.supabase or not self.supabase_ready:
            if table_name == 'conversations':
                conv = {'id': f'local-conv-{len(self._conversations)+1}', 'title': data.get('title'), 'created_at': __import__('datetime').datetime.utcnow().isoformat()}
                self._conversations.append(conv)
                return conv
            else:
                msg = {'id': f'local-msg-{len(self._messages)+1}', 'conversation_id': data.get('conversation_id'), 'role': data.get('role'), 'content': data.get('content'), 'created_at': __import__('datetime').datetime.utcnow().isoformat()}
                self._messages.append(msg)
                return msg
        # Try supabase insert with select if builder supports it
        try:
            table_builder = getattr(self.supabase, 'table')(table_name)
            insert_builder = table_builder.insert(data)
            # If insert builder supports select() chaining, use it
            if hasattr(insert_builder, 'select'):
                builder = insert_builder.select('*')
                if single and hasattr(builder, 'single'):
                    builder = builder.single()
                response = builder.execute()
            else:
                # plain execute
                response = insert_builder.execute()
            # Normalize response
            d = getattr(response, 'data', response)
            if isinstance(d, list) and len(d) > 0:
                return d[0]
            return d
        except Exception as e:
            # Only warn once per table_name to avoid log spam
            self._maybe_warn(table_name, f"Supabase insert failing for {table_name}: {e}. Falling back to simple insert or in-memory.")
            # Try fallback to plain insert
            try:
                resp = getattr(self.supabase, 'table')(table_name).insert(data).execute()
                d = getattr(resp, 'data', resp)
                if isinstance(d, list) and len(d) > 0:
                    return d[0]
                return d
            except Exception as e2:
                # On serious error, fallback to in-memory store
                self._maybe_warn(table_name, f"Supabase hard failure for {table_name}: {e2}. Using in-memory fallback.")
                if table_name == 'conversations':
                    conv = {'id': f'local-conv-{len(self._conversations)+1}', 'title': data.get('title'), 'created_at': __import__('datetime').datetime.utcnow().isoformat()}
                    self._conversations.append(conv)
                    return conv
                else:
                    msg = {'id': f'local-msg-{len(self._messages)+1}', 'conversation_id': data.get('conversation_id'), 'role': data.get('role'), 'content': data.get('content'), 'created_at': __import__('datetime').datetime.utcnow().isoformat()}
                    self._messages.append(msg)
                    return msg

    def run_workflow(self, user_query: str, query_vector: List[float], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the full agentic workflow.
        
        Args:
            user_query: The user's query.
            query_vector: Embedding vector for the query.
        
        Returns:
            Final report with all components.
        """
        try:
            # Step 1: Classify intent
            intent_result = self.intent_agent.run(user_query)
            intent = intent_result["intent"]
            logger.info(f"Workflow step 1: Intent classified as {intent}")
            
            # Step 2: Retrieve relevant chunks
            retrieval_result = self.retriever_agent.run(user_query, query_vector, intent=intent)
            chunks = retrieval_result["chunks"]
            logger.info(f"Workflow step 2: Retrieved {len(chunks)} chunks")
            
            # If a conversation id is provided and supabase is configured, fetch conversation messages for context
            conversation_messages = []
            if conversation_id and self.supabase and self.supabase_ready:
                try:
                    conversation_messages = self.get_messages(conversation_id)
                except Exception:
                    conversation_messages = []

            # Step 3: Analyze and synthesize insights
            analysis_result = self.analyzer_agent.run(chunks, intent, conversation_messages, conversation_id)
            logger.info("Workflow step 3: Analysis completed")
            
            # Step 4: Generate visualizations
            visual_result = self.visual_agent.run(analysis_result["insights"], chunks)
            logger.info("Workflow step 4: Visualizations generated")
            
            # Compile final report
            visualizations_list = visual_result.get('visualizations') if isinstance(visual_result, dict) else []
            visual_tables = visual_result.get('tables') if isinstance(visual_result, dict) else []

            final_report = {
                "user_query": user_query,
                "intent": intent_result,
                "retrieved_chunks": retrieval_result,
                "analysis": analysis_result,
                "visualizations": visualizations_list,
                "visualization_tables": visual_tables,
                "conversation_id": conversation_id,
                "final_output": analysis_result["draft_report"]  # Can be enhanced
            }
            
            logger.info("Workflow completed successfully")
            return final_report
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {
                "error": str(e),
                "final_output": "Workflow failed."
            }

    def run_workflow_stream(self, user_query: str, query_vector: List[float], conversation_id: Optional[str] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Generator that yields workflow progress events as dictionaries.
        Events types: status, partial, final
        """
        try:
            # Status: classifying intent
            yield {"type": "status", "status": "intent_start"}
            intent_result = self.intent_agent.run(user_query)
            yield {"type": "status", "status": "intent_done", "intent": intent_result}

            # Status: retrieval
            yield {"type": "status", "status": "retrieval_start"}
            retrieval_result = self.retriever_agent.run(user_query, query_vector, intent=intent_result.get('intent'))
            yield {"type": "status", "status": "retrieval_done", "retrieved_chunks": retrieval_result}

            # Fetch conversation messages if present
            conversation_messages = []
            if conversation_id and self.supabase and self.supabase_ready:
                try:
                    conversation_messages = self.get_messages(conversation_id)
                except Exception:
                    conversation_messages = []

            # Analysis start
            yield {"type": "status", "status": "analysis_start"}
            analysis_result = self.analyzer_agent.run(retrieval_result['chunks'], intent_result.get('intent'), conversation_messages, conversation_id)
            # stream partial analysis by chunking the 'analysis' field
            summary = analysis_result.get('analysis', '') or analysis_result.get('draft_report', '')
            chunk_size = 200
            for i in range(0, len(summary), chunk_size):
                yield {"type": "partial", "partial": summary[i:i+chunk_size]}
            yield {"type": "status", "status": "analysis_done", "analysis_result": analysis_result}

            # Create visualizations
            yield {"type": "status", "status": "visual_start"}
            visual_result = self.visual_agent.run(analysis_result.get('insights', []), retrieval_result.get('chunks', []))
            yield {"type": "status", "status": "visual_done", "visual_result": visual_result}

            # Final report
            final_report = {
                "user_query": user_query,
                "intent": intent_result,
                "retrieved_chunks": retrieval_result,
                "analysis": analysis_result,
                "visualizations": visual_result.get('visualizations') if isinstance(visual_result, dict) else [],
                "visualization_tables": visual_result.get('tables') if isinstance(visual_result, dict) else [],
                "conversation_id": conversation_id,
                "final_output": analysis_result.get('draft_report')
            }
            yield {"type": "final", "result": final_report}
        except Exception as e:
            yield {"type": "error", "error": str(e)}

    def get_all_conversations(self):
        if not self.supabase or not self.supabase_ready:
            return list(self._conversations)
        response = self.supabase.table('conversations').select('*').order('created_at', desc=True).execute()
        return response.data

    def create_conversation(self, title=None):
        data = {'title': title or 'New Conversation', 'user_id': None}
        conv = self._safe_insert('conversations', data, single=True)
        # Ensure created_at exists in the returned conv
        if isinstance(conv, dict) and not conv.get('created_at'):
            conv['created_at'] = __import__('datetime').datetime.utcnow().isoformat()
        return conv

    def get_messages(self, conversation_id):
        if not self.supabase or not self.supabase_ready:
            return [m for m in self._messages if m.get('conversation_id') == conversation_id]
        response = self.supabase.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at').execute()
        return response.data

    def create_message(self, conversation_id, role, content):
        data = {'conversation_id': conversation_id, 'role': role, 'content': content}
        msg = self._safe_insert('messages', data, single=True)
        if isinstance(msg, dict) and not msg.get('created_at'):
            msg['created_at'] = __import__('datetime').datetime.utcnow().isoformat()
        return msg