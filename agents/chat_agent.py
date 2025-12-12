import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(self, orchestrator, text_embedder=None):
        self.orchestrator = orchestrator
        self.text_embedder = text_embedder

    def run(self, conversation_id: Optional[str], user_content: str) -> Dict[str, Any]:
        """
        Accepts a user message, saves it to the conversation, runs the workflow to generate an
        assistant reply (using orchestrator.run_workflow), and stores the assistant message.
        Returns the assistant message object and the full workflow result.
        """
        if not conversation_id:
            # Create a conversation if none provided
            try:
                conv = self.orchestrator.create_conversation(None)
                conversation_id = conv.get('id') if isinstance(conv, dict) else conv
            except Exception:
                # Could not create conversation, continue with None and allow orchestrator to handle
                conversation_id = None
        # Save user message
        user_msg = None
        if conversation_id:
            try:
                user_msg = self.orchestrator.create_message(conversation_id, 'user', user_content)
            except Exception:
                user_msg = None

        # Optionally compute embedding
        query_vector = None
        if self.text_embedder:
            try:
                query_vector = self.text_embedder.embed([user_content])[0]
            except Exception:
                query_vector = None

        # Run the orchestrator workflow to generate analysis and final_output
        # Pass conversation_id to orchestrator for context-aware workflows
        result = self.orchestrator.run_workflow(user_content, query_vector or [], conversation_id=conversation_id)

        assistant_content = result.get('final_output', 'No response generated')

        assistant_msg = None
        if conversation_id:
            try:
                assistant_msg = self.orchestrator.create_message(conversation_id, 'assistant', assistant_content)
            except Exception:
                assistant_msg = {'role': 'assistant', 'content': assistant_content}

        return {
            'conversation_id': conversation_id,
            'assistant_message': assistant_msg,
            'workflow_result': result,
        }
