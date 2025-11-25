import logging
from typing import Dict, Any, List
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

    def run_workflow(self, user_query: str, query_vector: List[float]) -> Dict[str, Any]:
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
            
            # Step 3: Analyze and synthesize insights
            analysis_result = self.analyzer_agent.run(chunks, intent)
            logger.info("Workflow step 3: Analysis completed")
            
            # Step 4: Generate visualizations
            visual_result = self.visual_agent.run(analysis_result["insights"], chunks)
            logger.info("Workflow step 4: Visualizations generated")
            
            # Compile final report
            final_report = {
                "user_query": user_query,
                "intent": intent_result,
                "retrieved_chunks": retrieval_result,
                "analysis": analysis_result,
                "visualizations": visual_result,
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