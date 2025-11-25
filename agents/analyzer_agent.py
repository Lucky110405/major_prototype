import logging
from typing import Dict, Any, List
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyzerAgent:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the Analyzer Agent with a summarization model.
        
        Args:
            model_name: HuggingFace model for summarization/analysis.
        """
        self.summarizer = pipeline("summarization", model=model_name)

    def run(self, chunks: List[Dict], intent: str) -> Dict[str, Any]:
        """
        Analyze retrieved chunks and synthesize insights.
        
        Args:
            chunks: List of retrieved chunks.
            intent: Classified intent.
        
        Returns:
            Dict with analysis, insights, and draft report.
        """
        try:
            # Combine chunk texts
            combined_text = " ".join([chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text"])
            
            if not combined_text:
                return {
                    "analysis": "No textual content to analyze.",
                    "insights": [],
                    "draft_report": "Insufficient data for report."
                }
            
            # Summarize or analyze based on intent
            if intent == "descriptive":
                summary = self.summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
                insights = ["Descriptive summary generated."]
            elif intent == "diagnostic":
                summary = self.summarizer(combined_text, max_length=200, min_length=100, do_sample=False)[0]["summary_text"]
                insights = ["Diagnostic analysis: Identified key issues and causes."]
            elif intent == "predictive":
                summary = self.summarizer(combined_text, max_length=200, min_length=100, do_sample=False)[0]["summary_text"]
                insights = ["Predictive insights: Forecasted trends based on data."]
            elif intent == "prescriptive":
                summary = self.summarizer(combined_text, max_length=200, min_length=100, do_sample=False)[0]["summary_text"]
                insights = ["Prescriptive recommendations: Suggested actions."]
            else:
                summary = "Unknown intent."
                insights = []
            
            draft_report = f"Report for {intent} query:\n\n{summary}\n\nKey Insights:\n" + "\n".join(insights)
            
            logger.info(f"Generated analysis for {intent} intent")
            
            return {
                "analysis": summary,
                "insights": insights,
                "draft_report": draft_report
            }
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return {
                "analysis": "Error in analysis.",
                "insights": [],
                "draft_report": "Analysis failed."
            }