import logging
from typing import Dict, Any, cast
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentAgent:
    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli"):
        """
        Initialize the Intent Agent with a zero-shot classification model.
        
        Args:
            model_name: HuggingFace model for classification.
        """
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.labels = ["descriptive", "diagnostic", "predictive", "prescriptive"]

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Classify the intent of the user query.
        
        Args:
            user_query: The user's query string.
        
        Returns:
            Dict with intent, confidence, and reasoning.
        """
        try:
            result = cast(Dict[str, Any], self.classifier(user_query, self.labels))
            intent = result["labels"][0]
            confidence = result["scores"][0]
            
            logger.info(f"Classified query intent: {intent} with confidence {confidence}")
            
            return {
                "intent": intent,
                "confidence": confidence,
                "reasoning": f"Query classified as {intent} based on semantic analysis."
            }
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {
                "intent": "descriptive",  # Default fallback
                "confidence": 0.0,
                "reasoning": "Fallback due to error in classification."
            }