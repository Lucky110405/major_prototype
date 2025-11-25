import logging
from typing import Dict, Any, Optional
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityAgent:
    def __init__(self):
        """
        Initialize the Modality Agent for model selection.
        """
        self.available_models = {
            "text_embedding": {
                "tiny": "prajjwal1/bert-tiny",
                "small": "prajjwal1/bert-small",
                "base": "bert-base-uncased"
            },
            "text_classification": {
                "distilbert": "typeform/distilbert-base-uncased-mnli"
            },
            "text_summarization": {
                "flan_t5_small": "google/flan-t5-small",
                "flan_t5_base": "google/flan-t5-base"
            },
            "image_embedding": {
                "clip_vit_base": "openai/clip-vit-base-patch32",
                "clip_vit_large": "openai/clip-vit-large-patch14"
            },
            "audio_transcription": {
                "whisper_small": "openai/whisper-small",
                "whisper_base": "openai/whisper-base"
            },
            "multimodal_analysis": {
                "gemini_1_5_pro": "gemini-1.5-pro"
            }
        }

    def select_model(self, task: str, modality: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """
        Select the best model for a given task and modality.
        
        Args:
            task: The task (e.g., 'embedding', 'classification', 'summarization', 'transcription', 'analysis').
            modality: The modality (e.g., 'text', 'image', 'audio', 'multimodal').
            constraints: Optional constraints like 'speed', 'accuracy', 'size'.
        
        Returns:
            Model name string.
        """
        key = f"{modality}_{task}" if modality != "multimodal" else f"{modality}_analysis"
        
        if key not in self.available_models:
            logger.warning(f"No models available for {key}, using default.")
            return self.get_default_model(task, modality)
        
        models = self.available_models[key]
        
        if constraints:
            if "speed" in constraints and constraints["speed"] == "high":
                # Prefer smaller models
                for size in ["tiny", "small", "base"]:
                    if size in models:
                        return models[size]
            elif "accuracy" in constraints and constraints["accuracy"] == "high":
                # Prefer larger models
                for size in reversed(["tiny", "small", "base", "large"]):
                    if size in models:
                        return models[size]
        
        # Default: pick the first or a balanced one
        return list(models.values())[0]

    def get_default_model(self, task: str, modality: str) -> str:
        """
        Get a default model if specific one not found.
        """
        defaults = {
            "text_embedding": "prajjwal1/bert-tiny",
            "text_classification": "typeform/distilbert-base-uncased-mnli",
            "text_summarization": "google/flan-t5-small",
            "image_embedding": "openai/clip-vit-base-patch32",
            "audio_transcription": "openai/whisper-small",
            "multimodal_analysis": "gemini-1.5-pro"
        }
        key = f"{modality}_{task}" if modality != "multimodal" else f"{modality}_analysis"
        return defaults.get(key, "prajjwal1/bert-tiny")

    def recommend_pipeline(self, modality: str, tasks: list) -> Dict[str, str]:
        """
        Recommend a pipeline of models for a sequence of tasks on a modality.
        
        Args:
            modality: The primary modality.
            tasks: List of tasks (e.g., ['embedding', 'classification']).
        
        Returns:
            Dict of task to model.
        """
        pipeline = {}
        for task in tasks:
            pipeline[task] = self.select_model(task, modality)
        return pipeline