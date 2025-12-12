import logging
import os
import subprocess
from typing import Dict, Any, List, Optional
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
        # Optionally use a local Ollama model if requested via env var USE_OLLAMA
        self.use_ollama = os.getenv("USE_OLLAMA", "0") == "1"
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama2")
        if self.use_ollama:
            logging.info(f"AnalyzerAgent: configured to use Ollama model {self.ollama_model}")
            self.summarizer = None
        else:
            self.summarizer = pipeline("summarization", model=model_name)

    def _ollama_generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Generate text using local Ollama CLI. Falls back cleanly if CLI not present.

        This implementation calls the `ollama` command-line tool: `ollama run <model> <prompt>`
        and returns the stdout. It assumes `ollama` is installed and the model is available locally.
        """
        try:
            # Use subprocess to call ollama CLI. Provide prompt via stdin to avoid shell quoting issues.
            # Force UTF-8 decoding and replace undecodable bytes to avoid UnicodeDecodeError on Windows.
            proc = subprocess.run(
                ["ollama", "run", self.ollama_model],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=True,
            )
            return proc.stdout.strip()
        except FileNotFoundError:
            logging.error("Ollama CLI not found on PATH. Install Ollama or set USE_OLLAMA=0 to use HF pipeline.")
            return ""
        except subprocess.CalledProcessError as e:
            # Avoid failing on decode issues; log what we can.
            out = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else str(e.stdout)
            err = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else str(e.stderr)
            logging.error(f"Ollama generation failed: {e}; stdout: {out}; stderr: {err}")
            return ""

    def run(self, chunks: List[Dict], intent: str, conversation_messages: List[Dict] | None = None, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze retrieved chunks and synthesize insights.
        
        Args:
            chunks: List of retrieved chunks.
            intent: Classified intent.
        
        Returns:
            Dict with analysis, insights, and draft report.
        """
        try:
            # Build a prompt that includes cleaned chunk excerpts (prefer metadata['text_excerpt'] then metadata['text'] then chunk['text']).
            # Limit number of chunks and excerpt length to avoid exceeding model context.
            max_chunks = 6
            max_excerpt_chars = 800
            used_chunks = []
            prompt_parts = []
            for i, chunk in enumerate(chunks[:max_chunks], start=1):
                meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                # Select only a few metadata fields to include (avoid dumping full dict)
                asset = meta.get("asset") or meta.get("symbol") or ""
                filename = meta.get("filename") or ""
                row_range = meta.get("row_range") or meta.get("page") or ""
                excerpt = ""
                if isinstance(meta, dict):
                    excerpt = meta.get("text_excerpt") or meta.get("text") or ""
                if not excerpt:
                    excerpt = chunk.get("text") or ""
                excerpt = excerpt[:max_excerpt_chars]
                used_chunks.append({"id": chunk.get("id"), "metadata": {"asset": asset, "filename": filename, "row_range": row_range}, "text_excerpt": excerpt})
                meta_lines = []
                if asset:
                    meta_lines.append(f"asset: {asset}")
                if filename:
                    meta_lines.append(f"filename: {filename}")
                if row_range:
                    meta_lines.append(f"range: {row_range}")
                meta_str = ", ".join(meta_lines) if meta_lines else ""
                prompt_parts.append(f"=== CHUNK {i} ===\n{meta_str}\nEXCERPT:\n{excerpt}\n--- END CHUNK ---\n")

            if not prompt_parts:
                return {
                    "analysis": "No textual content to analyze.",
                    "insights": [],
                    "draft_report": "Insufficient data for report.",
                    "used_chunks": []
                }

            combined_text = "\n\n".join(prompt_parts)

            # Compose instruction for the model using the extracted chunks
            instruction = (
                f"You are an analyst. Using ONLY the following chunk excerpts and their metadata, perform a {intent} analysis.\n"
                "Do NOT introduce information that is not present in the excerpts.\n"
                "Produce a structured response with three clearly labeled sections: Summary (3-6 sentences), Key Insights (bullet list), and Recommended Next Steps (bullet list).\n\n"
            )
            # Add optional conversation history before chunk excerpts for context
            conv_history_str = ""
            if conversation_messages:
                # Use last N messages
                last_msgs = conversation_messages[-8:]
                parts = []
                for m in last_msgs:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    parts.append(f"{role.upper()}: {content}")
                conv_history_str = "\nCONVERSATION HISTORY:\n" + "\n".join(parts) + "\n\n"

            model_input = instruction + conv_history_str + combined_text
            
            # Summarize/analyze using the composed instruction and chunk excerpts. Adjust lengths by intent.
            if intent == "descriptive":
                max_len, min_len = 150, 50
            elif intent in ("diagnostic", "predictive", "prescriptive"):
                max_len, min_len = 250, 100
            else:
                max_len, min_len = 150, 50

            # Use the summarization pipeline or Ollama on the assembled prompt.
            # The HF pipeline expects shorter inputs; if the prompt is long the tokenizer will truncate.
            if self.use_ollama:
                # Use Ollama CLI to generate text; pass the instruction+excerpts as prompt.
                summary = self._ollama_generate(model_input, max_tokens=max_len)
                if not summary:
                    # Fallback message if Ollama failed
                    summary = "(Ollama generation failed or returned empty output.)"
            else:
                if self.summarizer is not None:
                    summary = self.summarizer(model_input, max_length=max_len, min_length=min_len, do_sample=False)[0].get("summary_text", "")
                else:
                    summary = "(No summarizer available.)"
            # Simple heuristic insights: extract sentences from summary or return a placeholder
            insights = [s.strip() for s in summary.split('\n') if s.strip()][:5]
            
            draft_report = f"Report for {intent} query:\n\n{summary}\n\nKey Insights:\n" + "\n".join(insights)

            logger.info(f"Generated analysis for {intent} intent")

            result = {
                "analysis": summary,
                "insights": insights,
                "draft_report": draft_report,
                "used_chunks": used_chunks
            }
            if conversation_id:
                result['conversation_id'] = conversation_id
            return result
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            result = {
                "analysis": "Error in analysis.",
                "insights": [],
                "draft_report": "Analysis failed."
            }
            if conversation_id:
                result['conversation_id'] = conversation_id
            return result