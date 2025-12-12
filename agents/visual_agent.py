import logging
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualAgent:
    def __init__(self):
        """
        Initialize the Visual Agent.
        """
        pass

    def run(self, insights: List[str], chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate visualizations from insights and chunks.
        
        Args:
            insights: List of insight strings.
            chunks: Retrieved chunks for data extraction.
        
        Returns:
            Dict with visualizations (base64 encoded images) and tables.
        """
        try:
            visualizations = []
            tables = []
            
            # Extract numerical data from chunks (simple example)
            data_points = []
            for chunk in chunks:
                # Prefer metadata text excerpts when present
                meta = chunk.get('metadata', {}) if isinstance(chunk, dict) else {}
                text = ''
                if isinstance(meta, dict):
                    text = meta.get('text_excerpt') or meta.get('text') or ''
                if not text:
                    text = chunk.get('text') or ''
                # Simple extraction: assume numbers in text
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    data_points.extend([int(n) for n in numbers])
            
            if data_points:
                # Create a simple bar chart
                plt.figure(figsize=(8, 6))
                plt.bar(range(len(data_points)), data_points)
                plt.title("Extracted Data Points")
                plt.xlabel("Index")
                plt.ylabel("Value")
                
                # Save to base64
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                # Add data URI prefix for convenience in frontend consumers
                data_uri = f"data:image/png;base64,{img_base64}"
                visualizations.append({"type": "chart", "data": data_uri})
                plt.close()
            
            # Create a summary table
            df = pd.DataFrame({"Insight": insights}) if insights else pd.DataFrame()
            table_html = df.to_html(index=False)
            tables.append(table_html)
            
            logger.info(f"Generated {len(visualizations)} visualizations and {len(tables)} tables")
            
            return {
                "visualizations": visualizations,
                "tables": tables
            }
        except Exception as e:
            logger.error(f"Error in visualization: {e}")
            return {
                "visualizations": [],
                "tables": []
            }