"""
Explanation Simplifier Agent
Combines technical rationales into simple, user-friendly explanations
"""

import logging
from typing import Dict
from .base_agent import BaseAgent
from config.prompts import EXPLANATION_SIMPLIFIER_PROMPT

logger = logging.getLogger(__name__)


class ExplanationSimplifier(BaseAgent):
    """Agent that simplifies technical explanations"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.5, max_tokens: int = 300):
        super().__init__(provider, model, temperature, max_tokens)
    
    def simplify(self, text_rationale: str, url_rationale: str, 
                 metadata_rationale: str) -> str:
        """
        Synthesize three agent rationales into one simple explanation
        
        Args:
            text_rationale: Rationale from Text Agent
            url_rationale: Rationale from URL Agent
            metadata_rationale: Rationale from Metadata Agent
        
        Returns:
            Simplified, user-friendly explanation
        """
        # Format prompt
        prompt = EXPLANATION_SIMPLIFIER_PROMPT.format(
            text_rationale=text_rationale or "No text analysis available",
            url_rationale=url_rationale or "No URL analysis available",
            metadata_rationale=metadata_rationale or "No metadata analysis available"
        )
        
        # Call LLM (no JSON format, just text)
        try:
            explanation = self.call_llm(prompt)
            
            # Clean up the response
            explanation = explanation.strip()
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in ExplanationSimplifier.simplify: {e}")
            # Fallback: combine rationales with simple formatting
            return (
                f"Based on our analysis:\n"
                f"Text Analysis: {text_rationale or 'Not available'}\n"
                f"URL Analysis: {url_rationale or 'Not available'}\n"
                f"Metadata Analysis: {metadata_rationale or 'Not available'}"
            )


