"""
Adversarial Agent
Generates phishing email variants for training data augmentation
"""

import logging
from typing import Dict
from .base_agent import BaseAgent
from config.prompts import ADVERSARIAL_AGENT_PROMPT

logger = logging.getLogger(__name__)


class AdversarialAgent(BaseAgent):
    """Agent that generates adversarial phishing email variants"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.7, max_tokens: int = 1000):
        # Higher temperature for more creative variants
        super().__init__(provider, model, temperature, max_tokens)
    
    def generate_variant(self, original_email: str) -> str:
        """
        Generate an adversarial variant of a phishing email
        
        Args:
            original_email: Original phishing email text
        
        Returns:
            Variant email text
        """
        if not original_email or not original_email.strip():
            logger.warning("Empty email provided for variant generation")
            return original_email
        
        # Format prompt
        prompt = ADVERSARIAL_AGENT_PROMPT.format(original_email=original_email)
        
        # Call LLM (no JSON format needed, just text)
        try:
            variant = self.call_llm(prompt)
            
            # Clean up the response
            variant = variant.strip()
            # Remove any markdown formatting if present
            if variant.startswith("```"):
                lines = variant.split("\n")
                variant = "\n".join(lines[1:-1]) if len(lines) > 2 else variant
            
            return variant
            
        except Exception as e:
            logger.error(f"Error in AdversarialAgent.generate_variant: {e}")
            return original_email  # Return original if generation fails
    
    def generate_multiple_variants(self, original_email: str, n: int = 3) -> list:
        """
        Generate multiple variants
        
        Args:
            original_email: Original phishing email text
            n: Number of variants to generate
        
        Returns:
            List of variant email texts
        """
        variants = []
        for i in range(n):
            variant = self.generate_variant(original_email)
            variants.append(variant)
            logger.info(f"Generated variant {i+1}/{n}")
        return variants


