"""
Text Agent
Analyzes email body text for phishing detection
"""

import logging
from typing import Dict
from .base_agent import BaseAgent
from config.prompts import TEXT_AGENT_PROMPT

logger = logging.getLogger(__name__)


class TextAgent(BaseAgent):
    """Agent that analyzes email text content"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.3, max_tokens: int = 500):
        super().__init__(provider, model, temperature, max_tokens)
    
    def analyze(self, email_body: str) -> Dict:
        """
        Analyze email body text
        
        Args:
            email_body: Email body text content
        
        Returns:
            Dictionary with verdict, confidence, and rationale
        """
        if not email_body or not email_body.strip():
            logger.warning("Empty email body provided")
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': 'No email body text to analyze'
            }
        
        # Format prompt
        prompt = TEXT_AGENT_PROMPT.format(email_body=email_body)
        
        # Call LLM with JSON response format
        try:
            response = self.call_llm(prompt, response_format={"type": "json_object"})
            result = self.parse_json_response(response)
            
            # Validate response
            if not self.validate_response(result):
                logger.warning("Invalid response format, using defaults")
                result = {
                    'verdict': 'legitimate',
                    'confidence': 0.5,
                    'rationale': 'Invalid response format from agent'
                }
            
            # Normalize confidence to 0-1
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
            
            # Convert verdict to probability
            # For phishing: confidence is probability of phishing
            # For legitimate: confidence is probability of legitimate, so invert
            if result['verdict'] == 'legitimate':
                result['phishing_probability'] = 1.0 - result['confidence']
            else:
                result['phishing_probability'] = result['confidence']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TextAgent.analyze: {e}")
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': f'Error during analysis: {str(e)}',
                'phishing_probability': 0.5
            }


