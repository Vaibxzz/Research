"""
URL Agent
Analyzes URLs from email for phishing detection
"""

import logging
from typing import Dict, List
from .base_agent import BaseAgent
from config.prompts import URL_AGENT_PROMPT

logger = logging.getLogger(__name__)


class URLAgent(BaseAgent):
    """Agent that analyzes URLs from emails"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.3, max_tokens: int = 500):
        super().__init__(provider, model, temperature, max_tokens)
    
    def analyze(self, urls: List[str]) -> Dict:
        """
        Analyze URLs from email
        
        Args:
            urls: List of URLs found in email
        
        Returns:
            Dictionary with verdict, confidence, and rationale
        """
        if not urls or len(urls) == 0:
            logger.warning("No URLs provided for analysis")
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': 'No URLs found in email',
                'phishing_probability': 0.5
            }
        
        # Format URLs as string
        urls_str = "\n".join([f"- {url}" for url in urls])
        
        # Format prompt
        prompt = URL_AGENT_PROMPT.format(urls=urls_str)
        
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
            if result['verdict'] == 'legitimate':
                result['phishing_probability'] = 1.0 - result['confidence']
            else:
                result['phishing_probability'] = result['confidence']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in URLAgent.analyze: {e}")
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': f'Error during analysis: {str(e)}',
                'phishing_probability': 0.5
            }


