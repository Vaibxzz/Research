"""
Metadata Agent
Analyzes email headers and authentication records for phishing detection
"""

import logging
from typing import Dict
from .base_agent import BaseAgent
from config.prompts import METADATA_AGENT_PROMPT

logger = logging.getLogger(__name__)


class MetadataAgent(BaseAgent):
    """Agent that analyzes email metadata (headers, SPF, DKIM, DMARC)"""
    
    def __init__(self, provider: str = "google", model: str = None, temperature: float = 0.3, max_tokens: int = 500):
        super().__init__(provider, model, temperature, max_tokens)
    
    def analyze(self, metadata: Dict) -> Dict:
        """
        Analyze email metadata
        
        Args:
            metadata: Dictionary with email metadata (sender, reply_to, spf_result, etc.)
        
        Returns:
            Dictionary with verdict, confidence, and rationale
        """
        # Extract relevant metadata fields
        sender = metadata.get('sender', 'unknown')
        reply_to = metadata.get('reply_to', 'none')
        spf_result = metadata.get('spf_result', 'none')
        dkim_result = metadata.get('dkim_result', 'none')
        dmarc_result = metadata.get('dmarc_result', 'none')
        
        # Format other headers (limit to most important ones)
        other_headers = {
            'subject': metadata.get('subject', ''),
            'sender_domain': metadata.get('sender_domain', ''),
            'spf_pass': metadata.get('spf_pass', False),
            'dkim_pass': metadata.get('dkim_pass', False),
            'dmarc_pass': metadata.get('dmarc_pass', False)
        }
        other_headers_str = "\n".join([f"{k}: {v}" for k, v in other_headers.items()])
        
        # Format prompt
        prompt = METADATA_AGENT_PROMPT.format(
            sender=sender,
            reply_to=reply_to,
            spf_result=spf_result,
            dkim_result=dkim_result,
            dmarc_result=dmarc_result,
            other_headers=other_headers_str
        )
        
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
            logger.error(f"Error in MetadataAgent.analyze: {e}")
            return {
                'verdict': 'legitimate',
                'confidence': 0.5,
                'rationale': f'Error during analysis: {str(e)}',
                'phishing_probability': 0.5
            }


