"""
Email Parser Utilities
Extracts body text, URLs, and metadata from email files
"""

import re
import email
from email import message
from email.header import decode_header
from email.utils import parseaddr
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


class EmailParser:
    """Parse emails and extract components for analysis"""
    
    # Common phishing keywords
    PHISHING_KEYWORDS = [
        'urgent', 'verify', 'suspend', 'account', 'security', 'update',
        'immediately', 'click here', 'limited time', 'congratulations',
        'winner', 'prize', 'activate', 'confirm', 'validate', 'expire',
        'expired', 'action required', 'verify your account', 'unauthorized'
    ]
    
    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def parse_email_file(self, email_path: str) -> Dict:
        """Parse an email file and return structured data"""
        with open(email_path, 'rb') as f:
            msg = email.message_from_bytes(f.read())
        return self.parse_email_message(msg)
    
    def parse_email_string(self, email_string: str) -> Dict:
        """Parse an email string and return structured data"""
        msg = email.message_from_string(email_string)
        return self.parse_email_message(msg)
    
    def parse_email_message(self, msg: message.Message) -> Dict:
        """Parse an email.message.Message object"""
        # Extract body text
        body_text = self._extract_body_text(msg)
        
        # Extract URLs
        urls = self._extract_urls(body_text)
        
        # Extract metadata
        metadata = self._extract_metadata(msg)
        
        return {
            'body_text': body_text,
            'urls': urls,
            'metadata': metadata,
            'phishing_keyword_count': self._count_phishing_keywords(body_text)
        }
    
    def _extract_body_text(self, msg: message.Message) -> str:
        """Extract plain text body from email"""
        body_text = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            body_text += payload.decode('utf-8', errors='ignore')
                        except:
                            body_text += str(payload)
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    body_text = payload.decode('utf-8', errors='ignore')
                except:
                    body_text = str(payload)
        
        # Clean up the text
        body_text = body_text.strip()
        # Remove excessive whitespace
        body_text = re.sub(r'\s+', ' ', body_text)
        
        return body_text
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = self.url_pattern.findall(text)
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        return unique_urls
    
    def _extract_metadata(self, msg: message.Message) -> Dict:
        """Extract email headers and metadata"""
        metadata = {}
        
        # Basic headers
        metadata['sender'] = self._decode_header(msg.get('From', ''))
        metadata['reply_to'] = self._decode_header(msg.get('Reply-To', ''))
        metadata['subject'] = self._decode_header(msg.get('Subject', ''))
        metadata['date'] = msg.get('Date', '')
        metadata['to'] = self._decode_header(msg.get('To', ''))
        metadata['cc'] = self._decode_header(msg.get('CC', ''))
        
        # Extract domain from sender
        sender_email = parseaddr(metadata['sender'])[1]
        if '@' in sender_email:
            metadata['sender_domain'] = sender_email.split('@')[1].lower()
        else:
            metadata['sender_domain'] = ''
        
        # Authentication headers (SPF, DKIM, DMARC)
        # These are typically added by receiving mail servers
        metadata['spf_result'] = msg.get('Received-SPF', 'none')
        metadata['dkim_result'] = msg.get('DKIM-Signature', 'none')
        metadata['dmarc_result'] = msg.get('Authentication-Results', 'none')
        
        # Check if SPF/DKIM/DMARC passed
        metadata['spf_pass'] = 'pass' in metadata['spf_result'].lower()
        metadata['dkim_pass'] = metadata['dkim_result'] != 'none'
        
        # Parse DMARC from Authentication-Results
        auth_results = metadata['dmarc_result'].lower()
        metadata['dmarc_pass'] = 'dmarc=pass' in auth_results
        
        # All headers for advanced analysis
        metadata['all_headers'] = dict(msg.items())
        
        return metadata
    
    def _decode_header(self, header_value: str) -> str:
        """Decode email header value"""
        if not header_value:
            return ""
        
        decoded_parts = decode_header(header_value)
        decoded_str = ""
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                try:
                    decoded_str += part.decode(encoding or 'utf-8', errors='ignore')
                except:
                    decoded_str += part.decode('utf-8', errors='ignore')
            else:
                decoded_str += str(part)
        
        return decoded_str
    
    def _count_phishing_keywords(self, text: str) -> int:
        """Count occurrences of phishing keywords in text"""
        text_lower = text.lower()
        count = 0
        for keyword in self.PHISHING_KEYWORDS:
            count += text_lower.count(keyword.lower())
        return count
    
    def get_url_count(self, urls: List[str]) -> int:
        """Get count of URLs"""
        return len(urls)





