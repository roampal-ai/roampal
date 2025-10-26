"""
Disclaimer Manager for Legal Compliance
Manages disclaimers across the application
"""

from typing import Dict, Optional
from datetime import datetime

class DisclaimerManager:
    """Centralized disclaimer management for legal compliance"""
    
    # Core disclaimers for different contexts
    DISCLAIMERS = {
        "chat_footer": "AI-generated response. Verify important information independently.",
        
        "chat_medical": "⚠️ Not medical advice. Consult healthcare professionals for medical concerns.",
        
        "chat_legal": "⚠️ Not legal advice. Consult qualified attorneys for legal matters.",
        
        "chat_financial": "⚠️ Not financial advice. Consult licensed advisors for financial decisions.",
        
        "code_generation": "⚠️ Generated code should be reviewed and tested before production use.",
        
        "first_message": (
            "Welcome! I'm an AI assistant. My responses are AI-generated and should be "
            "verified for accuracy. I'm not a substitute for professional advice."
        ),
        
        "error_fallback": "AI assistance temporarily limited. Response may be incomplete.",
        
        "image_analysis": "Image interpretation is AI-generated and may not be fully accurate.",
        
        "web_search": "Web results summarized by AI. Visit sources directly for complete information.",
        
        "book_knowledge": "Information synthesized from uploaded content. May reflect source biases.",
        
        "shard_specific": "This AI agent has specialized training. Responses reflect its configured personality."
    }
    
    # Keywords that trigger specific disclaimers
    TRIGGER_KEYWORDS = {
        "medical": ["health", "medical", "doctor", "symptom", "disease", "treatment", "medication", "diagnosis"],
        "legal": ["legal", "lawyer", "law", "court", "contract", "sue", "rights", "attorney"],
        "financial": ["invest", "stock", "trading", "financial", "money", "bitcoin", "crypto", "forex"],
        "code": ["code", "function", "class", "implement", "debug", "python", "javascript", "api"]
    }
    
    @staticmethod
    def get_disclaimer(context: str = "chat_footer") -> str:
        """Get appropriate disclaimer for context"""
        return DisclaimerManager.DISCLAIMERS.get(context, DisclaimerManager.DISCLAIMERS["chat_footer"])
    
    @staticmethod
    def check_triggered_disclaimers(text: str) -> Optional[str]:
        """Check if text triggers any special disclaimers"""
        text_lower = text.lower()
        
        for category, keywords in DisclaimerManager.TRIGGER_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                if category == "medical":
                    return DisclaimerManager.DISCLAIMERS["chat_medical"]
                elif category == "legal":
                    return DisclaimerManager.DISCLAIMERS["chat_legal"]
                elif category == "financial":
                    return DisclaimerManager.DISCLAIMERS["chat_financial"]
                elif category == "code":
                    return DisclaimerManager.DISCLAIMERS["code_generation"]
        
        return None
    
    @staticmethod
    def add_disclaimer_to_response(response: Dict, user_input: str = "", 
                                  is_first_message: bool = False) -> Dict:
        """Add appropriate disclaimer to chat response"""
        
        # First message gets welcome disclaimer
        if is_first_message:
            response["disclaimer"] = DisclaimerManager.DISCLAIMERS["first_message"]
            return response
        
        # Check for triggered disclaimers based on content
        triggered = DisclaimerManager.check_triggered_disclaimers(user_input)
        if triggered:
            response["disclaimer"] = triggered
        else:
            # Default footer disclaimer
            response["disclaimer"] = DisclaimerManager.DISCLAIMERS["chat_footer"]
        
        # Add disclaimer type for UI styling
        response["disclaimer_type"] = "warning" if triggered else "info"
        
        return response
    
    @staticmethod
    def get_terms_of_use() -> str:
        """Return full terms of use for display"""
        return """
TERMS OF USE - AI ASSISTANT

1. NO PROFESSIONAL ADVICE
This AI assistant does not provide medical, legal, financial, or other professional advice.
Always consult qualified professionals for important decisions.

2. ACCURACY DISCLAIMER  
AI-generated responses may contain errors or inaccuracies. 
Users are responsible for verifying information before relying on it.

3. PROHIBITED USES
Do not use for:
- Illegal activities
- Generating harmful content
- Making critical decisions without human review
- Activities that could cause harm to yourself or others

4. DATA & PRIVACY
- Your data is processed locally on your device
- Conversations may be stored for functionality
- You can delete your data at any time

5. LIABILITY LIMITATION
This software is provided "as is" without warranties.
We are not liable for any damages arising from use of this AI assistant.

6. ACCEPTANCE
By using this software, you accept these terms and agree to use the AI responsibly.

Last Updated: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d"))
    
    @staticmethod
    def get_startup_disclaimer() -> str:
        """Return disclaimer shown on application startup"""
        return """
╔════════════════════════════════════════════════════════════════╗
║                    IMPORTANT NOTICE                            ║
║                                                                ║
║  This is an AI Assistant. All responses are AI-generated      ║
║  and should be verified for accuracy.                         ║
║                                                                ║
║  NOT FOR: Medical, legal, or financial advice                 ║
║  REMEMBER: Critical decisions need human review               ║
║                                                                ║
║  By continuing, you accept our Terms of Use                   ║
╚════════════════════════════════════════════════════════════════╝
"""