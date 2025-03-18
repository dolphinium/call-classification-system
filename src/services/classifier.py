from typing import Dict, Optional, List
from dataclasses import dataclass
import json

from google import genai
from google.genai import types

from src.config.settings import GEMINI_API_KEYS
from src.utils.logging import logger

@dataclass
class ClassificationResult:
    classification: str
    category: str
    justification: str

class TranscriptClassifier:
    def __init__(self):
        if not GEMINI_API_KEYS:
            raise ValueError("No Gemini API keys found in configuration")
        self.api_keys = GEMINI_API_KEYS
        self.current_key_index = 0
        
        # Initialize with first key
        self._setup_gemini()
        
    def _setup_gemini(self) -> None:
        """Setup Gemini with current API key"""
        genai.configure(api_key=self.api_keys[self.current_key_index])
        
    def _rotate_api_key(self) -> None:
        """Rotate to next available API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._setup_gemini()
        
    def classify_transcript(self, transcript: str) -> ClassificationResult:
        """
        Classify a transcript using Gemini API with automatic key rotation on rate limits
        
        Args:
            transcript: The conversation transcript to classify
            
        Returns:
            ClassificationResult containing the classification details
        """
        attempts = 0
        max_attempts = len(self.api_keys)
        
        while attempts < max_attempts:
            try:
                return self._attempt_classification(transcript)
            except types.RateLimitError:
                logger.warning(f"Rate limit hit for API key {self.current_key_index}, rotating to next key")
                self._rotate_api_key()
                attempts += 1
            except Exception as e:
                logger.error(f"Unexpected error in classification: {str(e)}")
                raise
                
        raise Exception("All API keys exhausted due to rate limits")
    
    def _attempt_classification(self, transcript: str) -> ClassificationResult:
        """Attempt to classify using current API key"""
        model = genai.GenerativeModel("gemini-pro")
        
        prompt = self._build_prompt(transcript)
        response = model.generate_content(prompt)
        
        try:
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            raise
            
    def _build_prompt(self, transcript: str) -> str:
        """Build the classification prompt"""
        system_prompt = """
You are a call center expert that analyzes customer service calls transcripts in Turkish. 
In our business, users with service needs reach the most suitable service providers for them.
All phone calls between the service area and the service provider are listened to and analyzed by artificial intelligence.
Your task is to categorize and classify the call based on the conversation. 

You must output exactly one classification from this set:
    - potential_customer
    - unnecessary_call
    - empty_call
    - uncertain

If classification is "unnecessary_call", you must also choose exactly one category from:
    - guaranteed_product
    - irrelevant_sector
    - installation
    - service_fee_rejected
    - price_research
    - complaint
    - call_later
    - craftsman_didnt_come
    - basic_job
    - cancel_request
    - platform_membership

If classification is "potential_customer","empty_call" or "uncertain", your category must be "n/a".

Important details for categorizing:
1) If the product has a guarantee and it is not expired(We only deal with devices whose warranty has expired.) => classification=unnecessary_call, category=guaranteed_product
2) If the sector is irrelevant and customer service states this (i.e. "parça satışı yapmıyoruz maalesef" or "bu konuda hizmet veremiyoruz") => classification=unnecessary_call, category=irrelevant_sector
3) If the customer asks for installation => classification=unnecessary_call, category=installation
4) If the customer rejects the 500 TL service fee. Also, In cases where the customer is hesistant, other than the direct rejection of the service fee (for example, if sentences like "I understand, okay then" are used after the service fee is discussed and conversation ends afterwards) => classification=unnecessary_call, category=service_fee_rejected
5) If the customer's *primary interest* is to learn price rather than receiving service(for example, beyond the standard 500 TL service fee) => classification=unnecessary_call, category=price_research
6) If the customer complains about service => classification=unnecessary_call, category=complaint
7) If the conversation ends with the customer or technician agreeing to follow up at a later time or requesting a future call, you must classify it as "unnecessary_call" with the category "call_later."  
   • This includes the customer explicitly saying "Akşam eşimle konuşayım, döneceğim" "Daha sonra arayacağım," "Beni daha sonra arayın," "ararsın beni," "lütfen şimdi değil, sonra görüşelim," or any variation where the main outcome is not immediate service but deferring action to a future moment.  
   • Even if the customer seems interested or mentions potential repair, if the immediate result is "Call me later" rather than proceeding with the service during this call, it falls under "call_later."
8) If the customer has called before and created a service request but the craftsman or service has not arrived yet and the customer is calling for this reason => classification=unnecessary_call, category=craftsman_didnt_come
9) If the customer asks for the business location to bring something for repair => classification=uncertain, category=n/a
10) If the customer's primary intent is to resolve a minor issue by receiving guidance directly from a customer service representative—rather than requiring a technical service visit for a more complex problem => classification=unnecessary_call, category=basic_job
11) If customer wants to cancel service request => classification=unnecessary_call, category= cancel_request
12) If the customer is asking about platform membership details and how it works. => classification=unnecessary_call, category=platform_membership
13) If you are uncertain about transcription => classification=uncertain, category=n/a
14) If the transcript is too short or empty and incomprehensible to determine the purpose of the call.=> classification=empty_call, category=n/a
15) In all other cases => classification=potential_customer (with category=n/a)
   - For example, if a customer mentions or asks about price but also proceeds with scheduling or further service intentions, it is not just price_research, so it should be classified as potential_customer.

You will present your final result in the format:

<analysis>
<classification>CLASS_VALUE</classification>
<category>CATEGORY_VALUE</category>
<justification>
Short textual explanation referencing parts of the conversation.
</justification>
</analysis>
"""
        return f"{system_prompt}\n\nTranscript to analyze:\n{transcript}"
    
    def _parse_response(self, response_text: str) -> ClassificationResult:
        """Parse the LLM response into a structured format"""
        try:
            # Extract values using simple string parsing
            classification = self._extract_between(response_text, "<classification>", "</classification>")
            category = self._extract_between(response_text, "<category>", "</category>")
            justification = self._extract_between(response_text, "<justification>", "</justification>")
            
            if not all([classification, category, justification]):
                raise ValueError("Missing required fields in response")
                
            # Validate classification
            valid_classifications = {
                "potential_customer", "unnecessary_call",
                "empty_call", "uncertain"
            }
            if classification not in valid_classifications:
                raise ValueError(f"Invalid classification: {classification}")
                
            # Validate category
            if classification == "unnecessary_call":
                valid_categories = {
                    "guaranteed_product", "irrelevant_sector",
                    "installation", "service_fee_rejected",
                    "price_research", "complaint", "call_later",
                    "craftsman_didnt_come", "basic_job",
                    "cancel_request", "platform_membership"
                }
                if category not in valid_categories:
                    raise ValueError(f"Invalid category for unnecessary_call: {category}")
            elif category != "n/a":
                raise ValueError(f"Category must be 'n/a' for classification {classification}")
                
            return ClassificationResult(
                classification=classification.strip(),
                category=category.strip(),
                justification=justification.strip()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse classification response: {str(e)}")
            raise
            
    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> str:
        """Extract text between two markers"""
        try:
            start_idx = text.index(start) + len(start)
            end_idx = text.index(end, start_idx)
            return text[start_idx:end_idx].strip()
        except ValueError as e:
            raise ValueError(f"Could not find markers in text: {start}, {end}") from e 