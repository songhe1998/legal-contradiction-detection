import openai
import json
import time
from typing import Dict, List, Tuple
from config import *

class ContradictionDetector:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def detect_contradictions(self, contract_text: str) -> Tuple[bool, List[str], float]:
        """
        Detect contradictions in a legal contract
        Returns: (has_contradictions, contradictions_found, confidence_score)
        """
        
        # Smart prompt designed for comprehensive contradiction detection
        prompt = f"""
        You are an expert legal analyst specializing in contract review and contradiction detection. 
        
        Your task is to carefully analyze the following legal contract for any internal contradictions, inconsistencies, or conflicting terms.
        
        CONTRACT TO ANALYZE:
        {contract_text}
        
        ANALYSIS INSTRUCTIONS:
        1. Read through the entire contract carefully
        2. Look for contradictions in these areas:
           - Dates and time periods (start/end dates, duration, deadlines)
           - Payment terms and amounts (conflicting payment schedules, amounts)
           - Responsibilities and obligations (contradictory duties)
           - Termination clauses vs. contract duration
           - Governing law and jurisdiction
           - Definitions vs. actual usage in the contract
           - Performance standards and requirements
           - Liability and indemnification clauses
           - Intellectual property rights
           - Confidentiality terms
        
        3. Consider cross-references between different sections
        4. Look for subtle inconsistencies that might not be immediately obvious
        5. Distinguish between contradictions and mere ambiguities
        
        RESPONSE FORMAT:
        Provide your analysis in the following JSON format:
        {{
            "has_contradictions": true/false,
            "contradictions_found": [
                "Detailed description of contradiction 1",
                "Detailed description of contradiction 2",
                ...
            ],
            "confidence_score": 0.0-1.0,
            "reasoning": "Brief explanation of your analysis approach and confidence level"
        }}
        
        IMPORTANT: 
        - Be thorough but precise
        - Only flag actual contradictions, not ambiguities
        - If you find no contradictions, set has_contradictions to false
        - Confidence score should reflect how certain you are of your analysis
        - Include specific references to contract sections when describing contradictions
        """
        
        try:
            # Try primary model first
            # Use different parameters for o3 models vs others
            if MODEL_NAME.startswith('o3'):
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=MAX_TOKENS
                )
            else:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
        except Exception as primary_error:
            # Fallback to secondary model if primary fails
            print(f"Primary model {MODEL_NAME} failed, trying fallback model {FALLBACK_MODEL}")
            try:
                # Use appropriate parameter for fallback model
                if FALLBACK_MODEL.startswith('o3'):
                    response = self.client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=MAX_TOKENS
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE
                    )
            except Exception as fallback_error:
                print(f"Both models failed. Primary: {primary_error}, Fallback: {fallback_error}")
                return False, [], 0.0
        
        # Parse response content
        try:
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Find and extract JSON
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_content = content[json_start:json_end]
                
                result = json.loads(json_content)
                
                return (
                    result.get('has_contradictions', False),
                    result.get('contradictions_found', []),
                    result.get('confidence_score', 0.0)
                )
            except:
                # Fallback parsing
                has_contradictions = 'true' in content.lower() and 'contradiction' in content.lower()
                return has_contradictions, [], 0.5
                
        except Exception as e:
            print(f"Error in contradiction detection: {e}")
            return False, [], 0.0
    
    def analyze_contracts(self, contracts: List[Dict]) -> List[Dict]:
        """Analyze all contracts for contradictions"""
        results = []
        
        print(f"Analyzing {len(contracts)} contracts for contradictions...")
        
        for i, contract in enumerate(contracts):
            print(f"Analyzing contract {i+1}/{len(contracts)}: {contract['id']}")
            
            # Add delay to respect rate limits
            time.sleep(1)
            
            detection_result = self.detect_contradictions(contract['text'])
            if detection_result is None:
                print(f"Warning: Detection failed for contract {contract['id']}, using default values")
                has_contradictions, contradictions_found, confidence = False, [], 0.0
            else:
                has_contradictions, contradictions_found, confidence = detection_result
            
            result = {
                'contract_id': contract['id'],
                'contract_type': contract['type'],
                'ground_truth': contract['has_contradictions'],
                'known_contradictions': contract['known_contradictions'],
                'predicted_contradictions': has_contradictions,
                'detected_contradictions': contradictions_found,
                'confidence_score': confidence,
                'contract_text': contract['text']  # Keep for reference
            }
            
            results.append(result)
            
            # Print progress
            status = "✓" if has_contradictions == contract['has_contradictions'] else "✗"
            print(f"  {status} Predicted: {has_contradictions}, Actual: {contract['has_contradictions']}")
        
        return results
    
    def save_analysis_results(self, results: List[Dict], filename: str = "analysis_results.json"):
        """Save analysis results to JSON file"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved analysis results to {filepath}") 