import openai
import json
import os
import random
from typing import Dict, List, Tuple
from config import *

class LegalContractGenerator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def generate_contract_without_contradictions(self, contract_type: str) -> str:
        """Generate a realistic legal contract without contradictions"""
        
        prompt = f"""
        Generate a realistic, professional {contract_type.replace('_', ' ')} legal contract that is internally consistent and contains NO contradictions. 
        
        Requirements:
        1. Use proper legal language and structure
        2. Include all necessary clauses and sections
        3. Ensure all terms are consistent throughout the document
        4. Make it realistic and detailed (800-1200 words)
        5. Include specific dates, amounts, and parties
        6. Ensure all cross-references are accurate
        7. Make sure termination clauses align with contract duration
        8. Ensure payment terms are consistent throughout
        
        The contract should be professionally formatted and legally sound.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
        except Exception as primary_error:
            # Fallback to secondary model if primary fails
            print(f"Primary model {MODEL_NAME} failed, trying fallback model {FALLBACK_MODEL}")
            response = self.client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
        
        return response.choices[0].message.content
    
    def generate_contract_with_contradictions(self, contract_type: str) -> Tuple[str, List[str]]:
        """Generate a realistic legal contract with embedded contradictions"""
        
        prompt = f"""
        Generate a realistic, professional {contract_type.replace('_', ' ')} legal contract that contains exactly 2-3 SUBTLE contradictions that could realistically occur in legal documents.
        
        Requirements:
        1. Use proper legal language and structure
        2. Include all necessary clauses and sections
        3. Make the contradictions subtle but clear upon careful analysis
        4. Make it realistic and detailed (800-1200 words)
        5. Include specific dates, amounts, and parties
        
        Types of contradictions to include:
        - Conflicting dates (e.g., contract duration vs. termination clauses)
        - Inconsistent payment terms mentioned in different sections
        - Contradictory responsibilities or obligations
        - Conflicting governing law or jurisdiction clauses
        - Mismatched definitions vs. usage in contract body
        
        After generating the contract, provide a JSON object listing the contradictions:
        {{
            "contract": "FULL_CONTRACT_TEXT_HERE",
            "contradictions": [
                "Description of first contradiction",
                "Description of second contradiction",
                "Description of third contradiction (if applicable)"
            ]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE + 0.1  # Slightly higher temperature for variation
            )
        except Exception as primary_error:
            # Fallback to secondary model if primary fails
            print(f"Primary model {MODEL_NAME} failed, trying fallback model {FALLBACK_MODEL}")
            response = self.client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE + 0.1
            )
        
        try:
            # Parse the JSON response
            content = response.choices[0].message.content
            # Find JSON part (assumes it's at the end)
            json_start = content.find('{')
            if json_start != -1:
                json_content = content[json_start:]
                parsed = json.loads(json_content)
                return parsed['contract'], parsed['contradictions']
            else:
                # Fallback: return the full content as contract
                return content, ["Generated contract may contain contradictions"]
        except:
            # Fallback: return the full content as contract
            return response.choices[0].message.content, ["Generated contract may contain contradictions"]
    
    def generate_all_contracts(self) -> List[Dict]:
        """Generate all contracts for testing"""
        contracts = []
        
        print("Generating contracts without contradictions...")
        for i in range(NUM_CONTRACTS_WITHOUT_CONTRADICTIONS):
            contract_type = random.choice(CONTRACT_TYPES)
            contract_text = self.generate_contract_without_contradictions(contract_type)
            
            contracts.append({
                'id': f"clean_{i+1}",
                'type': contract_type,
                'text': contract_text,
                'has_contradictions': False,
                'known_contradictions': []
            })
            print(f"Generated clean contract {i+1}/{NUM_CONTRACTS_WITHOUT_CONTRADICTIONS}")
        
        print("Generating contracts with contradictions...")
        for i in range(NUM_CONTRACTS_WITH_CONTRADICTIONS):
            contract_type = random.choice(CONTRACT_TYPES)
            contract_text, contradictions = self.generate_contract_with_contradictions(contract_type)
            
            contracts.append({
                'id': f"contradiction_{i+1}",
                'type': contract_type,
                'text': contract_text,
                'has_contradictions': True,
                'known_contradictions': contradictions
            })
            print(f"Generated contradiction contract {i+1}/{NUM_CONTRACTS_WITH_CONTRADICTIONS}")
        
        return contracts
    
    def save_contracts(self, contracts: List[Dict], filename: str = "generated_contracts.json"):
        """Save contracts to JSON file"""
        os.makedirs(CONTRACTS_DIR, exist_ok=True)
        filepath = os.path.join(CONTRACTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(contracts, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(contracts)} contracts to {filepath}") 