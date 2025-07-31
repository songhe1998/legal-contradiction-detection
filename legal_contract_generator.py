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
        Generate a comprehensive, professional {contract_type.replace('_', ' ')} legal contract that is internally consistent and contains NO contradictions. 
        
        Requirements:
        1. Use proper legal language and detailed structure
        2. Include ALL necessary clauses and extensive sections
        3. Ensure all terms are consistent throughout the document
        4. Make it very detailed and comprehensive (2000-3500 words)
        5. Include specific dates, amounts, parties, and detailed provisions
        6. Ensure all cross-references are accurate
        7. Make sure termination clauses align with contract duration
        8. Ensure payment terms are consistent throughout
        9. Include detailed definitions section
        10. Add comprehensive liability and indemnification clauses
        11. Include detailed dispute resolution procedures
        12. Add force majeure and other standard legal provisions
        
        Structure should include:
        - Title and parties identification
        - Comprehensive definitions section
        - Detailed scope of work/services/obligations
        - Payment terms and schedules
        - Performance standards and metrics
        - Intellectual property provisions
        - Confidentiality clauses
        - Termination and renewal provisions
        - Liability and indemnification
        - Dispute resolution procedures
        - Governing law and jurisdiction
        - Force majeure and other standard clauses
        - Signatures and execution details
        
        The contract should be professionally formatted, legally comprehensive, and internally consistent.
        Make it as detailed and thorough as a real commercial contract would be.
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
        Generate a comprehensive, professional {contract_type.replace('_', ' ')} legal contract that contains exactly 3-5 SUBTLE contradictions that could realistically occur in complex legal documents.
        
        Requirements:
        1. Use proper legal language and detailed structure
        2. Include ALL necessary clauses and extensive sections
        3. Make the contradictions subtle but clear upon careful analysis
        4. Make it very detailed and comprehensive (2000-3500 words)
        5. Include specific dates, amounts, parties, and detailed provisions
        6. Include detailed definitions section
        7. Add comprehensive liability and indemnification clauses
        8. Include detailed dispute resolution procedures
        9. Add force majeure and other standard legal provisions
        
        Types of contradictions to include (choose 3-5):
        - Conflicting dates (e.g., contract duration vs. specific deadlines vs. termination clauses)
        - Inconsistent payment terms mentioned in different sections (amounts, schedules, penalties)
        - Contradictory responsibilities or obligations between parties
        - Conflicting governing law or jurisdiction clauses in different sections
        - Mismatched definitions vs. actual usage in contract body
        - Contradictory intellectual property ownership clauses
        - Conflicting confidentiality terms or duration
        - Inconsistent liability caps or indemnification terms
        - Contradictory termination conditions or notice periods
        - Conflicting dispute resolution procedures
        
        Structure should include:
        - Title and parties identification
        - Comprehensive definitions section
        - Detailed scope of work/services/obligations
        - Payment terms and schedules
        - Performance standards and metrics
        - Intellectual property provisions
        - Confidentiality clauses
        - Termination and renewal provisions
        - Liability and indemnification
        - Dispute resolution procedures
        - Governing law and jurisdiction
        - Force majeure and other standard clauses
        - Signatures and execution details
        
        After generating the contract, provide a JSON object listing the contradictions:
        {{
            "contract": "FULL_CONTRACT_TEXT_HERE",
            "contradictions": [
                "Detailed description of first contradiction with specific section references",
                "Detailed description of second contradiction with specific section references",
                "Detailed description of third contradiction with specific section references",
                "Detailed description of fourth contradiction with specific section references (if applicable)",
                "Detailed description of fifth contradiction with specific section references (if applicable)"
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
                return content, ["Generated comprehensive contract may contain contradictions"]
        except:
            # Fallback: return the full content as contract
            return response.choices[0].message.content, ["Generated comprehensive contract may contain contradictions"]
    
    def generate_all_contracts(self) -> List[Dict]:
        """Generate all contracts for testing"""
        contracts = []
        
        print("Generating comprehensive contracts without contradictions...")
        for i in range(NUM_CONTRACTS_WITHOUT_CONTRADICTIONS):
            contract_type = random.choice(CONTRACT_TYPES)
            print(f"Generating detailed {contract_type.replace('_', ' ')} {i+1}/{NUM_CONTRACTS_WITHOUT_CONTRADICTIONS}...")
            contract_text = self.generate_contract_without_contradictions(contract_type)
            
            # Calculate word count
            word_count = len(contract_text.split())
            
            contracts.append({
                'id': f"long_clean_{i+1}",
                'type': contract_type,
                'text': contract_text,
                'word_count': word_count,
                'has_contradictions': False,
                'known_contradictions': []
            })
            print(f"Generated clean contract {i+1}/{NUM_CONTRACTS_WITHOUT_CONTRADICTIONS} ({word_count} words)")
        
        print("Generating comprehensive contracts with contradictions...")
        for i in range(NUM_CONTRACTS_WITH_CONTRADICTIONS):
            contract_type = random.choice(CONTRACT_TYPES)
            print(f"Generating detailed {contract_type.replace('_', ' ')} with contradictions {i+1}/{NUM_CONTRACTS_WITH_CONTRADICTIONS}...")
            contract_text, contradictions = self.generate_contract_with_contradictions(contract_type)
            
            # Calculate word count
            word_count = len(contract_text.split())
            
            contracts.append({
                'id': f"long_contradiction_{i+1}",
                'type': contract_type,
                'text': contract_text,
                'word_count': word_count,
                'has_contradictions': True,
                'known_contradictions': contradictions
            })
            print(f"Generated contradiction contract {i+1}/{NUM_CONTRACTS_WITH_CONTRADICTIONS} ({word_count} words)")
        
        return contracts
    
    def save_contracts(self, contracts: List[Dict], filename: str = "long_contracts.json"):
        """Save contracts to JSON file"""
        os.makedirs(CONTRACTS_DIR, exist_ok=True)
        filepath = os.path.join(CONTRACTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(contracts, f, indent=2, ensure_ascii=False)
        
        # Calculate statistics
        total_words = sum(contract['word_count'] for contract in contracts)
        avg_words = total_words / len(contracts) if contracts else 0
        
        print(f"Saved {len(contracts)} comprehensive contracts to {filepath}")
        print(f"Total words: {total_words:,}")
        print(f"Average words per contract: {avg_words:.1f}")
        print(f"Longest contract: {max(contract['word_count'] for contract in contracts)} words")
        print(f"Shortest contract: {min(contract['word_count'] for contract in contracts)} words") 