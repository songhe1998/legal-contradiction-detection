#!/usr/bin/env python3
"""
Simple Contract Contradiction Experiment
Uses o3 to generate contracts and gpt-4o to detect contradictions
"""

import openai
import json
import time
from typing import List, Dict, Any
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleContractExperiment:
    def __init__(self):
        self.client = openai.OpenAI()
        self.results = []
        
    def generate_contract_with_contradiction(self) -> str:
        """Generate a 2-3 page realistic contract WITH contradictions using o3"""
        prompt = """Generate a very realistic contract content that has contradictions in it, around 2-3 pages. 
        Make it look like a real business contract with proper legal language, but include some contradictory clauses."""
        
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating contract with contradiction: {e}")
            return None
    
    def generate_contract_without_contradiction(self) -> str:
        """Generate a 2-3 page realistic contract WITHOUT contradictions using o3"""
        prompt = """Generate a very realistic contract content that does NOT have any contradictions in it, around 2-3 pages. 
        Make it look like a real business contract with proper legal language and ensure all clauses are consistent."""
        
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=8000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating contract without contradiction: {e}")
            return None
    
    def detect_contradiction(self, contract_text: str) -> bool:
        """Use gpt-4o to detect if there's a contradiction, returns True if contradiction detected"""
        prompt = f"""Tell me whether there is contradiction in this document, only tell me yes or no.

Document:
{contract_text}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            answer = response.choices[0].message.content.strip().lower()
            return "yes" in answer
        except Exception as e:
            print(f"Error detecting contradiction: {e}")
            return None
    
    def run_experiment(self, num_contracts_per_type: int = 5):
        """Run the complete experiment"""
        print(f"Starting Simple Contract Contradiction Experiment")
        print(f"Generating {num_contracts_per_type} contracts with contradictions and {num_contracts_per_type} without")
        print("-" * 60)
        
        total_contracts = 0
        correct_predictions = 0
        
        # Generate contracts WITH contradictions
        print("Generating contracts WITH contradictions...")
        for i in range(num_contracts_per_type):
            print(f"  Contract {i+1}/{num_contracts_per_type}...")
            
            contract = self.generate_contract_with_contradiction()
            if contract is None:
                continue
                
            time.sleep(1)  # Rate limiting
            
            detected = self.detect_contradiction(contract)
            if detected is None:
                continue
                
            total_contracts += 1
            if detected:  # Should be True for contracts with contradictions
                correct_predictions += 1
                
            self.results.append({
                'contract_id': f'with_contradiction_{i+1}',
                'has_contradiction': True,
                'detected_contradiction': detected,
                'correct': detected,
                'contract_text': contract  # Save full contract text
            })
            
            print(f"    Expected: Has contradiction, Detected: {'Yes' if detected else 'No'}, Correct: {detected}")
            time.sleep(1)  # Rate limiting
        
        # Generate contracts WITHOUT contradictions
        print("\nGenerating contracts WITHOUT contradictions...")
        for i in range(num_contracts_per_type):
            print(f"  Contract {i+1}/{num_contracts_per_type}...")
            
            contract = self.generate_contract_without_contradiction()
            if contract is None:
                continue
                
            time.sleep(1)  # Rate limiting
            
            detected = self.detect_contradiction(contract)
            if detected is None:
                continue
                
            total_contracts += 1
            if not detected:  # Should be False for contracts without contradictions
                correct_predictions += 1
                
            self.results.append({
                'contract_id': f'without_contradiction_{i+1}',
                'has_contradiction': False,
                'detected_contradiction': detected,
                'correct': not detected,
                'contract_text': contract  # Save full contract text
            })
            
            print(f"    Expected: No contradiction, Detected: {'Yes' if detected else 'No'}, Correct: {not detected}")
            time.sleep(1)  # Rate limiting
        
        # Calculate and report accuracy
        accuracy = correct_predictions / total_contracts if total_contracts > 0 else 0
        
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        print(f"Total contracts tested: {total_contracts}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        print("="*60)
        
        # Save detailed results
        self.save_results(accuracy, correct_predictions, total_contracts)
        
        return accuracy
    
    def save_results(self, accuracy: float, correct: int, total: int):
        """Save experiment results to JSON file"""
        results_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'generator_model': 'o3-mini',
                'detector_model': 'gpt-4o',
                'total_contracts': total,
                'correct_predictions': correct,
                'accuracy': accuracy
            },
            'detailed_results': self.results
        }
        
        filename = f"simple_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")

def main():
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    experiment = SimpleContractExperiment()
    
    # Run experiment with 5 contracts of each type (10 total)
    accuracy = experiment.run_experiment(num_contracts_per_type=20)
    
    print(f"\nFinal Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main() 