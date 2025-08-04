#!/usr/bin/env python3
"""
Multi-Model Detection Test
Tests detection accuracy of different models on fresh contract data
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

class MultiModelDetectionTest:
    def __init__(self):
        self.client = openai.OpenAI()
        self.contracts = []
        self.results = {}
        
        # Models to test (using available model names)
        self.models = [
            "gpt-4o",      # We know this works from earlier
            "o3-mini"      # We know this works
        ]
    
    def generate_test_contract(self, with_contradiction: bool) -> str:
        """Generate a fresh contract for testing"""
        if with_contradiction:
            prompt = """Generate a very realistic contract content that has contradictions in it, around 2-3 pages. 
            Make it look like a real business contract with proper legal language, but include some contradictory clauses."""
        else:
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
            print(f"Error generating contract: {e}")
            return None
    
    def generate_test_dataset(self, num_per_type: int = 5):
        """Generate a small test dataset of contracts"""
        print(f"Generating {num_per_type * 2} test contracts...")
        
        # Generate contracts with contradictions
        for i in range(num_per_type):
            print(f"  Generating contract with contradiction {i+1}/{num_per_type}...")
            contract = self.generate_test_contract(with_contradiction=True)
            if contract:
                self.contracts.append({
                    'contract_id': f'with_contradiction_{i+1}',
                    'has_contradiction': True,
                    'contract_text': contract
                })
            time.sleep(1)
        
        # Generate contracts without contradictions
        for i in range(num_per_type):
            print(f"  Generating contract without contradiction {i+1}/{num_per_type}...")
            contract = self.generate_test_contract(with_contradiction=False)
            if contract:
                self.contracts.append({
                    'contract_id': f'without_contradiction_{i+1}',
                    'has_contradiction': False,
                    'contract_text': contract
                })
            time.sleep(1)
        
        print(f"Generated {len(self.contracts)} test contracts")
    
    def detect_contradiction_with_model(self, contract_text: str, model: str) -> bool:
        """Use specified model to detect if there's a contradiction"""
        prompt = f"""Tell me whether there is contradiction in this document, only tell me yes or no.

Document:
{contract_text}"""
        
        try:
            if model == "o3-mini":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=10
                )
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0
                )
            
            answer = response.choices[0].message.content.strip().lower()
            return "yes" in answer
            
        except Exception as e:
            print(f"Error with model {model}: {e}")
            return None
    
    def test_model(self, model: str):
        """Test a specific model on all contracts"""
        print(f"\nTesting model: {model}")
        print("-" * 50)
        
        correct_predictions = 0
        total_contracts = 0
        model_results = []
        
        for i, contract in enumerate(self.contracts):
            print(f"  Contract {i+1}/{len(self.contracts)}...", end="")
            
            contract_text = contract['contract_text']
            detected = self.detect_contradiction_with_model(contract_text, model)
            
            if detected is None:
                print(" ERROR")
                continue
            
            expected = contract['has_contradiction']
            correct = (detected == expected)
            
            total_contracts += 1
            if correct:
                correct_predictions += 1
            
            model_results.append({
                'contract_id': contract['contract_id'],
                'expected': expected,
                'detected': detected,
                'correct': correct
            })
            
            print(f" {'✓' if correct else '✗'}")
            time.sleep(0.5)  # Rate limiting
        
        accuracy = correct_predictions / total_contracts if total_contracts > 0 else 0
        
        self.results[model] = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_contracts': total_contracts,
            'detailed_results': model_results
        }
        
        print(f"Model {model} Accuracy: {accuracy:.2%} ({correct_predictions}/{total_contracts})")
    
    def run_all_tests(self, num_per_type: int = 5):
        """Generate test data and test all models"""
        print("Multi-Model Contradiction Detection Test")
        print("="*60)
        
        # Generate fresh test contracts
        self.generate_test_dataset(num_per_type)
        
        print(f"\nTesting {len(self.contracts)} contracts with {len(self.models)} models")
        print("="*60)
        
        for model in self.models:
            try:
                self.test_model(model)
            except Exception as e:
                print(f"Failed to test model {model}: {e}")
                continue
        
        self.report_results()
        self.save_results()
    
    def report_results(self):
        """Report final results comparison"""
        print("\n" + "="*60)
        print("FINAL RESULTS COMPARISON")
        print("="*60)
        
        # Sort models by accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print(f"{'Model':<15} {'Accuracy':<10} {'Correct/Total':<15}")
        print("-" * 40)
        
        for model, result in sorted_results:
            accuracy = result['accuracy']
            correct = result['correct_predictions']
            total = result['total_contracts']
            print(f"{model:<15} {accuracy:<10.2%} {correct}/{total}")
        
        print("="*60)
    
    def save_results(self):
        """Save detailed results to JSON file"""
        results_data = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_contracts_tested': len(self.contracts),
                'models_tested': list(self.results.keys()),
                'generator_model': 'o3-mini'
            },
            'test_contracts': self.contracts,
            'model_results': self.results
        }
        
        filename = f"multi_model_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")

def main():
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    tester = MultiModelDetectionTest()
    # Test with 5 contracts of each type (10 total) for efficiency
    tester.run_all_tests(num_per_type=5)

if __name__ == "__main__":
    main() 