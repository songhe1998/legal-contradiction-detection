#!/usr/bin/env python3
"""
Comprehensive Model Matrix Experiment

This script tests ALL available OpenAI models as both:
1. Contract generators
2. Contradiction detectors

Creates a full performance matrix to identify optimal model combinations.
"""

import json
import os
import time
from typing import Dict, List, Tuple, Optional
import openai
from colorama import init, Fore, Style
import pandas as pd
import numpy as np
from config import OPENAI_API_KEY

# Initialize colorama
init(autoreset=True)

# Available OpenAI models to test
AVAILABLE_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "o3",
    "o3-mini"
]

# Contract types for generation
CONTRACT_TYPES = [
    "employment_agreement",
    "service_contract", 
    "rental_agreement",
    "purchase_agreement"
]

class ComprehensiveModelTester:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.working_models = []
        self.generation_results = {}
        self.detection_results = {}
        self.performance_matrix = {}
    
    def test_model_availability(self) -> List[str]:
        """Test which models are actually available"""
        print(Fore.CYAN + Style.BRIGHT + "🔍 TESTING MODEL AVAILABILITY...")
        print("-" * 50)
        
        working_models = []
        
        for model in AVAILABLE_MODELS:
            try:
                # Test with a simple prompt
                if model.startswith('o3'):
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_completion_tokens=10
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10,
                        temperature=0.1
                    )
                
                working_models.append(model)
                print(f"✅ {model} - Available")
                
            except Exception as e:
                print(f"❌ {model} - Not available: {str(e)[:50]}...")
        
        self.working_models = working_models
        print(f"\n📊 {len(working_models)} models available: {', '.join(working_models)}")
        return working_models
    
    def generate_contract(self, model_name: str, contract_type: str, has_contradictions: bool) -> Optional[Dict]:
        """Generate a contract using specified model"""
        
        if has_contradictions:
            prompt = f"""
            Generate a comprehensive {contract_type.replace('_', ' ')} legal contract (1500-2500 words) that contains exactly 3-4 SUBTLE contradictions.

            Types of contradictions to include:
            - Conflicting dates or terms
            - Inconsistent payment amounts/schedules
            - Contradictory obligations or responsibilities
            - Conflicting jurisdiction/governing law clauses
            - Mismatched definitions vs usage

            After generating the contract, provide a JSON object:
            {{
                "contract": "FULL_CONTRACT_TEXT_HERE",
                "contradictions": [
                    "Description of contradiction 1 with specific sections",
                    "Description of contradiction 2 with specific sections",
                    ...
                ]
            }}

            Make contradictions subtle but substantive - real legal conflicts, not formatting issues.
            """
        else:
            prompt = f"""
            Generate a comprehensive, legally consistent {contract_type.replace('_', ' ')} contract (1500-2500 words) with NO contradictions.

            Ensure:
            - All cross-references are accurate
            - All dates and terms are consistent
            - All obligations are clear and non-conflicting
            - Dispute resolution is coherent
            - All sections align perfectly

            Return ONLY the contract text.
            """
        
        try:
            if model_name.startswith('o3'):
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=3000
                )
            else:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3000,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content.strip()
            
            if has_contradictions:
                # Try to parse JSON
                try:
                    if '```json' in content:
                        json_part = content.split('```json')[1].split('```')[0].strip()
                    elif '```' in content:
                        json_part = content.split('```')[1].split('```')[0].strip()
                    else:
                        json_part = content
                    
                    data = json.loads(json_part)
                    return {
                        'text': data.get('contract', content),
                        'contradictions': data.get('contradictions', []),
                        'word_count': len(data.get('contract', content).split()),
                        'generator_model': model_name
                    }
                except json.JSONDecodeError:
                    return {
                        'text': content,
                        'contradictions': ["JSON parsing failed - manual review needed"],
                        'word_count': len(content.split()),
                        'generator_model': model_name
                    }
            else:
                return {
                    'text': content,
                    'contradictions': [],
                    'word_count': len(content.split()),
                    'generator_model': model_name
                }
                
        except Exception as e:
            print(f"❌ Generation failed with {model_name}: {e}")
            return None
    
    def detect_contradictions(self, contract_text: str, detector_model: str) -> Tuple[bool, List[str], float]:
        """Detect contradictions using specified model"""
        
        prompt = f"""
        Analyze this legal contract for contradictions. Focus on SUBSTANTIVE conflicts that create legal ambiguity, not formatting issues.

        Contract:
        {contract_text[:3000]}{"..." if len(contract_text) > 3000 else ""}

        Respond with ONLY a JSON object:
        {{
            "has_contradictions": true/false,
            "contradictions": [
                "Description of contradiction 1 with sections",
                "Description of contradiction 2 with sections"
            ],
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            if detector_model.startswith('o3'):
                response = self.client.chat.completions.create(
                    model=detector_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1000
                )
            else:
                response = self.client.chat.completions.create(
                    model=detector_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            return (
                result.get('has_contradictions', False),
                result.get('contradictions', []),
                result.get('confidence', 0.0)
            )
            
        except Exception as e:
            print(f"❌ Detection failed with {detector_model}: {e}")
            return False, [], 0.0
    
    def generate_all_contracts(self):
        """Generate contracts using all available models"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}📝 GENERATING CONTRACTS WITH ALL MODELS...")
        print("=" * 60)
        
        all_contracts = []
        
        for generator_model in self.working_models:
            print(f"\n🤖 Generating with {generator_model.upper()}...")
            print("-" * 40)
            
            model_contracts = []
            
            # Generate 2 clean + 2 contradiction contracts per model
            for i, contract_type in enumerate(CONTRACT_TYPES):
                # Clean contract
                print(f"🔄 Generating clean {contract_type}...")
                contract = self.generate_contract(generator_model, contract_type, False)
                if contract:
                    contract_id = f"{generator_model}_clean_{i+1}"
                    model_contracts.append({
                        'id': contract_id,
                        'type': contract_type,
                        'text': contract['text'],
                        'ground_truth': False,
                        'known_contradictions': [],
                        'word_count': contract['word_count'],
                        'generator_model': generator_model
                    })
                    print(f"  ✅ {contract['word_count']} words")
                else:
                    print(f"  ❌ Failed")
                
                # Contradiction contract
                print(f"🔄 Generating {contract_type} with contradictions...")
                contract = self.generate_contract(generator_model, contract_type, True)
                if contract:
                    contract_id = f"{generator_model}_contradiction_{i+1}"
                    model_contracts.append({
                        'id': contract_id,
                        'type': contract_type,
                        'text': contract['text'],
                        'ground_truth': True,
                        'known_contradictions': contract['contradictions'],
                        'word_count': contract['word_count'],
                        'generator_model': generator_model
                    })
                    print(f"  ✅ {contract['word_count']} words, {len(contract['contradictions'])} contradictions")
                else:
                    print(f"  ❌ Failed")
            
            self.generation_results[generator_model] = model_contracts
            all_contracts.extend(model_contracts)
            
            print(f"📊 {generator_model}: Generated {len(model_contracts)} contracts")
        
        # Save all contracts
        os.makedirs("generated_contracts", exist_ok=True)
        with open('generated_contracts/all_models_contracts.json', 'w', encoding='utf-8') as f:
            json.dump(all_contracts, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Total contracts generated: {len(all_contracts)}")
        return all_contracts
    
    def test_all_detectors(self, all_contracts: List[Dict]):
        """Test all models as detectors on all contracts"""
        print(f"\n{Fore.MAGENTA + Style.BRIGHT}🔍 TESTING ALL MODELS AS DETECTORS...")
        print("=" * 60)
        
        detection_matrix = {}
        
        for detector_model in self.working_models:
            print(f"\n🤖 Testing {detector_model.upper()} detection...")
            print("-" * 40)
            
            detector_results = []
            start_time = time.time()
            
            for contract in all_contracts:
                print(f"🔍 {contract['id']} with {detector_model}...", end=" ")
                
                has_contradictions, detected_contradictions, confidence = self.detect_contradictions(
                    contract['text'], detector_model
                )
                
                result = {
                    'contract_id': contract['id'],
                    'generator_model': contract['generator_model'],
                    'detector_model': detector_model,
                    'contract_type': contract['type'],
                    'ground_truth': contract['ground_truth'],
                    'predicted_contradictions': has_contradictions,
                    'detected_contradictions': detected_contradictions,
                    'confidence_score': confidence,
                    'word_count': contract['word_count']
                }
                
                detector_results.append(result)
                
                # Show result
                status = "✅" if has_contradictions == contract['ground_truth'] else "❌"
                print(f"{status} ({confidence:.2f})")
            
            processing_time = time.time() - start_time
            detection_matrix[detector_model] = {
                'results': detector_results,
                'processing_time': processing_time
            }
            
            print(f"⏱️  {detector_model}: {processing_time:.1f}s total")
        
        self.detection_results = detection_matrix
        
        # Save detection results
        os.makedirs("results", exist_ok=True)
        with open('results/comprehensive_detection_matrix.json', 'w', encoding='utf-8') as f:
            json.dump(detection_matrix, f, indent=2, ensure_ascii=False)
        
        return detection_matrix
    
    def calculate_performance_matrix(self):
        """Calculate comprehensive performance metrics"""
        print(f"\n{Fore.CYAN + Style.BRIGHT}📊 CALCULATING PERFORMANCE MATRIX...")
        print("=" * 60)
        
        # Create performance matrix
        matrix_data = []
        
        for detector_model in self.working_models:
            detector_results = self.detection_results[detector_model]['results']
            
            # Group by generator model
            for generator_model in self.working_models:
                # Get results for this generator-detector pair
                pair_results = [r for r in detector_results if r['generator_model'] == generator_model]
                
                if not pair_results:
                    continue
                
                # Calculate metrics
                tp = sum(1 for r in pair_results if r['predicted_contradictions'] and r['ground_truth'])
                tn = sum(1 for r in pair_results if not r['predicted_contradictions'] and not r['ground_truth'])
                fp = sum(1 for r in pair_results if r['predicted_contradictions'] and not r['ground_truth'])
                fn = sum(1 for r in pair_results if not r['predicted_contradictions'] and r['ground_truth'])
                
                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                avg_confidence = sum(r['confidence_score'] for r in pair_results) / len(pair_results)
                
                matrix_data.append({
                    'Generator': generator_model,
                    'Detector': detector_model,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'Specificity': specificity,
                    'Avg_Confidence': avg_confidence,
                    'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                    'Total_Contracts': total,
                    'Same_Model': generator_model == detector_model
                })
        
        # Create DataFrame
        df = pd.DataFrame(matrix_data)
        
        # Save to CSV
        df.to_csv('results/performance_matrix.csv', index=False)
        
        self.performance_matrix = df
        return df
    
    def display_results(self, df: pd.DataFrame):
        """Display comprehensive results"""
        print(f"\n{Fore.GREEN + Style.BRIGHT}🏆 COMPREHENSIVE RESULTS")
        print("=" * 80)
        
        # Best overall performers
        print(f"\n{Fore.YELLOW}📈 TOP PERFORMERS BY METRIC:")
        print("-" * 40)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
        for metric in metrics:
            best = df.loc[df[metric].idxmax()]
            print(f"{metric:12}: {best['Generator']:>12} → {best['Detector']:>12} ({best[metric]:.3f})")
        
        # Same-model vs cross-model performance
        print(f"\n{Fore.YELLOW}🔄 SAME-MODEL vs CROSS-MODEL ANALYSIS:")
        print("-" * 40)
        
        same_model_df = df[df['Same_Model'] == True]
        cross_model_df = df[df['Same_Model'] == False]
        
        for metric in ['Accuracy', 'F1_Score']:
            same_avg = same_model_df[metric].mean()
            cross_avg = cross_model_df[metric].mean()
            better = "Same-Model" if same_avg > cross_avg else "Cross-Model"
            print(f"{metric:12}: Same-Model {same_avg:.3f} vs Cross-Model {cross_avg:.3f} → {better} wins")
        
        # Model-specific insights
        print(f"\n{Fore.YELLOW}🤖 MODEL-SPECIFIC INSIGHTS:")
        print("-" * 40)
        
        # Best generators
        gen_performance = df.groupby('Generator')['Accuracy'].mean().sort_values(ascending=False)
        print("Best Generators (avg accuracy when others detect):")
        for model, score in gen_performance.head(3).items():
            print(f"  {model:>15}: {score:.3f}")
        
        # Best detectors
        det_performance = df.groupby('Detector')['Accuracy'].mean().sort_values(ascending=False)
        print("\nBest Detectors (avg accuracy across all contracts):")
        for model, score in det_performance.head(3).items():
            print(f"  {model:>15}: {score:.3f}")
        
        # Performance heatmap data
        print(f"\n{Fore.YELLOW}🔥 ACCURACY HEATMAP:")
        print("-" * 40)
        print("Generator\\Detector", end="")
        for detector in self.working_models:
            print(f"{detector:>12}", end="")
        print()
        
        for generator in self.working_models:
            print(f"{generator:>15}", end="")
            for detector in self.working_models:
                pair_data = df[(df['Generator'] == generator) & (df['Detector'] == detector)]
                if not pair_data.empty:
                    accuracy = pair_data['Accuracy'].iloc[0]
                    color = Fore.GREEN if accuracy > 0.8 else Fore.YELLOW if accuracy > 0.6 else Fore.RED
                    print(f"{color}{accuracy:>12.3f}", end="")
                else:
                    print(f"{Fore.WHITE}{'-':>12}", end="")
            print()
    
    def run_comprehensive_experiment(self):
        """Run the complete multi-model experiment"""
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        print(Fore.CYAN + Style.BRIGHT + "🧪 COMPREHENSIVE MODEL MATRIX EXPERIMENT")
        print(Fore.CYAN + Style.BRIGHT + "Testing ALL models as generators AND detectors")
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        print()
        
        # Step 1: Test model availability
        working_models = self.test_model_availability()
        if len(working_models) < 2:
            print("❌ Need at least 2 working models for comparison")
            return
        
        # Step 2: Generate contracts with all models
        all_contracts = self.generate_all_contracts()
        
        # Step 3: Test all models as detectors
        detection_matrix = self.test_all_detectors(all_contracts)
        
        # Step 4: Calculate performance matrix
        performance_df = self.calculate_performance_matrix()
        
        # Step 5: Display comprehensive results
        self.display_results(performance_df)
        
        print(f"\n{Fore.GREEN + Style.BRIGHT}🎉 COMPREHENSIVE EXPERIMENT COMPLETE!")
        print(f"📊 Results saved to:")
        print(f"  • generated_contracts/all_models_contracts.json")
        print(f"  • results/comprehensive_detection_matrix.json") 
        print(f"  • results/performance_matrix.csv")

def main():
    """Main function"""
    print("🚀 Starting Comprehensive Model Matrix Experiment")
    print("🎯 Testing all available OpenAI models as both generators and detectors")
    print()
    
    tester = ComprehensiveModelTester()
    tester.run_comprehensive_experiment()

if __name__ == "__main__":
    main() 