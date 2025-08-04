#!/usr/bin/env python3
"""
Dual Model Comparison: O3-mini Generation + Dual Detection

This script:
1. Generates contracts using O3-mini
2. Tests both O3-mini and GPT-4o for contradiction detection
3. Compares performance between the two detection models
"""

import json
import os
import time
from typing import Dict, List, Tuple
import openai
from colorama import init, Fore, Style
from config import *

# Initialize colorama
init(autoreset=True)

class DualModelTester:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.results = []
    
    def generate_contract_with_o3(self, contract_type: str, has_contradictions: bool) -> Dict:
        """Generate a contract using O3-mini"""
        
        if has_contradictions:
            prompt = f"""
            Generate a comprehensive, professional {contract_type.replace('_', ' ')} legal contract that contains exactly 3-5 SUBTLE contradictions that could realistically occur in complex legal documents.

            Make the contract very detailed and comprehensive (2000-3500 words) with multiple sections covering all standard provisions for this type of contract.

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

            After generating the contract, provide a JSON object listing the contradictions:
            {{
                "contract": "FULL_CONTRACT_TEXT_HERE",
                "contradictions": [
                    "Detailed description of first contradiction with specific section references",
                    "Detailed description of second contradiction with specific section references",
                    ...
                ]
            }}

            Make sure the contradictions are subtle but substantive - they should create real legal ambiguity or conflict, not just formatting issues.
            """
        else:
            prompt = f"""
            Generate a comprehensive, professional {contract_type.replace('_', ' ')} legal contract that is completely consistent and contains NO contradictions.

            Make the contract very detailed and comprehensive (2000-3500 words) with multiple sections covering all standard provisions for this type of contract.

            Ensure that:
            - All cross-references are accurate
            - All dates are consistent
            - All payment terms align across sections
            - All definitions are used consistently
            - All obligations are clear and non-conflicting
            - Dispute resolution procedures are coherent
            - Termination clauses are internally consistent

            Return ONLY the contract text - no JSON wrapper needed for clean contracts.
            """
        
        try:
            # Use O3-mini specific parameters
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=MAX_TOKENS
            )
            
            content = response.choices[0].message.content.strip()
            
            if has_contradictions:
                # Try to parse JSON
                try:
                    if content.startswith('```json'):
                        content = content.split('```json')[1].split('```')[0].strip()
                    elif content.startswith('```'):
                        content = content.split('```')[1].split('```')[0].strip()
                    
                    data = json.loads(content)
                    return {
                        'text': data.get('contract', content),
                        'contradictions': data.get('contradictions', []),
                        'word_count': len(data.get('contract', content).split())
                    }
                except json.JSONDecodeError:
                    # Fallback: treat entire content as contract text
                    return {
                        'text': content,
                        'contradictions': ["JSON parsing failed - manual review needed"],
                        'word_count': len(content.split())
                    }
            else:
                return {
                    'text': content,
                    'contradictions': [],
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            print(f"❌ O3-mini generation failed: {e}")
            # Fallback to GPT-4o
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                content = response.choices[0].message.content.strip()
                print(f"✅ Fallback to GPT-4o successful")
                
                if has_contradictions:
                    try:
                        if content.startswith('```json'):
                            content = content.split('```json')[1].split('```')[0].strip()
                        elif content.startswith('```'):
                            content = content.split('```')[1].split('```')[0].strip()
                        
                        data = json.loads(content)
                        return {
                            'text': data.get('contract', content),
                            'contradictions': data.get('contradictions', []),
                            'word_count': len(data.get('contract', content).split())
                        }
                    except json.JSONDecodeError:
                        return {
                            'text': content,
                            'contradictions': ["JSON parsing failed - manual review needed"],
                            'word_count': len(content.split())
                        }
                else:
                    return {
                        'text': content,
                        'contradictions': [],
                        'word_count': len(content.split())
                    }
            except Exception as fallback_error:
                print(f"❌ Fallback generation also failed: {fallback_error}")
                return None
    
    def detect_contradictions_with_model(self, contract_text: str, model_name: str) -> Tuple[bool, List[str], float]:
        """Detect contradictions using specified model"""
        
        prompt = f"""
        Analyze this legal contract for contradictions. A contradiction occurs when two or more provisions in the contract conflict with each other in a way that creates ambiguity or makes it impossible for both provisions to be true simultaneously.

        Focus on SUBSTANTIVE contradictions that would cause real legal problems, not formatting issues or cross-reference errors.

        Contract text:
        {contract_text}

        Respond with ONLY a JSON object in this exact format:
        {{
            "has_contradictions": true/false,
            "contradictions": [
                "Description of contradiction 1 with specific section references",
                "Description of contradiction 2 with specific section references"
            ],
            "confidence": 0.0-1.0
        }}
        """
        
        try:
            if model_name.startswith('o3'):
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=1500
                )
            else:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.1
                )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if content.startswith('```json'):
                content = content.split('```json')[1].split('```')[0].strip()
            elif content.startswith('```'):
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            
            return (
                result.get('has_contradictions', False),
                result.get('contradictions', []),
                result.get('confidence', 0.0)
            )
            
        except Exception as e:
            print(f"❌ Detection failed with {model_name}: {e}")
            return False, [], 0.0
    
    def run_comparison(self):
        """Run the full comparison experiment"""
        
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        print(Fore.CYAN + Style.BRIGHT + "🧪 DUAL MODEL COMPARISON EXPERIMENT")
        print(Fore.CYAN + Style.BRIGHT + "Generation: O3-mini | Detection: O3-mini vs GPT-4o")
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        print()
        
        # Generate contracts
        contracts = []
        
        print(Fore.YELLOW + "📝 GENERATING CONTRACTS WITH O3-MINI...")
        print("-" * 50)
        
        # Generate clean contracts
        for i in range(NUM_CONTRACTS // 2):
            contract_type = CONTRACT_TYPES[i % len(CONTRACT_TYPES)]
            print(f"🔄 Generating clean {contract_type}...")
            
            contract_data = self.generate_contract_with_o3(contract_type, False)
            if contract_data:
                contracts.append({
                    'id': f'o3_clean_{i+1}',
                    'type': contract_type,
                    'text': contract_data['text'],
                    'ground_truth': False,
                    'known_contradictions': [],
                    'word_count': contract_data['word_count'],
                    'generator': 'o3-mini'
                })
                print(f"✅ Generated ({contract_data['word_count']} words)")
            else:
                print(f"❌ Failed to generate")
        
        # Generate contradiction contracts
        for i in range(NUM_CONTRACTS // 2):
            contract_type = CONTRACT_TYPES[(i + NUM_CONTRACTS // 2) % len(CONTRACT_TYPES)]
            print(f"🔄 Generating {contract_type} with contradictions...")
            
            contract_data = self.generate_contract_with_o3(contract_type, True)
            if contract_data:
                contracts.append({
                    'id': f'o3_contradiction_{i+1}',
                    'type': contract_type,
                    'text': contract_data['text'],
                    'ground_truth': True,
                    'known_contradictions': contract_data['contradictions'],
                    'word_count': contract_data['word_count'],
                    'generator': 'o3-mini'
                })
                print(f"✅ Generated ({contract_data['word_count']} words, {len(contract_data['contradictions'])} contradictions)")
            else:
                print(f"❌ Failed to generate")
        
        # Save generated contracts
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(f'{OUTPUT_DIR}/o3_generated_contracts.json', 'w', encoding='utf-8') as f:
            json.dump(contracts, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Generated {len(contracts)} contracts and saved to o3_generated_contracts.json")
        print(f"📊 Average word count: {sum(c['word_count'] for c in contracts) / len(contracts):.1f}")
        
        # Test both models
        model_results = {}
        
        for model_name in DETECTION_MODELS.values():
            print(f"\n{Fore.MAGENTA}🤖 TESTING {model_name.upper()} DETECTION...")
            print("-" * 50)
            
            results = []
            start_time = time.time()
            
            for contract in contracts:
                print(f"🔍 Analyzing {contract['id']} with {model_name}...")
                
                has_contradictions, detected_contradictions, confidence = self.detect_contradictions_with_model(
                    contract['text'], model_name
                )
                
                result = {
                    'contract_id': contract['id'],
                    'contract_type': contract['type'],
                    'ground_truth': contract['ground_truth'],
                    'known_contradictions': contract['known_contradictions'],
                    'predicted_contradictions': has_contradictions,
                    'detected_contradictions': detected_contradictions,
                    'confidence_score': confidence,
                    'word_count': contract['word_count'],
                    'detector_model': model_name,
                    'generator_model': 'o3-mini'
                }
                
                results.append(result)
                
                # Show result
                status = "✅" if has_contradictions == contract['ground_truth'] else "❌"
                print(f"  {status} {has_contradictions} (confidence: {confidence:.3f})")
            
            processing_time = time.time() - start_time
            model_results[model_name] = {
                'results': results,
                'processing_time': processing_time
            }
            
            print(f"⏱️  Processing time: {processing_time:.1f}s")
        
        # Calculate and compare metrics
        self.compare_model_performance(model_results)
        
        # Save detailed results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f'{RESULTS_DIR}/o3_generation_dual_detection.json', 'w', encoding='utf-8') as f:
            json.dump(model_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to o3_generation_dual_detection.json")
    
    def compare_model_performance(self, model_results: Dict):
        """Compare performance between the two models"""
        
        print(f"\n{Fore.CYAN + Style.BRIGHT}📊 PERFORMANCE COMPARISON")
        print(Fore.CYAN + Style.BRIGHT + "="*60)
        
        comparison_data = {}
        
        for model_name, data in model_results.items():
            results = data['results']
            
            # Calculate metrics
            tp = sum(1 for r in results if r['predicted_contradictions'] and r['ground_truth'])
            tn = sum(1 for r in results if not r['predicted_contradictions'] and not r['ground_truth'])
            fp = sum(1 for r in results if r['predicted_contradictions'] and not r['ground_truth'])
            fn = sum(1 for r in results if not r['predicted_contradictions'] and r['ground_truth'])
            
            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            avg_confidence = sum(r['confidence_score'] for r in results) / len(results)
            
            comparison_data[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity,
                'avg_confidence': avg_confidence,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'processing_time': data['processing_time']
            }
            
            print(f"\n{Fore.YELLOW + Style.BRIGHT}{model_name.upper()} PERFORMANCE:")
            print(f"  Accuracy:     {accuracy:.3f} ({tp+tn}/{total})")
            print(f"  Precision:    {precision:.3f} ({tp}/{tp+fp})")
            print(f"  Recall:       {recall:.3f} ({tp}/{tp+fn})")
            print(f"  F1-Score:     {f1:.3f}")
            print(f"  Specificity:  {specificity:.3f} ({tn}/{tn+fp})")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Processing Time: {data['processing_time']:.1f}s")
            print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        
        # Show winner
        print(f"\n{Fore.GREEN + Style.BRIGHT}🏆 COMPARISON SUMMARY:")
        print("-" * 40)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity']
        for metric in metrics:
            o3_score = comparison_data['o3-mini'][metric]
            gpt4o_score = comparison_data['gpt-4o'][metric]
            
            if o3_score > gpt4o_score:
                winner = "O3-mini"
                color = Fore.BLUE
            elif gpt4o_score > o3_score:
                winner = "GPT-4o"
                color = Fore.GREEN
            else:
                winner = "Tie"
                color = Fore.YELLOW
            
            print(f"  {metric.capitalize():12}: {color}{winner} ({o3_score:.3f} vs {gpt4o_score:.3f})")
        
        # Processing speed
        if comparison_data['o3-mini']['processing_time'] < comparison_data['gpt-4o']['processing_time']:
            speed_winner = "O3-mini"
            speed_color = Fore.BLUE
        else:
            speed_winner = "GPT-4o"
            speed_color = Fore.GREEN
        
        print(f"  {'Speed':12}: {speed_color}{speed_winner} ({comparison_data['o3-mini']['processing_time']:.1f}s vs {comparison_data['gpt-4o']['processing_time']:.1f}s)")

def main():
    """Main function"""
    print(f"🧪 Starting Dual Model Comparison Experiment")
    print(f"📝 Generation Model: O3-mini")
    print(f"🔍 Detection Models: O3-mini vs GPT-4o")
    print()
    
    tester = DualModelTester()
    tester.run_comparison()
    
    print(f"\n{Fore.GREEN + Style.BRIGHT}🎉 Dual model comparison complete!")

if __name__ == "__main__":
    main() 