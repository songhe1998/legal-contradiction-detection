#!/usr/bin/env python3
"""
Demo Script for Legal Contradiction Detection Pipeline

This script runs a smaller version of the pipeline for testing purposes.
"""

import os
import sys
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Override config for demo
os.environ['NUM_CONTRACTS_WITH_CONTRADICTIONS'] = '2'
os.environ['NUM_CONTRACTS_WITHOUT_CONTRADICTIONS'] = '2'

from legal_contract_generator import LegalContractGenerator
from contradiction_detector import ContradictionDetector
from performance_evaluator import PerformanceEvaluator
from config import OPENAI_API_KEY

def run_demo():
    """Run a small demo of the pipeline"""
    
    print(Fore.CYAN + Style.BRIGHT + "🏛️  LEGAL CONTRADICTION DETECTION - DEMO MODE")
    print(Fore.CYAN + Style.BRIGHT + "="*60)
    print()
    
    # Check API key
    if not OPENAI_API_KEY:
        print(Fore.RED + "❌ Please set your OpenAI API key in .env file")
        print(Fore.RED + "   Run: python setup_env.py")
        sys.exit(1)
    
    print(Fore.GREEN + "✅ OpenAI API key found")
    print(Fore.YELLOW + "📝 Generating 4 sample contracts (2 with contradictions, 2 without)")
    print()
    
    # Generate contracts
    generator = LegalContractGenerator()
    
    # Generate 2 clean contracts
    contracts = []
    
    print(Fore.BLUE + "Generating clean contracts...")
    for i in range(2):
        contract_text = generator.generate_contract_without_contradictions("service_contract")
        contracts.append({
            'id': f"clean_{i+1}",
            'type': 'service_contract',
            'text': contract_text,
            'has_contradictions': False,
            'known_contradictions': []
        })
        print(f"  ✅ Generated clean contract {i+1}")
    
    print(Fore.BLUE + "Generating contradiction contracts...")
    for i in range(2):
        contract_text, contradictions = generator.generate_contract_with_contradictions("employment_agreement")
        contracts.append({
            'id': f"contradiction_{i+1}",
            'type': 'employment_agreement',
            'text': contract_text,
            'has_contradictions': True,
            'known_contradictions': contradictions
        })
        print(f"  ✅ Generated contradiction contract {i+1}")
    
    print()
    print(Fore.YELLOW + "🔍 Analyzing contracts for contradictions...")
    
    # Analyze contracts
    detector = ContradictionDetector()
    results = detector.analyze_contracts(contracts)
    
    print()
    print(Fore.YELLOW + "📊 Calculating performance metrics...")
    
    # Evaluate performance
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_metrics(results)
    
    # Print results
    print()
    print(Fore.CYAN + Style.BRIGHT + "DEMO RESULTS")
    print(Fore.CYAN + Style.BRIGHT + "="*30)
    
    pm = metrics['performance_metrics']
    cm = metrics['confusion_matrix']
    
    print(Fore.WHITE + Style.BRIGHT + "Performance Metrics:")
    print(f"  Accuracy:  {pm['accuracy']:.1%}")
    print(f"  Precision: {pm['precision']:.1%}")
    print(f"  Recall:    {pm['recall']:.1%}")
    print(f"  F1-Score:  {pm['f1_score']:.3f}")
    print()
    
    print(Fore.WHITE + Style.BRIGHT + "Confusion Matrix:")
    print(f"  True Positives:  {cm['true_positives']}")
    print(f"  True Negatives:  {cm['true_negatives']}")
    print(f"  False Positives: {cm['false_positives']}")
    print(f"  False Negatives: {cm['false_negatives']}")
    print()
    
    print(Fore.GREEN + Style.BRIGHT + "🎉 Demo completed successfully!")
    print(Fore.BLUE + "   To run the full pipeline, use: python main.py")

if __name__ == "__main__":
    run_demo() 