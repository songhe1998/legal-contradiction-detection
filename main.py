#!/usr/bin/env python3
"""
Legal Contradiction Detection Pipeline

This script orchestrates the complete pipeline for detecting contradictions in legal documents:
1. Generate synthetic legal contracts (with and without contradictions)
2. Run contradiction detection using OpenAI's o3 model
3. Evaluate performance and generate comprehensive reports
"""

import os
import sys
import json
import time
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored output
init(autoreset=True)

# Import our modules
from legal_contract_generator import LegalContractGenerator
from contradiction_detector import ContradictionDetector
from performance_evaluator import PerformanceEvaluator
from config import *

def print_banner():
    """Print a nice banner for the pipeline"""
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "🏛️  LEGAL CONTRADICTION DETECTION PIPELINE")
    print(Fore.CYAN + Style.BRIGHT + "    Using OpenAI's Advanced Models for Legal Analysis")
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print()

def check_environment():
    """Check if the environment is properly set up"""
    print(Fore.YELLOW + "🔍 Checking environment...")
    
    if not OPENAI_API_KEY:
        print(Fore.RED + "❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print(Fore.RED + "   Please create a .env file with your OpenAI API key")
        print(Fore.RED + "   Example: OPENAI_API_KEY=your_api_key_here")
        return False
    
    print(Fore.GREEN + "✅ OpenAI API key found")
    print(Fore.GREEN + f"✅ Model: {MODEL_NAME}")
    print(Fore.GREEN + f"✅ Total contracts to generate: {TOTAL_CONTRACTS}")
    print(Fore.GREEN + f"   - With contradictions: {NUM_CONTRACTS_WITH_CONTRADICTIONS}")
    print(Fore.GREEN + f"   - Without contradictions: {NUM_CONTRACTS_WITHOUT_CONTRADICTIONS}")
    print()
    return True

def create_directories():
    """Create necessary directories"""
    print(Fore.YELLOW + "📁 Creating directories...")
    
    directories = [OUTPUT_DIR, CONTRACTS_DIR, REPORTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(Fore.GREEN + f"✅ Created/verified directory: {directory}")
    print()

def generate_contracts():
    """Generate synthetic legal contracts"""
    print(Fore.YELLOW + "📝 Generating synthetic legal contracts...")
    print(Fore.BLUE + f"   This may take several minutes as we generate {TOTAL_CONTRACTS} contracts...")
    print()
    
    generator = LegalContractGenerator()
    start_time = time.time()
    
    try:
        contracts = generator.generate_all_contracts()
        generator.save_contracts(contracts)
        
        generation_time = time.time() - start_time
        print(Fore.GREEN + f"✅ Successfully generated {len(contracts)} contracts in {generation_time:.1f} seconds")
        print()
        
        return contracts
        
    except Exception as e:
        print(Fore.RED + f"❌ Error generating contracts: {str(e)}")
        return None

def analyze_contracts(contracts):
    """Analyze contracts for contradictions"""
    print(Fore.YELLOW + "🔍 Analyzing contracts for contradictions...")
    print(Fore.BLUE + f"   This may take several minutes as we analyze {len(contracts)} contracts...")
    print()
    
    detector = ContradictionDetector()
    start_time = time.time()
    
    try:
        results = detector.analyze_contracts(contracts)
        detector.save_analysis_results(results)
        
        analysis_time = time.time() - start_time
        print(Fore.GREEN + f"✅ Successfully analyzed {len(results)} contracts in {analysis_time:.1f} seconds")
        print()
        
        return results
        
    except Exception as e:
        print(Fore.RED + f"❌ Error analyzing contracts: {str(e)}")
        return None

def evaluate_performance(results):
    """Evaluate performance and generate reports"""
    print(Fore.YELLOW + "📊 Evaluating performance and generating reports...")
    
    evaluator = PerformanceEvaluator()
    
    try:
        # Calculate metrics
        metrics = evaluator.calculate_metrics(results)
        
        # Print performance report
        evaluator.print_performance_report(metrics)
        
        # Save detailed report
        evaluator.save_detailed_report(metrics)
        
        # Export to CSV
        evaluator.export_results_to_csv(metrics)
        
        # Create visualizations
        print(Fore.BLUE + "📈 Creating performance visualizations...")
        evaluator.create_visualizations(metrics)
        
        print(Fore.GREEN + "✅ Performance evaluation completed successfully")
        print()
        
        return metrics
        
    except Exception as e:
        print(Fore.RED + f"❌ Error evaluating performance: {str(e)}")
        return None

def print_summary(metrics):
    """Print a final summary"""
    if not metrics:
        return
    
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "📋 PIPELINE EXECUTION SUMMARY")
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print()
    
    pm = metrics['performance_metrics']
    cm = metrics['confusion_matrix']
    
    print(Fore.WHITE + Style.BRIGHT + "Key Performance Metrics:")
    print(f"  🎯 Accuracy:    {pm['accuracy']:.1%}")
    print(f"  🎯 Precision:   {pm['precision']:.1%}")
    print(f"  🎯 Recall:      {pm['recall']:.1%}")
    print(f"  🎯 F1-Score:    {pm['f1_score']:.3f}")
    print()
    
    print(Fore.WHITE + Style.BRIGHT + "Confusion Matrix:")
    print(f"  ✅ True Positives:  {cm['true_positives']}")
    print(f"  ✅ True Negatives:  {cm['true_negatives']}")
    print(f"  ❌ False Positives: {cm['false_positives']}")
    print(f"  ❌ False Negatives: {cm['false_negatives']}")
    print()
    
    print(Fore.WHITE + Style.BRIGHT + "Output Files:")
    print(f"  📄 Generated contracts: {CONTRACTS_DIR}/generated_contracts.json")
    print(f"  📄 Analysis results: {OUTPUT_DIR}/analysis_results.json")
    print(f"  📄 Detailed report: {REPORTS_DIR}/detailed_performance_report.json")
    print(f"  📄 CSV summary: {REPORTS_DIR}/results_summary.csv")
    print(f"  📄 Visualizations: {REPORTS_DIR}/performance_analysis.png")
    print()
    
    print(Fore.CYAN + Style.BRIGHT + "="*80)

def main():
    """Main pipeline execution"""
    start_time = time.time()
    
    print_banner()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Step 1: Generate contracts
    print(Fore.MAGENTA + Style.BRIGHT + "STEP 1: GENERATING LEGAL CONTRACTS")
    print(Fore.MAGENTA + "="*50)
    contracts = generate_contracts()
    if not contracts:
        print(Fore.RED + "❌ Pipeline failed at contract generation step")
        sys.exit(1)
    
    # Step 2: Analyze contracts
    print(Fore.MAGENTA + Style.BRIGHT + "STEP 2: ANALYZING CONTRACTS FOR CONTRADICTIONS")
    print(Fore.MAGENTA + "="*50)
    results = analyze_contracts(contracts)
    if not results:
        print(Fore.RED + "❌ Pipeline failed at contract analysis step")
        sys.exit(1)
    
    # Step 3: Evaluate performance
    print(Fore.MAGENTA + Style.BRIGHT + "STEP 3: EVALUATING PERFORMANCE")
    print(Fore.MAGENTA + "="*50)
    metrics = evaluate_performance(results)
    if not metrics:
        print(Fore.RED + "❌ Pipeline failed at performance evaluation step")
        sys.exit(1)
    
    # Print final summary
    total_time = time.time() - start_time
    print(Fore.GREEN + Style.BRIGHT + f"🎉 Pipeline completed successfully in {total_time:.1f} seconds!")
    print()
    
    print_summary(metrics)

if __name__ == "__main__":
    main() 