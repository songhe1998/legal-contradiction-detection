#!/usr/bin/env python3
"""
O3 Model Detection Test

This script tests the o3 model's performance on the same long contracts
to compare against GPT-4o's performance.
"""

import json
import time
from datetime import datetime
from colorama import init, Fore, Style
from contradiction_detector import ContradictionDetector
from performance_evaluator import PerformanceEvaluator
from config import *

# Initialize colorama
init(autoreset=True)

def load_existing_contracts():
    """Load the previously generated long contracts"""
    try:
        with open('generated_contracts/long_contracts.json', 'r', encoding='utf-8') as f:
            contracts = json.load(f)
        print(f"✅ Loaded {len(contracts)} existing contracts for o3 testing")
        return contracts
    except FileNotFoundError:
        print("❌ Long contracts file not found. Please run the main pipeline first.")
        return None

def test_o3_detection():
    """Test o3 model on existing contracts"""
    
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "🧠 O3 MODEL CONTRADICTION DETECTION TEST")
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print()
    
    # Load existing contracts
    contracts = load_existing_contracts()
    if not contracts:
        return
    
    print(Fore.YELLOW + f"🔍 Testing {MODEL_NAME} model on {len(contracts)} contracts...")
    print(Fore.BLUE + f"📊 Previous model (GPT-4o) achieved 81.2% accuracy")
    print(Fore.BLUE + f"🎯 Let's see if {MODEL_NAME} performs better!")
    print()
    
    # Initialize detector with o3
    detector = ContradictionDetector()
    start_time = time.time()
    
    # Run detection
    print(Fore.MAGENTA + "ANALYZING CONTRACTS WITH O3 MODEL")
    print(Fore.MAGENTA + "="*45)
    results = detector.analyze_contracts(contracts)
    
    analysis_time = time.time() - start_time
    print(f"✅ Analysis completed in {analysis_time:.1f} seconds")
    print()
    
    # Save results with o3 suffix
    detector.save_analysis_results(results, "o3_analysis_results.json")
    
    # Evaluate performance
    print(Fore.MAGENTA + "EVALUATING O3 PERFORMANCE")
    print(Fore.MAGENTA + "="*35)
    evaluator = PerformanceEvaluator()
    metrics = evaluator.calculate_metrics(results)
    
    # Print comparison report
    print_o3_comparison_report(metrics)
    
    # Save o3-specific reports
    evaluator.save_detailed_report(metrics, "o3_detailed_performance_report.json")
    evaluator.export_results_to_csv(metrics, "o3_results_summary.csv")
    
    # Create o3 visualizations
    print(Fore.BLUE + "📈 Creating o3 performance visualizations...")
    # Modify the create_visualizations to save with o3 prefix
    import matplotlib.pyplot as plt
    import os
    os.makedirs(REPORTS_DIR, exist_ok=True)
    evaluator.create_visualizations(metrics)
    
    # Rename the generated plot
    import shutil
    if os.path.exists(os.path.join(REPORTS_DIR, 'performance_analysis.png')):
        shutil.move(
            os.path.join(REPORTS_DIR, 'performance_analysis.png'),
            os.path.join(REPORTS_DIR, 'o3_performance_analysis.png')
        )
    
    print(f"Visualizations saved to {REPORTS_DIR}/o3_performance_analysis.png")
    
    return metrics

def print_o3_comparison_report(o3_metrics):
    """Print comparison between o3 and previous GPT-4o results"""
    
    print()
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "📊 O3 vs GPT-4O PERFORMANCE COMPARISON")
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print()
    
    # Previous GPT-4o results (from our experiment)
    gpt4o_results = {
        'accuracy': 0.812,
        'precision': 1.000,
        'recall': 0.625,
        'f1_score': 0.769,
        'tp': 5, 'tn': 8, 'fp': 0, 'fn': 3
    }
    
    # O3 results
    o3_pm = o3_metrics['performance_metrics']
    o3_cm = o3_metrics['confusion_matrix']
    
    print(Fore.WHITE + Style.BRIGHT + "PERFORMANCE COMPARISON:")
    print(f"{'Metric':<15} {'GPT-4o':<15} {'O3-Mini':<15} {'Change':<15}")
    print("-" * 60)
    
    # Calculate changes
    acc_change = o3_pm['accuracy'] - gpt4o_results['accuracy']
    prec_change = o3_pm['precision'] - gpt4o_results['precision']
    recall_change = o3_pm['recall'] - gpt4o_results['recall']
    f1_change = o3_pm['f1_score'] - gpt4o_results['f1_score']
    
    def format_change(change):
        if change > 0:
            return Fore.GREEN + f"+{change:.3f}" + Style.RESET_ALL
        elif change < 0:
            return Fore.RED + f"{change:.3f}" + Style.RESET_ALL
        else:
            return f"{change:.3f}"
    
    print(f"{'Accuracy':<15} {gpt4o_results['accuracy']:.3f}           {o3_pm['accuracy']:.3f}           {format_change(acc_change)}")
    print(f"{'Precision':<15} {gpt4o_results['precision']:.3f}           {o3_pm['precision']:.3f}           {format_change(prec_change)}")
    print(f"{'Recall':<15} {gpt4o_results['recall']:.3f}           {o3_pm['recall']:.3f}           {format_change(recall_change)}")
    print(f"{'F1-Score':<15} {gpt4o_results['f1_score']:.3f}           {o3_pm['f1_score']:.3f}           {format_change(f1_change)}")
    print()
    
    print(Fore.WHITE + Style.BRIGHT + "CONFUSION MATRIX COMPARISON:")
    print(f"{'Metric':<15} {'GPT-4o':<10} {'O3-Mini':<10} {'Change':<10}")
    print("-" * 45)
    print(f"{'True Positives':<15} {gpt4o_results['tp']:<10} {o3_cm['true_positives']:<10} {o3_cm['true_positives'] - gpt4o_results['tp']:+d}")
    print(f"{'True Negatives':<15} {gpt4o_results['tn']:<10} {o3_cm['true_negatives']:<10} {o3_cm['true_negatives'] - gpt4o_results['tn']:+d}")
    print(f"{'False Positives':<15} {gpt4o_results['fp']:<10} {o3_cm['false_positives']:<10} {o3_cm['false_positives'] - gpt4o_results['fp']:+d}")
    print(f"{'False Negatives':<15} {gpt4o_results['fn']:<10} {o3_cm['false_negatives']:<10} {o3_cm['false_negatives'] - gpt4o_results['fn']:+d}")
    print()
    
    # Determine winner
    if o3_pm['accuracy'] > gpt4o_results['accuracy']:
        winner = Fore.GREEN + "🏆 O3-MINI WINS!" + Style.RESET_ALL
    elif o3_pm['accuracy'] < gpt4o_results['accuracy']:
        winner = Fore.YELLOW + "🏆 GPT-4O WINS!" + Style.RESET_ALL
    else:
        winner = Fore.BLUE + "🤝 TIE!" + Style.RESET_ALL
    
    print(Fore.WHITE + Style.BRIGHT + f"OVERALL WINNER: {winner}")
    print()
    
    # Print detailed analysis
    evaluator = PerformanceEvaluator()
    evaluator.print_performance_report(o3_metrics)

def main():
    """Main function"""
    print(f"🧠 Testing {MODEL_NAME} model performance...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    metrics = test_o3_detection()
    total_time = time.time() - start_time
    
    if metrics:
        print()
        print(Fore.GREEN + Style.BRIGHT + f"🎉 O3 testing completed in {total_time:.1f} seconds!")
        print()
        print(Fore.CYAN + Style.BRIGHT + "Files generated:")
        print(f"📄 results/o3_analysis_results.json")
        print(f"📄 reports/o3_detailed_performance_report.json") 
        print(f"📄 reports/o3_results_summary.csv")
        print(f"📄 reports/o3_performance_analysis.png")

if __name__ == "__main__":
    main() 