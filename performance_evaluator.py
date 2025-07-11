import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from config import *

class PerformanceEvaluator:
    def __init__(self):
        self.results = None
        self.metrics = {}
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate all performance metrics"""
        
        # Extract predictions and ground truth
        y_true = [r['ground_truth'] for r in results]
        y_pred = [r['predicted_contradictions'] for r in results]
        
        # Calculate confusion matrix components
        tp = sum(1 for i in range(len(y_true)) if y_true[i] == True and y_pred[i] == True)
        tn = sum(1 for i in range(len(y_true)) if y_true[i] == False and y_pred[i] == False)
        fp = sum(1 for i in range(len(y_true)) if y_true[i] == False and y_pred[i] == True)
        fn = sum(1 for i in range(len(y_true)) if y_true[i] == True and y_pred[i] == False)
        
        # Calculate derived metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate confidence statistics
        confidences = [r['confidence_score'] for r in results]
        avg_confidence = np.mean(confidences)
        
        # Calculate per-contract-type performance
        contract_types = {}
        for result in results:
            contract_type = result['contract_type']
            if contract_type not in contract_types:
                contract_types[contract_type] = {'correct': 0, 'total': 0}
            
            contract_types[contract_type]['total'] += 1
            if result['ground_truth'] == result['predicted_contradictions']:
                contract_types[contract_type]['correct'] += 1
        
        # Calculate accuracy per contract type
        type_accuracy = {}
        for contract_type, stats in contract_types.items():
            type_accuracy[contract_type] = stats['correct'] / stats['total']
        
        metrics = {
            'confusion_matrix': {
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            },
            'performance_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'average_confidence': avg_confidence
            },
            'contract_type_accuracy': type_accuracy,
            'detailed_results': results
        }
        
        self.metrics = metrics
        return metrics
    
    def print_performance_report(self, metrics: Dict):
        """Print comprehensive performance report"""
        
        print("="*80)
        print("LEGAL CONTRADICTION DETECTION - PERFORMANCE REPORT")
        print("="*80)
        print()
        
        # Basic Statistics
        print("DATASET STATISTICS:")
        print(f"Total Contracts Analyzed: {len(metrics['detailed_results'])}")
        print(f"Contracts with Contradictions: {sum(1 for r in metrics['detailed_results'] if r['ground_truth'])}")
        print(f"Clean Contracts: {sum(1 for r in metrics['detailed_results'] if not r['ground_truth'])}")
        print()
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        print("CONFUSION MATRIX:")
        print(f"True Positives (TP):  {cm['true_positives']}")
        print(f"True Negatives (TN):  {cm['true_negatives']}")
        print(f"False Positives (FP): {cm['false_positives']}")
        print(f"False Negatives (FN): {cm['false_negatives']}")
        print()
        
        # Performance Metrics
        pm = metrics['performance_metrics']
        print("PERFORMANCE METRICS:")
        print(f"Accuracy:    {pm['accuracy']:.3f} ({pm['accuracy']*100:.1f}%)")
        print(f"Precision:   {pm['precision']:.3f} ({pm['precision']*100:.1f}%)")
        print(f"Recall:      {pm['recall']:.3f} ({pm['recall']*100:.1f}%)")
        print(f"Specificity: {pm['specificity']:.3f} ({pm['specificity']*100:.1f}%)")
        print(f"F1-Score:    {pm['f1_score']:.3f}")
        print(f"Avg Confidence: {pm['average_confidence']:.3f}")
        print()
        
        # Contract Type Performance
        print("PERFORMANCE BY CONTRACT TYPE:")
        for contract_type, accuracy in metrics['contract_type_accuracy'].items():
            print(f"{contract_type.replace('_', ' ').title():20s}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print()
        
        # Detailed Analysis
        print("DETAILED ANALYSIS:")
        
        # False Positives
        false_positives = [r for r in metrics['detailed_results'] 
                          if not r['ground_truth'] and r['predicted_contradictions']]
        if false_positives:
            print(f"\nFALSE POSITIVES ({len(false_positives)}):")
            for fp in false_positives:
                print(f"  - {fp['contract_id']} ({fp['contract_type']})")
                if fp['detected_contradictions']:
                    for contradiction in fp['detected_contradictions']:
                        print(f"    * {contradiction}")
        
        # False Negatives
        false_negatives = [r for r in metrics['detailed_results'] 
                          if r['ground_truth'] and not r['predicted_contradictions']]
        if false_negatives:
            print(f"\nFALSE NEGATIVES ({len(false_negatives)}):")
            for fn in false_negatives:
                print(f"  - {fn['contract_id']} ({fn['contract_type']})")
                if fn['known_contradictions']:
                    print(f"    Known contradictions:")
                    for contradiction in fn['known_contradictions']:
                        print(f"    * {contradiction}")
        
        print()
        print("="*80)
    
    def create_visualizations(self, metrics: Dict):
        """Create performance visualizations"""
        
        # Create results directory
        import os
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm_data = metrics['confusion_matrix']
        cm_matrix = np.array([[cm_data['true_positives'], cm_data['false_negatives']],
                             [cm_data['false_positives'], cm_data['true_negatives']]])
        
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted: No Contradiction', 'Predicted: Contradiction'],
                   yticklabels=['Actual: No Contradiction', 'Actual: Contradiction'],
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        
        # 2. Performance Metrics Bar Chart
        pm = metrics['performance_metrics']
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        metrics_values = [pm['accuracy'], pm['precision'], pm['recall'], pm['specificity'], pm['f1_score']]
        
        bars = axes[0,1].bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[0,1].set_title('Performance Metrics')
        axes[0,1].set_ylim(0, 1)
        axes[0,1].set_ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Contract Type Accuracy
        type_accuracy = metrics['contract_type_accuracy']
        contract_types = list(type_accuracy.keys())
        accuracies = list(type_accuracy.values())
        
        bars = axes[1,0].bar(range(len(contract_types)), accuracies, color='lightcoral')
        axes[1,0].set_title('Accuracy by Contract Type')
        axes[1,0].set_xlabel('Contract Type')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_xticks(range(len(contract_types)))
        axes[1,0].set_xticklabels([t.replace('_', ' ').title() for t in contract_types], rotation=45)
        axes[1,0].set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, accuracies):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Confidence Score Distribution
        confidences = [r['confidence_score'] for r in metrics['detailed_results']]
        axes[1,1].hist(confidences, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Distribution of Confidence Scores')
        axes[1,1].set_xlabel('Confidence Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(np.mean(confidences), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(confidences):.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {REPORTS_DIR}/performance_analysis.png")
    
    def save_detailed_report(self, metrics: Dict, filename: str = "detailed_performance_report.json"):
        """Save detailed performance report to JSON"""
        
        import os
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filepath = os.path.join(REPORTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed report saved to {filepath}")
    
    def export_results_to_csv(self, metrics: Dict, filename: str = "results_summary.csv"):
        """Export results summary to CSV"""
        
        results = metrics['detailed_results']
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'Contract ID': result['contract_id'],
                'Contract Type': result['contract_type'],
                'Ground Truth': result['ground_truth'],
                'Predicted': result['predicted_contradictions'],
                'Correct': result['ground_truth'] == result['predicted_contradictions'],
                'Confidence': result['confidence_score'],
                'Num Detected Contradictions': len(result['detected_contradictions'])
            })
        
        df = pd.DataFrame(summary_data)
        
        import os
        os.makedirs(REPORTS_DIR, exist_ok=True)
        filepath = os.path.join(REPORTS_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Results summary exported to {filepath}") 