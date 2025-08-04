#!/usr/bin/env python3
"""
Matrix Results Analysis

Analyzes the comprehensive model matrix experiment results and generates
detailed insights, visualizations, and recommendations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

class MatrixResultsAnalyzer:
    def __init__(self):
        self.performance_df = None
        self.detection_results = None
        self.contracts_data = None
    
    def load_data(self):
        """Load all experiment data"""
        print(f"{Fore.CYAN}📊 Loading experiment data...")
        
        # Load performance matrix
        self.performance_df = pd.read_csv('results/performance_matrix.csv')
        
        # Load detection results
        with open('results/comprehensive_detection_matrix.json', 'r') as f:
            self.detection_results = json.load(f)
        
        # Load contracts data
        with open('generated_contracts/all_models_contracts.json', 'r') as f:
            self.contracts_data = json.load(f)
        
        print(f"✅ Loaded {len(self.performance_df)} performance combinations")
        print(f"✅ Loaded {len(self.contracts_data)} contracts")
    
    def analyze_generation_quality(self):
        """Analyze contract generation quality by model"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}📝 GENERATION QUALITY ANALYSIS")
        print("=" * 60)
        
        # Group contracts by generator model
        generation_stats = {}
        
        for contract in self.contracts_data:
            model = contract['generator_model']
            if model not in generation_stats:
                generation_stats[model] = {
                    'word_counts': [],
                    'clean_contracts': 0,
                    'contradiction_contracts': 0,
                    'total_contradictions': 0
                }
            
            generation_stats[model]['word_counts'].append(contract['word_count'])
            
            if contract['ground_truth']:
                generation_stats[model]['contradiction_contracts'] += 1
                generation_stats[model]['total_contradictions'] += len(contract['known_contradictions'])
            else:
                generation_stats[model]['clean_contracts'] += 1
        
        # Display generation quality
        for model, stats in generation_stats.items():
            avg_words = np.mean(stats['word_counts'])
            min_words = min(stats['word_counts'])
            max_words = max(stats['word_counts'])
            
            print(f"\n{Fore.CYAN}{model.upper()}:")
            print(f"  📊 Word Count: {avg_words:.1f} avg ({min_words}-{max_words} range)")
            print(f"  📄 Clean contracts: {stats['clean_contracts']}")
            print(f"  ⚠️  Contradiction contracts: {stats['contradiction_contracts']}")
            print(f"  🔍 Total contradictions: {stats['total_contradictions']}")
            
            # Flag potential issues
            if min_words < 100:
                print(f"  🚨 WARNING: Some very short contracts (min: {min_words} words)")
            if avg_words < 500:
                print(f"  ⚠️  CONCERN: Below target length (avg: {avg_words:.1f} words)")
    
    def analyze_detection_patterns(self):
        """Analyze detection patterns and biases"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}🔍 DETECTION PATTERN ANALYSIS")
        print("=" * 60)
        
        # Same-model vs cross-model performance
        same_model_performance = self.performance_df[self.performance_df['Same_Model'] == True]
        cross_model_performance = self.performance_df[self.performance_df['Same_Model'] == False]
        
        print(f"\n{Fore.CYAN}🔄 SAME-MODEL vs CROSS-MODEL COMPARISON:")
        print("-" * 40)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'Specificity']
        for metric in metrics:
            same_avg = same_model_performance[metric].mean()
            cross_avg = cross_model_performance[metric].mean()
            difference = abs(same_avg - cross_avg)
            
            if same_avg > cross_avg:
                winner = f"{Fore.BLUE}Same-Model"
                loser = "Cross-Model"
            else:
                winner = f"{Fore.GREEN}Cross-Model"
                loser = "Same-Model"
            
            print(f"  {metric:12}: {winner} wins by {difference:.3f} ({same_avg:.3f} vs {cross_avg:.3f})")
        
        # Model-specific biases
        print(f"\n{Fore.CYAN}🤖 MODEL-SPECIFIC BIASES:")
        print("-" * 40)
        
        for model in self.performance_df['Detector'].unique():
            model_results = self.performance_df[self.performance_df['Detector'] == model]
            same_model_acc = model_results[model_results['Same_Model'] == True]['Accuracy'].mean()
            cross_model_acc = model_results[model_results['Same_Model'] == False]['Accuracy'].mean()
            
            bias = same_model_acc - cross_model_acc
            bias_type = "Self-favorable" if bias > 0.1 else "Cross-favorable" if bias < -0.1 else "Neutral"
            
            print(f"  {model:>10}: {bias_type:>15} bias ({bias:+.3f})")
    
    def identify_optimal_combinations(self):
        """Identify the best model combinations"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}🏆 OPTIMAL MODEL COMBINATIONS")
        print("=" * 60)
        
        # Sort by different metrics
        metrics_to_analyze = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for metric in metrics_to_analyze:
            top_3 = self.performance_df.nlargest(3, metric)
            
            print(f"\n{Fore.GREEN}🥇 TOP 3 BY {metric.upper()}:")
            print("-" * 30)
            
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                same_model_indicator = " (Self)" if row['Same_Model'] else " (Cross)"
                
                print(f"  {emoji} {row['Generator']} → {row['Detector']}{same_model_indicator}")
                print(f"      {metric}: {row[metric]:.3f} | Confidence: {row['Avg_Confidence']:.3f}")
    
    def analyze_problematic_cases(self):
        """Identify and analyze problematic cases"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}🚨 PROBLEMATIC CASES ANALYSIS")
        print("=" * 60)
        
        # Low performance cases
        low_performance = self.performance_df[self.performance_df['Accuracy'] < 0.3]
        
        if not low_performance.empty:
            print(f"\n{Fore.RED}⚠️  LOW PERFORMANCE COMBINATIONS (Accuracy < 30%):")
            print("-" * 50)
            
            for _, row in low_performance.iterrows():
                print(f"  ❌ {row['Generator']} → {row['Detector']}: {row['Accuracy']:.3f} accuracy")
                print(f"      Issues: TP={row['TP']}, TN={row['TN']}, FP={row['FP']}, FN={row['FN']}")
        
        # High confidence but wrong predictions
        wrong_but_confident = []
        for detector_model, data in self.detection_results.items():
            for result in data['results']:
                if (result['predicted_contradictions'] != result['ground_truth'] and 
                    result['confidence_score'] > 0.8):
                    wrong_but_confident.append({
                        'detector': detector_model,
                        'contract': result['contract_id'],
                        'generator': result['generator_model'],
                        'confidence': result['confidence_score'],
                        'predicted': result['predicted_contradictions'],
                        'actual': result['ground_truth']
                    })
        
        if wrong_but_confident:
            print(f"\n{Fore.RED}🎯 OVERCONFIDENT WRONG PREDICTIONS (>80% confidence):")
            print("-" * 50)
            
            for case in wrong_but_confident[:5]:  # Show top 5
                print(f"  🚨 {case['detector']} analyzing {case['generator']} contract")
                print(f"      Predicted: {case['predicted']} | Actual: {case['actual']} | Confidence: {case['confidence']:.3f}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}📈 CREATING VISUALIZATIONS...")
        print("=" * 60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a comprehensive dashboard
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('Legal Contradiction Detection: Comprehensive Model Matrix Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Accuracy Heatmap
        ax1 = plt.subplot(3, 3, 1)
        pivot_accuracy = self.performance_df.pivot(index='Generator', columns='Detector', values='Accuracy')
        sns.heatmap(pivot_accuracy, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Accuracy'}, ax=ax1)
        ax1.set_title('Accuracy Heatmap\n(Generator → Detector)', fontweight='bold')
        
        # 2. F1-Score Heatmap
        ax2 = plt.subplot(3, 3, 2)
        pivot_f1 = self.performance_df.pivot(index='Generator', columns='Detector', values='F1_Score')
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'F1-Score'}, ax=ax2)
        ax2.set_title('F1-Score Heatmap\n(Generator → Detector)', fontweight='bold')
        
        # 3. Same-Model vs Cross-Model Performance
        ax3 = plt.subplot(3, 3, 3)
        same_model_data = self.performance_df[self.performance_df['Same_Model'] == True]
        cross_model_data = self.performance_df[self.performance_df['Same_Model'] == False]
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        same_means = [same_model_data[metric].mean() for metric in metrics]
        cross_means = [cross_model_data[metric].mean() for metric in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, same_means, width, label='Same-Model', alpha=0.8)
        ax3.bar(x + width/2, cross_means, width, label='Cross-Model', alpha=0.8)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Same-Model vs Cross-Model\nPerformance', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Model Performance as Generators
        ax4 = plt.subplot(3, 3, 4)
        gen_performance = self.performance_df.groupby('Generator')['Accuracy'].mean().sort_values(ascending=True)
        gen_performance.plot(kind='barh', ax=ax4, color='skyblue')
        ax4.set_title('Average Accuracy by Generator\n(How well others detect their contracts)', fontweight='bold')
        ax4.set_xlabel('Average Accuracy')
        ax4.grid(True, alpha=0.3)
        
        # 5. Model Performance as Detectors
        ax5 = plt.subplot(3, 3, 5)
        det_performance = self.performance_df.groupby('Detector')['Accuracy'].mean().sort_values(ascending=True)
        det_performance.plot(kind='barh', ax=ax5, color='lightcoral')
        ax5.set_title('Average Accuracy by Detector\n(How well they detect all contracts)', fontweight='bold')
        ax5.set_xlabel('Average Accuracy')
        ax5.grid(True, alpha=0.3)
        
        # 6. Confidence vs Accuracy Scatter
        ax6 = plt.subplot(3, 3, 6)
        colors = ['red' if same else 'blue' for same in self.performance_df['Same_Model']]
        scatter = ax6.scatter(self.performance_df['Avg_Confidence'], self.performance_df['Accuracy'], 
                            c=colors, alpha=0.6, s=100)
        ax6.set_xlabel('Average Confidence')
        ax6.set_ylabel('Accuracy')
        ax6.set_title('Confidence vs Accuracy\n(Red=Same-Model, Blue=Cross-Model)', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add diagonal line for perfect calibration
        ax6.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax6.legend()
        
        # 7. Contract Generation Quality
        ax7 = plt.subplot(3, 3, 7)
        generation_stats = {}
        for contract in self.contracts_data:
            model = contract['generator_model']
            if model not in generation_stats:
                generation_stats[model] = []
            generation_stats[model].append(contract['word_count'])
        
        models = list(generation_stats.keys())
        word_counts = [generation_stats[model] for model in models]
        
        bp = ax7.boxplot(word_counts, labels=models, patch_artist=True)
        ax7.set_title('Contract Length Distribution\nby Generator Model', fontweight='bold')
        ax7.set_ylabel('Word Count')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # 8. Processing Time Comparison
        ax8 = plt.subplot(3, 3, 8)
        processing_times = {}
        for detector_model, data in self.detection_results.items():
            processing_times[detector_model] = data['processing_time']
        
        models = list(processing_times.keys())
        times = list(processing_times.values())
        
        bars = ax8.bar(models, times, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax8.set_title('Processing Time by Detector\n(Total for 32 contracts)', fontweight='bold')
        ax8.set_ylabel('Time (seconds)')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time:.1f}s', ha='center', va='bottom')
        
        # 9. Confusion Matrix Summary
        ax9 = plt.subplot(3, 3, 9)
        
        # Calculate overall confusion matrix
        total_tp = self.performance_df['TP'].sum()
        total_tn = self.performance_df['TN'].sum()
        total_fp = self.performance_df['FP'].sum()
        total_fn = self.performance_df['FN'].sum()
        
        confusion_matrix = np.array([[total_tn, total_fp],
                                   [total_fn, total_tp]])
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted No', 'Predicted Yes'],
                   yticklabels=['Actual No', 'Actual Yes'], ax=ax9)
        ax9.set_title('Overall Confusion Matrix\n(All Model Combinations)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_matrix_analysis.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved comprehensive visualization to 'results/comprehensive_matrix_analysis.png'")
        
        # Show the plot
        plt.show()
    
    def generate_final_report(self):
        """Generate a comprehensive final report"""
        print(f"\n{Fore.YELLOW + Style.BRIGHT}📋 GENERATING FINAL REPORT...")
        print("=" * 60)
        
        # Best overall combination
        best_overall = self.performance_df.loc[self.performance_df['Accuracy'].idxmax()]
        
        # Most reliable combination (high accuracy + high confidence)
        reliability_score = self.performance_df['Accuracy'] * self.performance_df['Avg_Confidence']
        most_reliable = self.performance_df.loc[reliability_score.idxmax()]
        
        # Generate report
        report = f"""
# COMPREHENSIVE MODEL MATRIX ANALYSIS REPORT

## 🎯 EXECUTIVE SUMMARY

This experiment tested 4 OpenAI models (GPT-4.1, GPT-4o, O3, O3-mini) as both contract generators and contradiction detectors, creating a 4×4 performance matrix with 32 contracts and 128 detection tests.

## 🏆 KEY FINDINGS

### Best Overall Performance
- **Combination**: {best_overall['Generator']} → {best_overall['Detector']}
- **Accuracy**: {best_overall['Accuracy']:.3f}
- **F1-Score**: {best_overall['F1_Score']:.3f}
- **Type**: {'Same-Model' if best_overall['Same_Model'] else 'Cross-Model'} detection

### Most Reliable Combination
- **Combination**: {most_reliable['Generator']} → {most_reliable['Detector']}
- **Reliability Score**: {reliability_score.max():.3f}
- **Accuracy**: {most_reliable['Accuracy']:.3f}
- **Confidence**: {most_reliable['Avg_Confidence']:.3f}

## 📊 PERFORMANCE INSIGHTS

### Generator Rankings (by avg accuracy when others detect):
"""
        
        gen_performance = self.performance_df.groupby('Generator')['Accuracy'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(gen_performance.items(), 1):
            report += f"{i}. **{model}**: {score:.3f}\n"
        
        report += f"\n### Detector Rankings (by avg accuracy across all contracts):\n"
        
        det_performance = self.performance_df.groupby('Detector')['Accuracy'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(det_performance.items(), 1):
            report += f"{i}. **{model}**: {score:.3f}\n"
        
        # Same-model vs cross-model analysis
        same_model_acc = self.performance_df[self.performance_df['Same_Model'] == True]['Accuracy'].mean()
        cross_model_acc = self.performance_df[self.performance_df['Same_Model'] == False]['Accuracy'].mean()
        
        report += f"""
## 🔄 SAME-MODEL vs CROSS-MODEL ANALYSIS

- **Same-Model Average Accuracy**: {same_model_acc:.3f}
- **Cross-Model Average Accuracy**: {cross_model_acc:.3f}
- **Winner**: {'Same-Model' if same_model_acc > cross_model_acc else 'Cross-Model'} detection
- **Difference**: {abs(same_model_acc - cross_model_acc):.3f}

## 🚨 CRITICAL ISSUES IDENTIFIED

### Generation Quality Problems:
"""
        
        # Identify generation issues
        for contract in self.contracts_data:
            if contract['word_count'] < 100:
                report += f"- **{contract['generator_model']}**: Generated very short contract ({contract['word_count']} words) - {contract['id']}\n"
        
        report += f"""
### Detection Reliability Issues:
"""
        
        # Find models with high variance in confidence
        confidence_variance = self.performance_df.groupby('Detector')['Avg_Confidence'].std().sort_values(ascending=False)
        for model, variance in confidence_variance.items():
            if variance > 0.3:
                report += f"- **{model}**: High confidence variance ({variance:.3f}) - inconsistent certainty\n"
        
        report += f"""
## 💡 RECOMMENDATIONS

### For Production Use:
1. **Best Generator**: {gen_performance.index[0]} (most detectable contradictions)
2. **Best Detector**: {det_performance.index[0]} (highest overall accuracy)
3. **Recommended Pipeline**: {gen_performance.index[0]} for generation → {det_performance.index[0]} for detection

### For Further Research:
1. Investigate why some models generate very short contracts
2. Study the cross-model detection bias patterns
3. Improve JSON parsing for contradiction extraction
4. Test with longer, more complex legal documents

## 📈 STATISTICAL SUMMARY

- **Total Contracts Generated**: {len(self.contracts_data)}
- **Total Detection Tests**: {len(self.performance_df) * 32 // 16}
- **Average Accuracy**: {self.performance_df['Accuracy'].mean():.3f}
- **Best Accuracy**: {self.performance_df['Accuracy'].max():.3f}
- **Worst Accuracy**: {self.performance_df['Accuracy'].min():.3f}
- **Average Processing Time**: {sum(data['processing_time'] for data in self.detection_results.values()) / len(self.detection_results):.1f}s per model

---
*Generated by Legal Contradiction Detection Matrix Analysis*
"""
        
        # Save report
        with open('results/COMPREHENSIVE_MATRIX_REPORT.md', 'w') as f:
            f.write(report)
        
        print(f"✅ Saved comprehensive report to 'results/COMPREHENSIVE_MATRIX_REPORT.md'")
        
        # Print key insights
        print(f"\n{Fore.GREEN + Style.BRIGHT}🎯 KEY INSIGHTS:")
        print("-" * 40)
        print(f"🏆 Best Overall: {best_overall['Generator']} → {best_overall['Detector']} ({best_overall['Accuracy']:.3f})")
        print(f"🔒 Most Reliable: {most_reliable['Generator']} → {most_reliable['Detector']} ({reliability_score.max():.3f})")
        print(f"📈 Top Generator: {gen_performance.index[0]} ({gen_performance.iloc[0]:.3f})")
        print(f"🔍 Top Detector: {det_performance.index[0]} ({det_performance.iloc[0]:.3f})")
    
    def run_analysis(self):
        """Run the complete analysis"""
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        print(Fore.CYAN + Style.BRIGHT + "🔬 COMPREHENSIVE MATRIX RESULTS ANALYSIS")
        print(Fore.CYAN + Style.BRIGHT + "="*80)
        
        self.load_data()
        self.analyze_generation_quality()
        self.analyze_detection_patterns()
        self.identify_optimal_combinations()
        self.analyze_problematic_cases()
        self.create_visualizations()
        self.generate_final_report()
        
        print(f"\n{Fore.GREEN + Style.BRIGHT}🎉 ANALYSIS COMPLETE!")

def main():
    analyzer = MatrixResultsAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 