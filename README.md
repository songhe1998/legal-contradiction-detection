# Legal Contradiction Detection Pipeline

A comprehensive pipeline for detecting contradictions in legal documents using OpenAI's o3 model. This project generates synthetic legal contracts, analyzes them for contradictions, and provides detailed performance metrics.

## 🎯 Project Overview

This pipeline addresses the critical need for automated contradiction detection in legal documents. It uses advanced AI to:

1. **Generate Realistic Legal Contracts**: Creates synthetic legal contracts with and without contradictions
2. **Detect Contradictions**: Uses OpenAI's o3 model with sophisticated prompting to identify contradictions
3. **Evaluate Performance**: Provides comprehensive performance metrics including TP, TN, FP, FN rates
4. **Generate Reports**: Creates detailed reports with visualizations and analysis

## 🚀 Features

- **Smart Contract Generation**: Generates 8 different types of legal contracts
- **Advanced Contradiction Detection**: Uses carefully crafted prompts for accurate analysis
- **Comprehensive Evaluation**: Calculates accuracy, precision, recall, F1-score, and more
- **Visual Reports**: Creates charts and graphs for performance analysis
- **Export Options**: Saves results in JSON, CSV, and PNG formats
- **Progress Tracking**: Real-time progress updates during execution

## 📋 Requirements

- Python 3.8+
- OpenAI API key with access to o3 model
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd legal-contradiction-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env file and add your OpenAI API key
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### System Configuration

The pipeline can be configured in `config.py`:

- **Contract Generation**: Number of contracts to generate
- **Model Settings**: OpenAI model parameters
- **Contract Types**: Types of legal contracts to generate
- **Output Directories**: Where to save results

## 🎮 Usage

### Running the Complete Pipeline

```bash
python main.py
```

This will execute the entire pipeline:
1. Generate synthetic legal contracts
2. Analyze them for contradictions
3. Evaluate performance
4. Generate comprehensive reports

### Running Individual Components

You can also run components separately:

```python
# Generate contracts only
from legal_contract_generator import LegalContractGenerator
generator = LegalContractGenerator()
contracts = generator.generate_all_contracts()

# Analyze contracts only
from contradiction_detector import ContradictionDetector
detector = ContradictionDetector()
results = detector.analyze_contracts(contracts)

# Evaluate performance only
from performance_evaluator import PerformanceEvaluator
evaluator = PerformanceEvaluator()
metrics = evaluator.calculate_metrics(results)
```

## 📊 Output Files

The pipeline generates several output files:

### Generated Contracts
- `generated_contracts/generated_contracts.json`: All generated contracts with metadata

### Analysis Results
- `results/analysis_results.json`: Detailed analysis results for each contract

### Performance Reports
- `reports/detailed_performance_report.json`: Comprehensive performance metrics
- `reports/results_summary.csv`: Summary of results in CSV format
- `reports/performance_analysis.png`: Performance visualizations

## 📈 Performance Metrics

The pipeline calculates comprehensive performance metrics:

### Confusion Matrix
- **True Positives (TP)**: Correctly identified contradictions
- **True Negatives (TN)**: Correctly identified clean contracts
- **False Positives (FP)**: Incorrectly flagged clean contracts
- **False Negatives (FN)**: Missed contradictions

### Derived Metrics
- **Accuracy**: Overall correctness rate
- **Precision**: Accuracy of positive predictions
- **Recall**: Completeness of positive predictions
- **Specificity**: Accuracy of negative predictions
- **F1-Score**: Harmonic mean of precision and recall

### Additional Analysis
- **Per-Contract-Type Performance**: Accuracy by contract type
- **Confidence Score Distribution**: Analysis of model confidence
- **Detailed Error Analysis**: Breakdown of false positives and negatives

## 🏗️ Architecture

### Core Components

1. **LegalContractGenerator**: Generates synthetic legal contracts
2. **ContradictionDetector**: Analyzes contracts for contradictions
3. **PerformanceEvaluator**: Calculates metrics and generates reports

### Contract Types Supported

- Employment Agreements
- Service Contracts
- Rental Agreements
- Purchase Agreements
- Partnership Agreements
- Licensing Agreements
- Confidentiality Agreements
- Consulting Agreements

### Contradiction Types Detected

- Conflicting dates and time periods
- Inconsistent payment terms
- Contradictory responsibilities
- Conflicting governing law clauses
- Mismatched definitions vs. usage
- Inconsistent termination clauses

## 🎯 Prompting Strategy

The pipeline uses sophisticated prompts designed for legal analysis:

### Contract Generation Prompts
- Structured prompts for realistic contract generation
- Specific instructions for embedding subtle contradictions
- JSON-formatted output for easy parsing

### Contradiction Detection Prompts
- Expert-level legal analysis instructions
- Comprehensive checklist of contradiction types
- Structured JSON output with confidence scores

## 📊 Sample Results

A typical run might produce results like:

```
PERFORMANCE METRICS:
Accuracy:    0.880 (88.0%)
Precision:   0.850 (85.0%)
Recall:      0.920 (92.0%)
F1-Score:    0.884

CONFUSION MATRIX:
True Positives:  23
True Negatives:  21
False Positives: 4
False Negatives: 2
```

## 🔍 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your OpenAI API key is valid
   - Check that you have access to the o3 model
   - Verify the API key is correctly set in `.env`

2. **Rate Limiting**
   - The pipeline includes automatic delays between API calls
   - Reduce batch sizes if you encounter rate limits

3. **Model Availability**
   - Ensure the o3 model is available in your region
   - Check OpenAI's model availability status

### Performance Tips

- Use a fast internet connection for API calls
- Run during off-peak hours for better API performance
- Consider running smaller batches for testing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the o3 model
- The legal technology community for inspiration
- Contributors and testers

## 📞 Support

For questions or issues, please open an issue on the repository or contact the maintainers.

---

**Note**: This pipeline is for research and educational purposes. Always consult with legal professionals for actual legal document analysis. 