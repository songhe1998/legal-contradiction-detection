# Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure OpenAI API Key
```bash
# Edit the .env file (already created by setup)
# Add your OpenAI API key:
OPENAI_API_KEY=your_api_key_here
```

### Step 3: Run the Pipeline
```bash
# Option 1: Use the interactive runner (Recommended)
python run_pipeline.py

# Option 2: Run demo directly (4 contracts)
python test_demo.py

# Option 3: Run full pipeline directly (50 contracts)
python main.py
```

## 📊 What You Get

After running the pipeline, you'll get:

### Performance Metrics
- **TP (True Positives)**: Contradictions correctly identified
- **TN (True Negatives)**: Clean contracts correctly identified  
- **FP (False Positives)**: Clean contracts incorrectly flagged
- **FN (False Negatives)**: Contradictions missed
- **Accuracy, Precision, Recall, F1-Score**

### Generated Files
- `generated_contracts/` - All synthetic contracts
- `results/` - Analysis results
- `reports/` - Performance reports and visualizations

### Sample Output
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

## 🎯 Project Structure

```
lcd/
├── main.py                    # Main pipeline
├── run_pipeline.py           # Interactive runner
├── test_demo.py              # Quick demo
├── setup_env.py              # Environment setup
├── legal_contract_generator.py # Contract generation
├── contradiction_detector.py  # Contradiction detection
├── performance_evaluator.py  # Performance analysis
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── README.md                 # Full documentation
├── QUICK_START.md           # This file
└── .env                     # Your API key (create this)
```

## 🔧 Troubleshooting

### Common Issues

1. **Missing API Key**
   - Edit `.env` file and add your OpenAI API key
   - Get one from: https://platform.openai.com/api-keys

2. **Package Installation Issues**
   - Try: `pip install --upgrade pip`
   - Then: `pip install -r requirements.txt`

3. **Model Access Issues**
   - The pipeline uses `o3-mini` with `gpt-4o` fallback
   - Works with most OpenAI accounts

### Need Help?
Check the full README.md for detailed documentation and troubleshooting guide.

## 🎉 You're Ready!

The pipeline is now ready to detect contradictions in legal documents using advanced AI. Run it and see how well it performs! 