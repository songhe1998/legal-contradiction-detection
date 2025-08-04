# config.py - Configuration settings for Legal Contradiction Detection

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "o3-mini"  # Using o3-mini for contract generation
FALLBACK_MODEL = "gpt-4o"  # Fallback to gpt-4o if o3-mini fails
MAX_TOKENS = 4000  # Increased for longer contracts
TEMPERATURE = 0.1  # Low temperature for consistent, analytical responses

# Generation Configuration
NUM_CONTRACTS = 8  # Reduced for testing (4 clean + 4 with contradictions)
LONG_CONTRACT_MODE = True  # Enable long contract generation
MIN_CONTRACT_LENGTH = 2000  # Minimum word count for long contracts
MAX_CONTRACT_LENGTH = 3500  # Maximum word count for long contracts

# Contract Types
CONTRACT_TYPES = [
    "employment_agreement",
    "service_contract", 
    "rental_agreement",
    "purchase_agreement",
    "consulting_agreement",
    "licensing_agreement",
    "partnership_agreement",
    "non_disclosure_agreement"
]

# Output Configuration
OUTPUT_DIR = "generated_contracts"
RESULTS_DIR = "results"

# Detection Models for Comparison
DETECTION_MODELS = {
    "o3-mini": "o3-mini",
    "gpt-4o": "gpt-4o"
} 