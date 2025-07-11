import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"  # Use gpt-4o directly since o3-mini is failing
FALLBACK_MODEL = "gpt-4o-mini"  # Fallback to mini version
MAX_TOKENS = 2500  # Reduced for faster generation
TEMPERATURE = 0.1  # Low temperature for consistent, analytical responses

# Generation Configuration - Reduced for faster testing
NUM_CONTRACTS_WITH_CONTRADICTIONS = 10  # Reduced from 25
NUM_CONTRACTS_WITHOUT_CONTRADICTIONS = 10  # Reduced from 25
TOTAL_CONTRACTS = NUM_CONTRACTS_WITH_CONTRADICTIONS + NUM_CONTRACTS_WITHOUT_CONTRADICTIONS

# Legal Contract Types
CONTRACT_TYPES = [
    "employment_agreement",
    "service_contract",
    "rental_agreement",
    "purchase_agreement",
    "partnership_agreement",
    "licensing_agreement",
    "confidentiality_agreement",
    "consulting_agreement"
]

# Output Configuration
OUTPUT_DIR = "results"
CONTRACTS_DIR = "generated_contracts"
REPORTS_DIR = "reports" 