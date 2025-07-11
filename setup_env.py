#!/usr/bin/env python3
"""
Environment Setup Script

This script helps users set up their environment for the legal contradiction detection pipeline.
"""

import os

def create_env_file():
    """Create .env file with template"""
    
    env_content = """# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Uncomment and modify these settings if needed
# MODEL_NAME=o3
# MAX_TOKENS=4000
# TEMPERATURE=0.1
"""
    
    if os.path.exists('.env'):
        print("⚠️  .env file already exists. Skipping creation.")
        return
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("📝 Please edit the .env file and add your OpenAI API key")
    print("   Example: OPENAI_API_KEY=sk-...")

def main():
    """Main setup function"""
    print("🔧 Setting up Legal Contradiction Detection Pipeline")
    print("="*60)
    
    # Create .env file
    create_env_file()
    
    # Create directories
    directories = ['results', 'generated_contracts', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main() 