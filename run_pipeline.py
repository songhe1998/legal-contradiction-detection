#!/usr/bin/env python3
"""
Legal Contradiction Detection Pipeline Runner

This script provides a comprehensive way to run the legal contradiction detection pipeline
with proper environment checking, error handling, and user guidance.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_packages():
    """Check if required packages are installed"""
    required_packages = [
        'openai', 'python-dotenv', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'tqdm', 'colorama'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - NOT INSTALLED")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Run: python setup_env.py")
        return False
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("❌ OpenAI API key not configured")
        print("   Please edit .env file with your API key")
        return False
    
    print("✅ .env file configured")
    return True

def run_setup():
    """Run the setup script"""
    print("\n🔧 Running setup...")
    try:
        subprocess.run([sys.executable, 'setup_env.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False

def run_demo():
    """Run the demo script"""
    print("\n🧪 Running demo...")
    try:
        subprocess.run([sys.executable, 'test_demo.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False

def run_full_pipeline():
    """Run the full pipeline"""
    print("\n🚀 Running full pipeline...")
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        return False

def main():
    """Main function"""
    print("🏛️  Legal Contradiction Detection Pipeline Runner")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking packages...")
    if not check_packages():
        print("\n❌ Please install required packages first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n🔑 Checking environment...")
    if not check_env_file():
        print("\n❌ Environment not configured. Please:")
        print("   1. Run: python setup_env.py")
        print("   2. Edit .env file with your OpenAI API key")
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Ask user what to run
    print("\nWhat would you like to run?")
    print("1. Demo (4 contracts) - Quick test")
    print("2. Full Pipeline (50 contracts) - Complete analysis")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        if run_demo():
            print("\n🎉 Demo completed successfully!")
        else:
            print("\n❌ Demo failed")
            sys.exit(1)
    
    elif choice == '2':
        print("\n⚠️  Full pipeline will take 10-20 minutes and use API credits")
        confirm = input("Continue? (y/N): ").strip().lower()
        
        if confirm == 'y':
            if run_full_pipeline():
                print("\n🎉 Full pipeline completed successfully!")
            else:
                print("\n❌ Full pipeline failed")
                sys.exit(1)
        else:
            print("Operation cancelled")
    
    elif choice == '3':
        print("Goodbye!")
        sys.exit(0)
    
    else:
        print("Invalid choice")
        sys.exit(1)

if __name__ == "__main__":
    main() 