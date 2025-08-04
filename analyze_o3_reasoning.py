#!/usr/bin/env python3
"""
O3 Reasoning Analysis

This script asks O3 to show detailed step-by-step reasoning
for its contradiction detection decisions.
"""

import json
import openai
from colorama import init, Fore, Style
from config import *

# Initialize colorama
init(autoreset=True)

class O3ReasoningAnalyzer:
    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    def analyze_with_reasoning(self, contract_text: str, contract_id: str, ground_truth: bool):
        """Ask O3 to analyze a contract with detailed reasoning"""
        
        prompt = f"""
        You are an expert legal analyst. I want you to analyze this legal contract for contradictions, but I want to see your COMPLETE reasoning process.

        CONTRACT TO ANALYZE:
        {contract_text}

        ANALYSIS INSTRUCTIONS:
        1. Think step-by-step through your analysis
        2. Show your reasoning for each potential issue you identify
        3. Explain why something is or isn't a contradiction
        4. Distinguish between formatting issues vs substantive contradictions

        RESPONSE FORMAT:
        Please provide your analysis in this exact format:

        STEP-BY-STEP REASONING:
        [Show your detailed thought process here - what you're looking for, what you find, why it matters]

        POTENTIAL ISSUES IDENTIFIED:
        [List each potential issue and your reasoning about whether it's actually a contradiction]

        FINAL DECISION:
        Has contradictions: [true/false]
        Confidence: [0.0-1.0]

        CONTRADICTIONS FOUND:
        [If any, list them with specific reasoning]

        IMPORTANT: 
        - Show your complete reasoning process
        - Explain the difference between cross-reference errors and actual contradictions
        - Consider that legal documents may have formatting flexibility
        - Focus on substantive conflicts that would cause legal problems
        """
        
        try:
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000  # More tokens for detailed reasoning
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting O3 reasoning: {e}")
            return None

def analyze_specific_contracts():
    """Analyze specific contracts with detailed O3 reasoning"""
    
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print(Fore.CYAN + Style.BRIGHT + "🧠 O3 DETAILED REASONING ANALYSIS")
    print(Fore.CYAN + Style.BRIGHT + "="*80)
    print()
    
    # Load contracts
    with open('generated_contracts/long_contracts.json', 'r', encoding='utf-8') as f:
        contracts = json.load(f)
    
    analyzer = O3ReasoningAnalyzer()
    
    # Analyze interesting cases
    interesting_cases = [
        # A clean contract that O3 incorrectly flagged (false positive)
        ("long_clean_3", "employment_agreement", False, "O3 False Positive"),
        # A clean contract that O3 correctly identified (true negative) 
        ("long_clean_1", "rental_agreement", False, "O3 True Negative"),
        # A contradiction contract that O3 correctly identified (true positive)
        ("long_contradiction_3", "service_contract", True, "O3 True Positive"),
        # A contradiction contract that O3 missed (false negative)
        ("long_contradiction_1", "purchase_agreement", True, "O3 False Negative")
    ]
    
    for contract_id, contract_type, ground_truth, case_type in interesting_cases:
        print(Fore.YELLOW + f"\n📋 ANALYZING: {contract_id} ({case_type})")
        print(Fore.BLUE + f"Contract Type: {contract_type}")
        print(Fore.BLUE + f"Ground Truth: {'Has contradictions' if ground_truth else 'Clean contract'}")
        print(Fore.MAGENTA + "="*60)
        
        # Find the contract
        contract = next((c for c in contracts if c['id'] == contract_id), None)
        if not contract:
            print(f"❌ Contract {contract_id} not found")
            continue
        
        # Get O3's reasoning
        reasoning = analyzer.analyze_with_reasoning(
            contract['text'], 
            contract_id, 
            ground_truth
        )
        
        if reasoning:
            print(Fore.WHITE + reasoning)
        else:
            print("❌ Failed to get reasoning")
        
        print(Fore.CYAN + "\n" + "="*80)
        
        # Ask user if they want to continue
        user_input = input(f"\n{Fore.GREEN}Press Enter to continue to next contract, or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
    
    print(Fore.GREEN + Style.BRIGHT + "\n🎉 Reasoning analysis complete!")

def compare_reasoning_styles():
    """Compare O3 vs GPT-4o reasoning on the same contract"""
    
    print(Fore.CYAN + Style.BRIGHT + "\n🔍 COMPARING O3 vs GPT-4O REASONING")
    print(Fore.CYAN + Style.BRIGHT + "="*50)
    
    # Load a false positive case
    with open('generated_contracts/long_contracts.json', 'r', encoding='utf-8') as f:
        contracts = json.load(f)
    
    # Get the employment contract that O3 flagged but GPT-4o didn't
    contract = next((c for c in contracts if c['id'] == 'long_clean_3'), None)
    
    if not contract:
        print("❌ Contract not found")
        return
    
    print(Fore.YELLOW + "📋 Contract: long_clean_3 (Employment Agreement)")
    print(Fore.BLUE + "Ground Truth: Clean contract (no contradictions)")
    print(Fore.RED + "O3 Result: Found contradictions (FALSE POSITIVE)")
    print(Fore.GREEN + "GPT-4o Result: No contradictions (CORRECT)")
    print()
    
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Get GPT-4o reasoning
    print(Fore.MAGENTA + "🤖 GPT-4O REASONING:")
    print("-" * 30)
    
    gpt4o_prompt = f"""
    Analyze this employment contract for contradictions. Show your reasoning process:

    {contract['text'][:2000]}...

    Explain step-by-step why you think this contract does or doesn't have contradictions.
    """
    
    try:
        gpt4o_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": gpt4o_prompt}],
            max_tokens=1000,
            temperature=0.1
        )
        print(Fore.WHITE + gpt4o_response.choices[0].message.content)
    except Exception as e:
        print(f"❌ GPT-4o failed: {e}")
    
    print(Fore.MAGENTA + "\n🧠 O3-MINI REASONING:")
    print("-" * 30)
    
    # Get O3 reasoning (we already have this from the previous analysis)
    analyzer = O3ReasoningAnalyzer()
    o3_reasoning = analyzer.analyze_with_reasoning(contract['text'], 'long_clean_3', False)
    if o3_reasoning:
        print(Fore.WHITE + o3_reasoning)
    else:
        print("❌ O3 reasoning failed")

def main():
    """Main function"""
    print(f"🧠 Analyzing O3-mini's reasoning process...")
    print(f"🎯 Understanding why O3 gave different results than GPT-4o")
    print()
    
    choice = input(f"{Fore.YELLOW}What would you like to do?\n1. Analyze specific contracts with O3 reasoning\n2. Compare O3 vs GPT-4o reasoning side-by-side\n3. Both\nChoice (1/2/3): ")
    
    if choice == '1':
        analyze_specific_contracts()
    elif choice == '2':
        compare_reasoning_styles()
    elif choice == '3':
        analyze_specific_contracts()
        print("\n" + "="*80)
        compare_reasoning_styles()
    else:
        print("Invalid choice. Running default analysis...")
        analyze_specific_contracts()

if __name__ == "__main__":
    main() 