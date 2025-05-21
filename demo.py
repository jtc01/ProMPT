#!/usr/bin/env python3
"""
ProMPT Usage Examples
This script demonstrates how to use both versions of ProMPT in your applications.
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def demo_v1():
    """
    Demonstrate the usage of ProMPT v1
    """
    from main import AIAgent
    
    print("\n=== ProMPT Version 1 Demo ===\n")
    
    # Initialize the agent
    agent = AIAgent()
    
    # Example user input
    user_input = "How do I make a chocolate cake?"
    print(f"User input: {user_input}")
    
    # Process through ProMPT v1
    response = agent.process_user_input(user_input)
    
    print(f"\nFinal response: {response}")
    print("\n=== End of Demo ===\n")

def demo_v2():
    """
    Demonstrate the usage of ProMPT v2
    """
    from main_v2 import ProMPT
    
    print("\n=== ProMPT Version 2 Demo ===\n")
    
    # Initialize the system
    prompt_system = ProMPT()
    
    # Example inputs for different workflows
    examples = {
        "simple_task": "How do I make a chocolate cake?",
        "complex_task": "Explain the environmental impact of plastic pollution and possible solutions",
        "unclear_task": "Roses are red, violets are blue",
        "object_based": "Compare these two cars: https://example.com/car1 and https://example.com/car2",
        "opinion_based": "Is AI consciousness possible?"
    }
    
    # Choose which example to run
    example_type = "simple_task"  # Change this to test different workflows
    user_input = examples[example_type]
    
    print(f"Example type: {example_type}")
    print(f"User input: {user_input}")
    
    # Process through ProMPT v2
    response = prompt_system.process_user_input(user_input)
    
    print(f"\nFinal response: {response}")
    print("\n=== End of Demo ===\n")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "v1":
        demo_v1()
    else:
        demo_v2()  # Default to v2
