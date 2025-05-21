from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json
import time

# Import all agents
from agents import (
    DelegatorAgent,
    # Generators
    SimpleTaskPromptGenerator, 
    UnclearTaskPromptGenerator,
    ObjectBasedPromptGenerator,
    OpinionBasedPromptGenerator,
    GenericPromptGenerator,
    # Reviewers
    SimpleTaskReviewer,
    UnclearTaskReviewer,
    ObjectBasedReviewer,
    OpinionBasedReviewer,
    GenericReviewer,
    # Editors
    SimpleTaskEditor,
    UnclearTaskEditor,
    ObjectBasedEditor,
    OpinionBasedEditor,
    GenericEditor
)

load_dotenv()

class ProMPT:
    """
    ProMPT version 2: An AI agent pipeline that optimizes user prompts with specialized workflows
    """
    def __init__(self, 
                 chatbot_model="llama3.2",
                 max_reviewer_iterations=2):
        
        # Initialize the chatbot LLM
        self.chatbot_llm = OllamaLLM(model=chatbot_model)
        
        # Initialize the delegator
        self.delegator = DelegatorAgent()
        
        # Maximum number of review iterations
        self.max_reviewer_iterations = max_reviewer_iterations
        
        # Initialize workflow components
        
        # Generators
        self.generators = {
            "simple_task": SimpleTaskPromptGenerator(),
            "unclear_task": UnclearTaskPromptGenerator(),
            "object_based": ObjectBasedPromptGenerator(),
            "opinion_based": OpinionBasedPromptGenerator(),
            "other": GenericPromptGenerator()
        }
        
        # Reviewers
        self.reviewers = {
            "simple_task": SimpleTaskReviewer(),
            "unclear_task": UnclearTaskReviewer(),
            "object_based": ObjectBasedReviewer(),
            "opinion_based": OpinionBasedReviewer(),
            "other": GenericReviewer()
        }
        
        # Editors
        self.editors = {
            "simple_task": SimpleTaskEditor(),
            "unclear_task": UnclearTaskEditor(),
            "object_based": ObjectBasedEditor(),
            "opinion_based": OpinionBasedEditor(),
            "other": GenericEditor()
        }
        
    def process_user_input(self, user_input):
        """
        Process the user input through the AI pipeline with delegator, generators, reviewers, and editors
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The final response to give to the user
        """
        print("\n=== WORKFLOW START ===")
        
        # Step 1: Use the delegator to classify the input
        print("Delegator analyzing input...")
        delegation_result = self.delegator.analyze_input(user_input)
        input_category = delegation_result["category"]
        reasoning = delegation_result["reasoning"]
        
        print(f"Input classified as: {input_category}")
        print(f"Reasoning: {reasoning}\n")
        
        # Step 2: Generate an optimized prompt using the appropriate generator
        print(f"Using {input_category} prompt generator...")
        generator = self.generators[input_category]
        optimized_prompt = generator.generate_prompt(user_input)
        
        print(f"Generated prompt: {optimized_prompt}\n")
        
        # Step 3: Review the optimized prompt
        print(f"Using {input_category} reviewer...")
        reviewer = self.reviewers[input_category]
        review_result = reviewer.review_prompt(user_input, optimized_prompt)
        
        prompt_approved = review_result["approved"]
        feedback = review_result.get("feedback", "")
        specific_changes = review_result.get("specific_changes", [])
        
        print(f"Prompt approved: {prompt_approved}")
        
        # Step 4: If not approved, use the editor to improve it
        iterations = 0
        current_prompt = optimized_prompt
        
        while not prompt_approved and iterations < self.max_reviewer_iterations:
            print(f"Prompt not approved. Feedback: {feedback}")
            print(f"Specific changes requested: {specific_changes}")
            
            # Use the editor to make minimal changes
            print(f"Using {input_category} editor...")
            editor = self.editors[input_category]
            edited_prompt = editor.edit_prompt(
                user_input, current_prompt, feedback, specific_changes
            )
            
            print(f"Edited prompt: {edited_prompt}\n")
            
            # Review the edited prompt
            print(f"Reviewing edited prompt...")
            review_result = reviewer.review_prompt(user_input, edited_prompt)
            
            prompt_approved = review_result["approved"]
            feedback = review_result.get("feedback", "")
            specific_changes = review_result.get("specific_changes", [])
            
            print(f"Prompt approved: {prompt_approved}")
            
            # Update for next iteration if needed
            current_prompt = edited_prompt
            iterations += 1
        
        # Step 5: Use the final prompt with the chatbot
        final_prompt = current_prompt
        print(f"\nFinal prompt: {final_prompt}")
        
        # Send to chatbot
        print("\nSending to chatbot...")
        response = self.chatbot_llm.invoke(final_prompt)
        
        print("=== WORKFLOW COMPLETE ===\n")
        return response


def main():
    """
    Main function to run the ProMPT v2 system
    """
    # Create the ProMPT instance
    prompt_system = ProMPT()
    
    print("ProMPT v2 initialized. Enter 'x' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() == 'x':
            print("Exiting ProMPT v2.")
            break
        
        # Process the input and get response
        response = prompt_system.process_user_input(user_input)
        
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()