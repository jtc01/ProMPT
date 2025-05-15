from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

class AIAgent:
    def __init__(self):
        # Keep Ollama for the input manager and reviewer
        self.llm = OllamaLLM(model="llama3.2")
        
        # Use ChatGPT for the output manager
        # You'll need to set your OpenAI API key as an environment variable
        # or pass it directly here
        
        """
        self.output_llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Most commonly used ChatGPT model
            temperature=0.7,
            openai_api_key=os.getenv("OPEN_API_KEY")  # Set this environment variable
        )
        """

        self.output_llm = OllamaLLM(model="llama3.2")


        self.input_manager_prompt = """
        You are an expert in prompt engineering for LLMs. You will be given a prompt for an AI chatbot asked by a human, and your job is to rephrase the prompt so that the AI chatbot will provide better responses.

        Your task is to create an optimized prompt based on this input: {user_input}

        Consider the following when rephrasing the prompt:
        1. Give the chatbot a persona that will help it accomplish the described task
        2. Clearly state the task
        3. Provide as much context as possible
        4. Provide references if possible
        5. Remove unnecessary information or ambiguities
        
        Respond with only the prompt
        """
        
        self.reviewer_prompt = """
        You are a Prompt Reviewer Agent. Your job is to evaluate the quality of an optimized prompt created by an Input Manager Agent.

        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}

        Evaluate the optimized prompt based on the following criteria:
        1. Does it provide clear instructions for the AI?
        2. Does it include relevant context?
        3. Is it free from unnecessary information?
        4. Will it likely lead to a high-quality response?

        Respond in JSON format:
        {{
            "approved": true/false,
            "feedback": "Your detailed feedback if not approved. Empty string if approved."
        }}
        """
    
    def process_user_input(self, user_input, max_iterations=3):
        """
        Process the user input through the AI pipeline with reviewer feedback.
        
        Args:
            user_input (str): The raw input from the user
            max_iterations (int): Maximum number of improvement iterations
                
        Returns:
            str: The final response to give to the user
        """
        # Initialize tracking variables
        iterations = 0
        prompt_approved = False
        
        # Initialize the input manager prompt template
        input_manager_template = ChatPromptTemplate.from_template(self.input_manager_prompt)
        
        # Begin the iteration loop
        while not prompt_approved and iterations < max_iterations:
            # Step 1: Generate an optimized prompt using the Input Manager Agent
            if iterations == 0:
                # First iteration uses the original user input
                input_manager_prompt = input_manager_template.format(user_input=user_input)
            else:
                # Subsequent iterations include reviewer feedback
                input_manager_prompt = input_manager_template.format(
                    user_input=user_input,
                    feedback=reviewer_feedback
                )
            
            # Step 2: Generate the optimized prompt
            optimized_prompt = self.llm.invoke(input_manager_prompt)
            
            # Print the optimized prompt for debug purposes
            print(f"Iteration {iterations+1} prompt: {optimized_prompt}")
            
            # Step 3: Review the optimized prompt
            reviewer_prompt = self.reviewer_prompt.format(
                original_input=user_input,
                optimized_prompt=optimized_prompt
            )
            reviewer_response = self.llm.invoke(reviewer_prompt)
            
            # Parse the reviewer response to determine if the prompt is approved
            # This assumes the reviewer returns a structured response that can be parsed
            review_result = self._parse_reviewer_response(reviewer_response)
            prompt_approved = review_result["approved"]
            reviewer_feedback = review_result.get("feedback", "")
            
            # Print reviewer feedback for debugging
            print(f"Prompt approved: {prompt_approved}")
            if not prompt_approved:
                print(f"Reviewer feedback: {reviewer_feedback}")
            
            iterations += 1
        
        # Step 4: Generate the final response using the approved prompt
        final_response = self.output_llm.invoke(optimized_prompt)
        
        return final_response
    
    def _parse_reviewer_response(self, response):
        """Parse the reviewer's response to extract approval status and feedback."""
        try:
            import json
            return json.loads(response)
        except:
            # Fallback if parsing fails
            if "approved" in response.lower() and "true" in response.lower():
                return {"approved": True, "feedback": ""}
            else:
                return {"approved": False, "feedback": response}

def main():


    agent = AIAgent()
    
    print("AI Agent initialized. Enter 'x' to quit.")
    
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() == 'x':
            print("Exiting AI Agent.")
            break
        
        response = agent.process_user_input(user_input)
        print(f"\nAI: {response}")

if __name__ == "__main__":
    main()