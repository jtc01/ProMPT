from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class UnclearTaskReviewer(BaseAgent):
    """
    Reviewer for unclear task prompts
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt reviewer specializing in evaluating prompts for unclear or ambiguous tasks.
        
        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}
        
        Evaluate the optimized prompt based on these criteria:
        1. Does it make a reasonable inference about the user's likely intent?
        2. Does it provide clear direction for the AI to respond helpfully?
        3. Is the inferred intent plausible given the original input?
        4. Does it include appropriate context or structure?
        5. Will it likely lead to a response that would satisfy the user?
        
        Respond in JSON format:
        {{
            "approved": true/false,
            "feedback": "Your detailed feedback if not approved. Empty string if approved.",
            "specific_changes": ["Specific suggestion 1", "Specific suggestion 2"] 
        }}
        
        If not approved, be specific about what changes are needed.
        """)
    
    def review_prompt(self, original_input, optimized_prompt):
        """
        Review an unclear task prompt
        
        Args:
            original_input (str): The original user input
            optimized_prompt (str): The optimized prompt to review
                
        Returns:
            dict: Review result with approval status and feedback
        """
        response = self.invoke(
            original_input=original_input,
            optimized_prompt=optimized_prompt
        )
        
        return self._parse_json_response(response, default={
            "approved": False,
            "feedback": "Failed to parse reviewer response",
            "specific_changes": ["Make a clearer inference about the likely user intent"]
        })