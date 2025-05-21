from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class SimpleTaskReviewer(BaseAgent):
    """
    Reviewer for simple task prompts
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt reviewer specializing in evaluating prompts for simple, clear tasks.
        
        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}
        
        Evaluate the optimized prompt based on these criteria:
        1. Does it maintain the core intent of the original question?
        2. Does it provide clear instructions for the AI?
        3. Does it include an appropriate expert persona for the AI?
        4. Is it free from unnecessary information?
        5. Will it likely lead to a high-quality, focused response?
        
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
        Review a simple task prompt
        
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
            "specific_changes": ["Ensure the prompt maintains the original question's intent"]
        })