from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class ObjectBasedReviewer(BaseAgent):
    """
    Reviewer for object-based task prompts
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt reviewer specializing in evaluating prompts for tasks that involve 
        specific objects, links, data, or other resources.
        
        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}
        
        Evaluate the optimized prompt based on these criteria:
        1. Does it properly identify the objects or data mentioned in the original input?
        2. Does it provide clear instructions on how to analyze or work with these objects?
        3. Does it maintain the user's intended comparison or analysis goal?
        4. Does it include an appropriate expert persona for handling such objects?
        5. Will it likely lead to a useful analysis or comparison of the referenced objects?
        
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
        Review an object-based task prompt
        
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
            "specific_changes": ["Ensure the prompt properly addresses the objects mentioned in the user input"]
        })