from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class OpinionBasedReviewer(BaseAgent):
    """
    Reviewer for opinion-based task prompts
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt reviewer specializing in evaluating prompts for tasks 
        that involve subjective judgments, opinions, or philosophical questions.
        
        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}
        
        Evaluate the optimized prompt based on these criteria:
        1. Does it maintain the core subjective question from the original input?
        2. Does it encourage balanced perspective-taking rather than one-sided views?
        3. Does it avoid introducing bias toward particular opinions?
        4. Does it include an appropriate thoughtful persona for addressing the question?
        5. Will it likely lead to a nuanced, balanced response that considers multiple viewpoints?
        
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
        Review an opinion-based task prompt
        
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
            "specific_changes": ["Ensure the prompt encourages balanced perspective-taking"]
        })