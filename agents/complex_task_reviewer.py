from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class ComplexTaskReviewer(BaseAgent):
    """
    Reviewer for complex task prompts
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt reviewer specializing in evaluating prompts for complex, multi-step or multi-aspect tasks.
        
        Original user input: {original_input}
        Optimized prompt: {optimized_prompt}
        
        Evaluate the optimized prompt based on these criteria:
        1. Does it break down the complex task into clear components or steps?
        2. Does it provide structured guidance for addressing each component?
        3. Does it maintain balance between all aspects of the original request?
        4. Does it include an appropriate expert persona for addressing the task?
        5. Does it provide clear organizational structure for the response?
        6. Will it likely lead to a comprehensive, well-structured response?
        
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
        Review a complex task prompt
        
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
            "specific_changes": ["Ensure the prompt breaks down the complex task into clear components"]
        })