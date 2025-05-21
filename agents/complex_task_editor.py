from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class ComplexTaskEditor(BaseAgent):
    """
    Editor for complex task prompts - makes minimal changes for improvement
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt editor specializing in complex, multi-step or multi-aspect tasks.
        Your job is to make minimal but effective improvements to a prompt that was not approved.
        
        Original user input: {original_input}
        Current prompt: {current_prompt}
        Review feedback: {feedback}
        Specific change requests: {change_requests}
        
        Your task is to edit the prompt with the SMALLEST POSSIBLE CHANGES needed to address 
        the feedback while maintaining the multi-aspect structure. Unlike a prompt generator, you should 
        NOT rewrite the entire prompt - make targeted edits to fix specific issues.
        
        Focus particularly on:
        1. Improving how the task is broken down into components
        2. Enhancing the structure and organization
        3. Addressing the specific weaknesses identified in the review
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def edit_prompt(self, original_input, current_prompt, feedback, change_requests):
        """
        Make minimal but effective edits to an unapproved prompt
        
        Args:
            original_input (str): The original user input
            current_prompt (str): The current unapproved prompt
            feedback (str): The reviewer's feedback
            change_requests (list): Specific change suggestions
                
        Returns:
            str: The edited prompt
        """
        # Convert change_requests list to string if needed
        if isinstance(change_requests, list):
            change_requests = "\n".join(change_requests)
            
        return self.invoke(
            original_input=original_input,
            current_prompt=current_prompt,
            feedback=feedback,
            change_requests=change_requests
        )