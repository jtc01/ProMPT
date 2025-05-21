from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class SimpleTaskPromptGenerator(BaseAgent):
    """
    Prompt generator for simple, clear tasks
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        # Create prompt template for simple tasks
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer specializing in simple, clear tasks.
        
        Original user input: {user_input}
        
        Your job is to transform this input into an enhanced prompt that will help an LLM generate 
        the best possible response. Since this is a simple task with clear intent, focus on:
        
        1. Maintaining the core question/request intact
        2. Adding a helpful persona for the AI to adopt (e.g., "You are an expert chef" for cooking questions)
        3. Providing sufficient context and structure
        4. Ensuring clarity in what's being asked
        5. Removing any ambiguities or unnecessary information
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for a simple task
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced prompt
        """
        return self.invoke(user_input=user_input)