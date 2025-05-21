from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class GenericPromptGenerator(BaseAgent):
    """
    Fallback prompt generator for tasks that don't clearly fit other categories
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer with versatile skills for handling a wide variety of user requests.
        
        Original user input: {user_input}
        
        Your job is to create a general-purpose enhanced prompt that will help an LLM provide 
        a helpful, accurate response. Since this input doesn't clearly fit standard categories, focus on:
        
        1. Identifying the most likely core intent or request
        2. Creating a clear, structured prompt that addresses that intent
        3. Providing sufficient context and guidance for the AI
        4. Adding an appropriate expert persona if relevant
        5. Ensuring the prompt will lead to a helpful response
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for miscellaneous tasks
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced generic prompt
        """
        return self.invoke(user_input=user_input)