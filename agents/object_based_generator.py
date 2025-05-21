from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class ObjectBasedPromptGenerator(BaseAgent):
    """
    Prompt generator for tasks involving specific objects, links, or data
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        # Specialized LLMs like Claude or GPT-4V with vision capabilities would excel here
        # for handling objects that might include images or complex data
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer specializing in tasks that involve analyzing specific 
        objects, links, data, or other resources.
        
        Original user input: {user_input}
        
        Your job is to create a prompt that will help an LLM effectively analyze and work with 
        the objects or data in the user's input. Focus on:
        
        1. Identifying the specific objects, links, or data mentions in the input
        2. Clarifying what analysis or comparison the user wants performed
        3. Creating a structured approach to analyzing the objects
        4. Giving the AI a relevant expert persona (e.g., data analyst, product reviewer)
        5. Ensuring the prompt clearly communicates how to handle the objects referenced
        
        For example, if the user asks to compare two products with links, make sure the prompt 
        explains what aspects to compare and how to present the comparison.
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for an object-based task
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced prompt focused on object analysis
        """
        return self.invoke(user_input=user_input)