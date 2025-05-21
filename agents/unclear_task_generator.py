from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class UnclearTaskPromptGenerator(BaseAgent):
    """
    Prompt generator for unclear or ambiguous tasks
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        # Mistral or GPT-4 would likely excel at inferring intent from unclear inputs
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer specializing in clarifying ambiguous or unclear user requests.
        
        Original user input: {user_input}
        
        Your job is to analyze this input, infer the most likely intent, and create a clear prompt 
        that will help an LLM generate a helpful response. Since this input is unclear or ambiguous:
        
        1. Carefully analyze what the user is MOST LIKELY trying to achieve
        2. Infer a reasonable intent based on the content and context clues
        3. Create a clear prompt that captures this inferred intent
        4. Include a persona for the AI that would best address this intent
        5. Structure the prompt to elicit a helpful response
        
        For example, if the user just shares a poem without context, they may want analysis, 
        appreciation, or perhaps to understand the poem's meaning and literary devices.
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for an unclear task by inferring intent
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced prompt with inferred intent
        """
        return self.invoke(user_input=user_input)