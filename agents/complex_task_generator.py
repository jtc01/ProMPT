from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class ComplexTaskPromptGenerator(BaseAgent):
    """
    Prompt generator for complex, multi-step or multi-aspect tasks
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        # GPT-4 or Claude would excel at structuring complex tasks due to their 
        # strong planning and decomposition capabilities
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer specializing in complex tasks that involve multiple steps, 
        components, or aspects.
        
        Original user input: {user_input}
        
        Your job is to transform this input into an enhanced prompt that will help an LLM generate 
        a comprehensive, well-structured response. Since this is a complex task with multiple aspects, focus on:
        
        1. Breaking down the task into clear logical components or steps
        2. Providing a structured approach for addressing each component
        3. Adding a helpful expert persona for the AI to adopt
        4. Ensuring all aspects of the request are addressed in a balanced way
        5. Including guidance on how to organize and present the information
        6. Suggesting relevant frameworks or methodologies if applicable
        
        For example, if the user asks for "Explain the environmental impact of plastic pollution and possible solutions",
        your prompt should structure both the explanation of impacts AND the solutions, possibly suggesting 
        categorization by ecosystem, timeframe, or scale.
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for a complex task
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced prompt with structured approach
        """
        return self.invoke(user_input=user_input)