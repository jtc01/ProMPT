from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class OpinionBasedPromptGenerator(BaseAgent):
    """
    Prompt generator for tasks involving subjective judgments or opinions
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        # Claude or Anthropic models might excel here due to their nuanced handling
        # of subjective topics and balanced perspective-taking
        
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert prompt engineer specializing in tasks that involve subjective judgments, 
        opinions, or philosophical questions.
        
        Original user input: {user_input}
        
        Your job is to create a prompt that will help an LLM provide a balanced, thoughtful, 
        and nuanced response to questions seeking opinions or value judgments. Focus on:
        
        1. Identifying the core subjective question or opinion being sought
        2. Creating a prompt that encourages balanced perspective-taking
        3. Ensuring multiple viewpoints will be considered in the response
        4. Giving the AI a thoughtful, philosophical persona where appropriate
        5. Structuring the prompt to avoid biased or one-sided responses
        
        For example, if a user asks "Is AI consciousness possible?", ensure the prompt 
        encourages exploring various philosophical and scientific perspectives on the topic.
        
        Generate ONLY the improved prompt, nothing else.
        """)
    
    def generate_prompt(self, user_input):
        """
        Generate an enhanced prompt for an opinion-based task
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            str: The enhanced prompt focused on balanced perspective-taking
        """
        return self.invoke(user_input=user_input)