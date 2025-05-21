from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

class DelegatorAgent(BaseAgent):
    """
    Delegator agent that analyzes user input and determines which prompt generator to use
    """
    def __init__(self, model="llama3.2"):
        super().__init__(model)
        
        # Create prompt template for delegator
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are a Delegator Agent responsible for analyzing user input and determining 
        which specialized prompt generator should handle it.
        
        User input: {user_input}
        
        Analyze the input and classify it into one of the following categories:
        1. "simple_task" - Clear, straightforward requests where the task is explicit or reasonably inferred
           Example: "How to make pizza?" or "What is photosynthesis?"
           
        2. "unclear_task" - Inputs where the user's intent is ambiguous or implicit
           Example: Just sharing a poem without context, or "Here's some text I found..."
           
        3. "object_based" - Requests that involve specific objects, links, or data to analyze, which should be preserved in the optimized prompt
           Example: "Compare these two products" or "Analyze this data set"
           
        4. "opinion_based" - Requests asking for subjective judgments or personal views
           Example: "What do you think about modern art?" or "Is AI consciousness possible?" or "Who should I vote for?"
           
        5. "other" - Any inputs that don't clearly fit the above categories
        
        Respond in JSON format as follows:
        {{
            "category": "category_name",
            "reasoning": "Brief explanation of why you chose this category"
        }}
        """)
    
    def analyze_input(self, user_input):
        """
        Analyze user input and determine which path to take in the workflow
        
        Args:
            user_input (str): The raw input from the user
                
        Returns:
            dict: Classification result with category and reasoning
        """
        response = self.invoke(user_input=user_input)
        result = self._parse_json_response(response, default={
            "category": "simple_task",  # Default to simple_task if parsing fails
            "reasoning": "Defaulted due to parsing error"
        })
        
        return result