from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json

class BaseAgent:
    """
    Base class for all ProMPT agents
    """
    def __init__(self, model="llama3.2"):
        self.llm = OllamaLLM(model=model)
        self.prompt_template = None
        
    def invoke(self, **kwargs):
        """
        Process inputs using the agent's prompt template and LLM
        
        Args:
            **kwargs: Arguments to feed into the prompt template
                
        Returns:
            str: LLM response
        """
        if not self.prompt_template:
            raise NotImplementedError("Each agent must define a prompt template")
            
        formatted_prompt = self.prompt_template.format(**kwargs)
        return self.llm.invoke(formatted_prompt)
    
    def _parse_json_response(self, response, default=None):
        """Parse a JSON response from the LLM, with fallback handling"""
        try:
            # Find JSON content inside the response if needed
            # Some LLMs might wrap JSON in ```json ``` or other markers
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            return json.loads(response)
        except:
            return default if default is not None else {"error": "Failed to parse JSON response"}