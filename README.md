# ProMPT
An AI agent pipeline that optimizes user prompts before sending them to language models. The system uses a multi-agent approach to enhance prompt quality through iterative refinement.
Show Image
Overview
Smart ProMPT consists of three main components:

Input Manager - Optimizes user queries based on prompt engineering best practices
Reviewer - Evaluates optimized prompts and provides feedback
Output Manager - Processes the final approved prompt through an LLM

The system aims to improve AI responses by automatically enhancing the quality of user prompts before they reach the language model.
Requirements

Python 3.8+
Ollama (for running LLaMA 3.2 locally)
OpenAI API key (optional, for using ChatGPT models)

Installation

Clone this repository:

git clone https://github.com/yourusername/smart-prompt.git
cd smart-prompt

Install required dependencies:

pip install langchain-ollama langchain-openai langchain-core python-dotenv

Install Ollama following the instructions at ollama.ai and download the LLaMA 3.2 model:

ollama pull llama3.2

Create a .env file in the project root (required only if using OpenAI models):

OPEN_API_KEY=your_openai_api_key_here
Configuration
The default configuration uses Ollama's LLaMA 3.2 model for all components. To use OpenAI for the output manager:

Uncomment the OpenAI section in the __init__ method
Comment out the Ollama output manager line
Make sure your OpenAI API key is in the .env file

Usage
Run the application:
python main.py
The program will prompt you for input. Type your question or prompt, and the AI agent will:

Optimize your prompt
Review and possibly refine it based on quality criteria
Process the optimized prompt through the selected LLM
Display the response

Type 'x' to exit the program.
Example
AI Agent initialized. Enter 'x' to quit.

User: How do I make a chocolate cake?

Iteration 1 prompt: You are a professional baker with decades of experience in pastry making. Your task is to provide a detailed and easy-to-follow recipe for making a delicious chocolate cake.

Please include:
1. A complete list of ingredients with precise measurements
2. Step-by-step instructions for preparation and baking
3. Tips for common mistakes to avoid
4. Suggestions for frosting and decoration
5. Information about baking time and temperature

Reference popular baking techniques and explain why certain steps are important for achieving the perfect texture and flavor.

Prompt approved: True

AI: [Detailed chocolate cake recipe response]
Customization
You can modify the agent's behavior by editing the prompt templates in the AIAgent class:

input_manager_prompt - Controls how user input is optimized
reviewer_prompt - Sets the criteria for prompt evaluation

You can also adjust the maximum number of refinement iterations by changing the max_iterations parameter in the process_user_input method.
