# ProMPT Version 2

An advanced AI agent pipeline that optimizes user prompts before sending them to language models, using specialized workflows for different types of requests.

## Overview

ProMPT v2 uses a multi-agent approach with specialized workflows:

1. **Delegator** - Analyzes user input and routes it to the appropriate workflow
2. **Specialized Prompt Generators** - Optimize different types of user requests:
   - Simple Task Generator - For clear, straightforward requests
   - Unclear Task Generator - For ambiguous inputs requiring intent inference
   - Object-Based Generator - For requests involving specific objects or data analysis
   - Opinion-Based Generator - For subjective questions requiring balanced perspective
   - Generic Generator - Fallback for other types of requests

3. **Specialized Reviewers** - Evaluate optimized prompts with criteria specific to each task type

4. **Specialized Editors** - Unlike v1 which sent prompts back to generators, v2 uses targeted editors that make minimal necessary changes to address reviewer feedback

5. **Chatbot** - Processes the final approved prompt and delivers the response to the user

## Workflow

1. The Delegator analyzes the user's question and decides which specialized workflow to use
2. The appropriate prompt generator creates an enhanced prompt
3. A specialized reviewer evaluates the prompt
4. If approved, the prompt goes directly to the chatbot
5. If not approved, an editor makes minimal necessary changes to address the reviewer feedback
6. The edited prompt is reviewed again, and the cycle continues until approved or max iterations reached
7. The final approved prompt is sent to the chatbot for processing
8. The response is returned to the user

## Key Improvements from v1

- **Specialized Workflows** - Different prompt generators and reviewers tailored to specific types of requests
- **Minimal Editing** - Editors make surgical changes rather than complete rewrites when revisions are needed
- **Task-Specific Criteria** - Each reviewer uses evaluation criteria tailored to the task type
- **Opinion-Based Handling** - Special workflow for subjective questions requiring balanced perspectives

## Requirements

- Python 3.8+
- Ollama (for running LLaMA 3.2 locally)
- OpenAI API key (optional, for using ChatGPT models)

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ProMPT.git
cd ProMPT
```

2. Install required dependencies:
```
pip install langchain-ollama langchain-openai langchain-core python-dotenv
```

3. Install Ollama following the instructions at ollama.ai and download the LLaMA 3.2 model:
```
ollama pull llama3.2
```

4. Create a .env file in the project root (required only if using OpenAI models):
```
OPEN_API_KEY=your_openai_api_key_here
```

## Usage

Run the version 2 application:
```
python main_v2.py
```

The program will prompt you for input. Type your question or prompt, and the ProMPT system will:

1. Analyze your input and choose the appropriate workflow
2. Optimize your prompt
3. Review and possibly refine it based on quality criteria
4. Process the optimized prompt through the selected LLM
5. Display the response

Type 'x' to exit the program.

## LLM Recommendations

While the default implementation uses LLaMA 3.2, these specialized tasks might benefit from different models:

- **Unclear Task Generator**: Mistral or GPT-4 might excel at inferring intent from ambiguous inputs
- **Object-Based Tasks**: Claude or GPT-4V with vision capabilities would be ideal for handling objects that might include images or complex data
- **Opinion-Based Tasks**: Claude or Anthropic models might excel due to their nuanced handling of subjective topics and balanced perspective-taking
