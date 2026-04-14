*This project has been created as part of the 42 curriculum by moel-han*

<h1 align="center">
  Call Me Maybe
<h1>

## Description
Call Me Maybe is a function-calling system that transforms natural-language prompts into structured function calls. Instead of answering user requests directly, the project identifies the most relevant function and extracts its parameters with the correct types, then outputs a valid JSON object containing the prompt, the function name, and the parameters. The project is built around a small language model and uses constrained decoding to ensure reliable, machine-readable output. Its main goal is to demonstrate that even a lightweight model can perform robust function selection and parameter extraction when generation is guided token by token by structural rules