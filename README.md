*This project has been created as part of the 42 curriculum by moel-han*

<h1 align="center">
  Call Me Maybe
<h1>

## Description
Call Me Maybe is a function-calling system that transforms natural-language prompts into structured function calls. Instead of answering user requests directly, the project identifies the most relevant function and extracts its parameters with the correct types, then outputs a valid JSON object containing the prompt, the function name, and the parameters. The project is built around a small language model and uses constrained decoding to ensure reliable, machine-readable output. Its main goal is to demonstrate that even a lightweight model can perform robust function selection and parameter extraction when generation is guided token by token by structural rules


## Instructions

### Algorithm

Constrained decoding is a technique that manipulates a Large Language Model’s (LLM) token generation process, restricting outputs to follow specific rules, grammars (like JSON or SQL), or formats. By filtering out invalid tokens at each step, it guarantees compliant, structured output, improves reliability, and increases generation speed for production applications. 

# Compilation

```bash
make install
```
This will install all dependencies to prepare the envirement to run the project

Before running the project, please ensure that the virtual environment is created and all dependencies are installed and activated.

```bash
make run
```
or 
```bash
uv run -m src --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

#### input
Two json files: the first contains the definition of the functions and the second the tests or prompts for calling the functions.

#### output
json file containing the program's output in the form user_prompt, function name, parameters

### Exemple Output

```bash
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
      "a": 2.0,
      "b": 3.0
  }
}
```

## Resources
-[constrained decoding](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Constrained_Decoding/README.html)
 