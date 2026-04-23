*This project has been created as part of the 42 curriculum by moel-han*

<h1 align="center">
  Call Me Maybe
<h1>

## Description
Call Me Maybe is a function-calling system that transforms natural-language prompts into structured function calls. Instead of answering user requests directly, the project identifies the most relevant function and extracts its parameters with the correct types, then outputs a valid JSON object containing the prompt, the function name, and the parameters. The project is built around a small language model and uses constrained decoding to ensure reliable, machine-readable output. Its main goal is to demonstrate that even a lightweight model can perform robust function selection and parameter extraction when generation is guided token by token by structural rules


## Instructions

### Algorithm

Constrained decoding is a technique that manipulates a Large Language Model’s (LLM) token generation process, restricting outputs to follow specific rules, grammars (like JSON or SQL), or formats. By filtering out invalid tokens at each step, it guarantees compliant, structured output, improves reliability, and increases generation speed for production applications.

#### Step1 : Build prompt

generation prompt like:

```bash
You are a function selection assistant.
Choose exactly one function name from the available functions.

User request:
What is the sum of 2 and 3?

Available functions:
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- fn_reverse_string: Reverse a string and return the reversed result.

Return only the function name.
```
#### Step2 : Encode the prompt

convert the prompt to input ids/tokens

```bash
prompt_ids = [101, 55, 23, 900, ...]
```

#### Step3 : Start generation

start with empty output

```bash
generated_text = ""
generated_ids = []
```

#### Step4 : Model logits

LLM calculates all scores for the next tokens

```bash
logits[token_1] = 0.3
logits[token_2] = 2.7
logits[token_3] = 1.1
...
```

#### Step5 : Apply constaints

allowd names

```bash
["fn_add_numbers", "fn_greet", "fn_reverse_string"]
```

From `generated_text="`, the only tokens that can start with one of these names are valid.

For example, valid tokens could be:

"f"
"fn"
"fn_"

And invalid tokens like:

"The"
"2"
"hello"

These invalid tokens are discarded.

#### Step6 : Choose best valid token

from the valid tokens , we choose who has the highest logits

exmple 
- token "fn" score = 3.2
- token "f" score = 1.4

```bash
generated_text="fn"
```

#### Step7 : Repeat

```bash
generated_text="fn_"
generated_text="fn_add"
...
```

#### Step9 : Continue until complete match

```bash
generated_text="fn_add_numbers"
```
now the this text exactly become  one of the allowed names

#### Step10 : stop

result

```bash
fn_add_numbers
```



## Compilation

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

By default, the program reads
- data/input/function_definitions.json
- data/input/function_calling_tests.json

#### output
json file containing the program's output in the form user_prompt, function name, parameters

and writes the results to:
- data/output/function_calling_results.json
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
- [constrained decoding](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Feature_Guide/Constrained_Decoding/README.html)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [uv](https://docs.astral.sh/uv/)

### AI Assistances

During the development of this function calling system project, IA Assistance helped me understand the concept of function calling, constrained decoding, and token-by-token generation. It was also useful for brainstorming the project structure, clarifying the role of each module, and improving the overall organization of the code.
 