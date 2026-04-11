from src.decoder import decode_parameters_object, decode_function_name
from src.models import FunctionCallResult
def find_function_by_name(functions, function_name):
    for function in functions:
        if function.name == function_name:
            return function
    raise ValueError(f"{function_name} function not found ")


def decode_parameter(user_prompt, function_def, engine, vocabulary):
    return decode_parameters_object(
        user_prompt,
        function_def,
        engine,
        vocabulary
    )


def process_prompt(prompt, functions, engine, vocabuary):
    user_prompt = prompt.prompt
    selected_function = decode_function_name(user_prompt, functions, engine, vocabuary)
    function_def = find_function_by_name(functions, selected_function)

    parameters = decode_parameter(user_prompt, function_def, engine, vocabuary)
    result = FunctionCallResult.from_definition(
        user_prompt,
        function_def,
        parameters
    )
    return result

def pipeline(prompts, functions, engine, vocabulary):
    results = []
    for prompt in prompts:
        try:
            result = process_prompt(prompt, functions, engine, vocabulary)
            results.append(result)
        except Exception as e:
            print(f"Error processing prompt {prompt.id}: {e}")
    return results