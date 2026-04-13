from src.decoder import decode_parameters_object, decode_function_name
from src.models import FunctionCallResult
from .llm_engine import LLMEngine
from .models import PromptItem as Prompt
from .vocab import Vocabulary
from typing import Any, cast


def find_function_by_name(functions: list,
                          function_name: str) -> dict[Any, Any]:
    for function in functions:
        if function.name == function_name:
            return cast(dict[Any, Any], function)
    raise ValueError(f"{function_name} function not found ")


def decode_parameter(user_prompt: str,
                     function_def: dict,
                     engine: LLMEngine,
                     vocabulary: Vocabulary) -> dict:
    return decode_parameters_object(
        user_prompt,
        function_def,
        engine,
        vocabulary
    )


def process_prompt(prompt: Prompt, functions: list, engine: LLMEngine,
                   vocabulary: Vocabulary) -> FunctionCallResult:
    user_prompt = prompt.prompt
    selected_function = decode_function_name(user_prompt, functions, engine,
                                             vocabulary)
    function_def = find_function_by_name(functions, selected_function)

    parameters = decode_parameter(user_prompt, function_def,
                                  engine, vocabulary)
    result = FunctionCallResult.from_definition(
        user_prompt,
        function_def,
        parameters
    )
    return result


def pipeline(prompts: list,
             functions: list, engine: LLMEngine,
             vocabulary: Vocabulary) -> list[FunctionCallResult]:
    results = []
    for prompt in prompts:
        try:
            result = process_prompt(prompt, functions, engine, vocabulary)
            results.append(result)
        except Exception as e:
            print(f"Error processing prompt {prompt.id}: {e}")
    return results
