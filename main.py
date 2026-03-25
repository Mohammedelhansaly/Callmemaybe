from src.models import validate_function_definitions, validate_prompt_items
import json


def main():
    try:
        with open("data/input/functions_definition.json", "r") as f:
            function_definition = json.load(f)
        
        with open("data/input/function_calling_tests.json", "r") as f:
            prompt_items = json.load(f)
        
        finctions = validate_function_definitions(function_definition)
        prompts = validate_prompt_items(prompt_items)

        print(finctions[0].name)
        print(prompts[0].prompt)
    except Exception as e:
        print(e)
    



if __name__ == "__main__":
    main()
