from src.models import validate_function_definitions, validate_prompt_items
import json
from src.utils import load_json_file, save_json_file
from src.vocab import Vocabulary
from src.utils import is_valid_prefix, get_valid_next_token, is_complete_match
from src.llm_engine import LLMEngine
from src.decoder import decode_function_name, decode_string_value,get_function_names
def main():
    # id_to_token={
    #     1: "fn_",
    #     2: "add",
    #     3: "greet",
    #     4: "a",
    #     5: "g",
    #     6: "z",
    # }
    # token_to_ids = {}
    # for token_id, token_text in id_to_token.items():
    #     token_to_ids.setdefault(token_text, []).append(token_id)

    # vocub = Vocabulary(id_to_token=id_to_token, token_to_ids=token_to_ids)
    # allowed_strings = ["fn_add_numbers", "fn_greet"]

    # print(is_valid_prefix("fn_", allowed_strings))
    # print(is_valid_prefix("fn_ad", allowed_strings))
    # print(is_complete_match("fn_add_numbers",allowed_strings))
    # print("-------------------------------------")
    # valid_ids = get_valid_next_token("fn_", allowed_strings, vocub)
    # print(valid_ids)
    # print([vocub.get_token_text(i) for i in valid_ids])

    # engine  = LLMEngine()
    # functions = [
    #         {"name": "fn_add_numbers", "description": "Add two numbers together"},
    #         {"name": "fn_greet", "description": "Greet someone"}
    #     ]
    # promt_text = engine.build_prompt(
    #     "What is 2 + 2?",
    #     functions
    # )
    # promtpt_ids = engine.encode_to_list(promt_text)
    # print(promtpt_ids)
    # try:
        with open("data/input/functions_definition.json", "r") as f:
            function_definitions = json.load(f)

        with open("data/input/function_calling_tests.json", "r") as f:
            prompt_items = json.load(f)    
        
        validate_function = validate_function_definitions(function_definitions)
        # print(validate_function[0].name)
        validate_prompt = validate_prompt_items(prompt_items)
        # print(get_function_names(validate_function))
        engine = LLMEngine()
        vocab_path = engine.model.get_path_to_vocab_file()
        vocabulary = Vocabulary.from_json_file(vocab_path)
        generated_text = decode_string_value(
            user_prompt=validate_prompt[4].prompt,
            function_name="fn_reverse_string",
            parameter="s",
            engine=engine,
            vocabulary=vocabulary
        )
        print(generated_text)
    # except Exception as e:
    #     print(e)
    



if __name__ == "__main__":
    main()
