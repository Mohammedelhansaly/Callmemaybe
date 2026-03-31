from src.models import validate_function_definitions, validate_prompt_items
import json
from src.utils import load_json_file, save_json_file
from src.vocab import Vocabulary
from src.utils import is_valid_prefix, get_valid_next_token, is_complete_match
from src.llm_engine import LLMEngine
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

    engine  = LLMEngine()
    promt_text = engine.build_prompot(
        "What is 2 + 2?",
        [
            {"name": "fn_add_numbers", "description": "Add two numbers together"},
            {"name": "fn_greet", "description": "Greet someone"}
        ]
    )
    promtpt_ids = engine.encode_to_list(promt_text)
    print(promt_text)

    



if __name__ == "__main__":
    main()
