from src.llm_engine import LLMEngine
from src.utils import get_valid_next_token
from src.vocab import Vocabulary
def get_function_names(functions):
    return [function["name"] for function in functions]


def select_best_valid_token(logits, valid_token_ids):
    best_token_id = valid_token_ids[0]
    best_score = logits[best_token_id]

    for token_id in valid_token_ids[1:]:
        if logits[token_id] > best_score:
            best_score = logits[token_id]
            best_token_id = token_id

    return best_token_id


def decode_function_name(user_prompt, functions, engine, vocabulary) -> str:
    function_names = get_function_names(functions)
    prompt_text = engine.build_prompt(user_prompt, function_names)
    prompt_ids = engine.encode_to_list(prompt_text)

    generated_ids = []
    generated_text = ""
    max_steps = 100

    for _ in range(max_steps):
        full_ids = prompt_ids + generated_ids
        logits = engine.get_next_token_logits(full_ids)
        valid_next_token_ids = get_valid_next_token(generated_text, function_names, vocabulary)
    
    if not valid_next_token_ids:
        raise ValueError("No valid tokens available.")

    next_token_id = select_best_valid_token(
        logits,
        valid_next_token_ids,
    )

    token_text = vocabulary.get_token_text(next_token_id)
    generated_ids.append(next_token_id)
    generated_text += token_text

    if generated_text in function_names:
        return generated_text
    raise ValueError("Could not decode a valid function name.")

