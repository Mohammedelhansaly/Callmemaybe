from src.utils import get_valid_next_token, get_valid_string_token_ids, get_valid_number_token_ids, get_valid_boolean_token_ids
import json


def get_function_names(functions):
    return [function.name for function in functions]


def get_function_desc(functions):
    return [function.description for function in functions]


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
    function_desc = get_function_desc(functions)
    prompt_text = engine.build_prompt(user_prompt, function_names,
                                      function_desc)
    prompt_ids = engine.encode_to_list(prompt_text)

    generated_ids = []
    generated_text = ""
    max_steps = 100

    for _ in range(max_steps):
        full_ids = prompt_ids + generated_ids
        logits = engine.get_next_token_logits(full_ids)

        valid_next_token_ids = get_valid_next_token(
            generated_text,
            function_names,
            vocabulary,
        )

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


def decode_string_value(user_prompt, function_name,
                        parameter, engine, vocabulary) -> str:
    prompt_text = engine.build_prompt_parameters(
        user_prompt,
        function_name,
        parameter,
        "string",
    )
    prompt_ids = engine.encode_to_list(prompt_text)

    generated_ids = []
    generated_text = ""
    max_steps = 100

    for _ in range(max_steps):
        full_ids = prompt_ids + generated_ids
        logits = engine.get_next_token_logits(full_ids)

        valid_next_token_ids = get_valid_string_token_ids(
            generated_text,
            vocabulary,
        )

        if not valid_next_token_ids:
            raise ValueError("No valid tokens available.")

        next_token_id = select_best_valid_token(
            logits,
            valid_next_token_ids,
        )

        # token_text = vocabulary.get_token_text(next_token_id)
        generated_ids.append(next_token_id)
        generated_text = engine.decode_ids(generated_ids)

        # print("generated_text:", repr(generated_text))
        # print(
        #     "valid tokens:",
        #     [repr(vocabulary.get_token_text(t)) for t in valid_next_token_ids[:20]],
        # )
        # print("chosen token:", repr(token_text))

        try:
            value = json.loads(generated_text)
            if isinstance(value, str):
                return value
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not decode a valid string value.")


def decode_number_value(user_prompt, function_name, parameter, engine, vocabulary):
    
    promtpt_text = engine.build_number_prompt_parameters(
        user_prompt,
        function_name,
        parameter,
        "number"
    )
    prompt_ids = engine.encode_to_list(promtpt_text)
    generated_ids = []
    max_steps = 100
    generated_text = ""
    for _ in range(max_steps):
        full_ids = prompt_ids + generated_ids
        logits = engine.get_next_token_logits(full_ids)

        valid_next_token_ids = get_valid_number_token_ids(
            generated_text,
            vocabulary
        )
        if not valid_next_token_ids:
            raise ValueError("No valid tokens available.")
        
        next_token_id = select_best_valid_token(
            logits,
            valid_next_token_ids
        )
        generated_ids.append(next_token_id)
        generated_text = engine.decode_ids(generated_ids)

        try:
            value = float(generated_text)
            return value
        except ValueError:
            return False

    raise ValueError("Could not decode a valid number value.")

def decode_boolean_value(
    user_prompt,
    function_name,
    parameter,
    engine,
    vocabulary,
):
    prompt_text = engine.build_boolean_prompt_parameters(
        user_prompt,
        function_name,
        parameter,
        "boolean",
    )
    prompt_ids = engine.encode_to_list(prompt_text)

    max_steps = 20
    generated_ids = []

    for _ in range(max_steps):
        generated_text = engine.decode_ids(generated_ids).strip()

        full_ids = prompt_ids + generated_ids
        logits = engine.get_next_token_logits(full_ids)

        valid_next_token_ids = get_valid_boolean_token_ids(
            generated_ids,
            engine,
            vocabulary,
        )

        if not valid_next_token_ids:
            raise ValueError("No valid tokens available.")

        next_token_id = select_best_valid_token(
            logits,
            valid_next_token_ids,
        )

        generated_ids.append(next_token_id)
        generated_text = engine.decode_ids(generated_ids).strip()

        if generated_text == "true":
            return True
        if generated_text == "false":
            return False

    raise ValueError("Could not decode a valid boolean value.")

