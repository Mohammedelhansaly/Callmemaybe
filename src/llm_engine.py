
from llm_sdk import Small_LLM_Model


class LLMEngine:
    def __init__(self):
        self.model = Small_LLM_Model()

    def build_prompt(self, prompt, functions_names, functions_desc):
        lines = []
        for name, desc in zip(functions_names, functions_desc):
            lines.append(f"- {name} : {desc}")
        joined_functions = "\n".join(lines)
        return (
            "You are a function selection assistant.\n"
            "Choose exactly one function name from the "
            "following list of functions\n"
            f"User prompt: \n{prompt}\n"
            f"Available functions: \n{joined_functions}\n"
            "return only the name of the function you choose\n"
        )

    def build_prompt_parameters(self, prompt, function_name,
                                parameter_name, parameter_type):
        return (
            "You are extracting one function parameter.\n\n"
            f"User request:\n{prompt}\n\n"
            f"Selected function:\n{function_name}\n\n"
            f"Parameter name:\n{parameter_name}\n\n"
            f"Parameter type:\n{parameter_type}\n\n"
            "Return only a valid JSON value for this parameter.\n"
        )

    def encode_to_list(self, text):
        tensor_ids = self.model.encode(text)
        return tensor_ids[0].tolist()

    def get_next_token_logits(self, token_ids):
        return self.model.get_logits_from_input_ids(token_ids)
