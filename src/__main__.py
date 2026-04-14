from src.models import validate_function_definitions, validate_prompt_items
from src.utils import load_json_file, save_json_file
from src.vocab import Vocabulary
from src.utils import ReadFileError
from src.llm_engine import LLMEngine
from src.pipeline import pipeline
import argparse
import sys
import time

DEFAULT_FUNCTION_DEFINITIONS_PATH = "data/input/functions_definition.json"
DEFAULT_INPUT_PATH = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_PATH = "data/output/function_calling_results.json"


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process function "
                                     "definitions and prompt items.")
    parser.add_argument("--functions_definition",
                        help="Path to the function definitions JSON file.")
    parser.add_argument("--input",
                        help="path to the input prompts JSON file.")
    parser.add_argument("--output",
                        help="path to the output JSON file.")
    return parser


def main() -> None:
    parser = parse_arguments()
    args = parser.parse_args()

    try:
        start = time.time()
        function_definitions = load_json_file(args.functions_definition)
        prompt_items = load_json_file(args.input)
        validate_function = validate_function_definitions(function_definitions)
        validate_prompt = validate_prompt_items(prompt_items)
        # print(get_function_names(validate_function))
        engine = LLMEngine()
        vocab_path = engine.model.get_path_to_vocab_file()
        vocabulary = Vocabulary.from_json_file(vocab_path)
        results = pipeline(validate_prompt,
                           validate_function,
                           engine,
                           vocabulary)
        output_data = [result.model_dump() for result in results]
        save_json_file(output_data, args.output)
        end = time.time()
        print(end - start)
    except ReadFileError as e:
        print(f"File error {e} ", file=sys.stderr)
    except ValueError as e:
        print(f"validation error {e} ", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
