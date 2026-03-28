from pathlib import Path
from .vocab import Vocabulary
import json


class ReadFileError(Exception):
    pass


def load_json_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise ReadFileError(f"File not found: {file_path}")
    if not path.is_file():
        raise ReadFileError(f"Path is not a file: {file_path}")

    try:
        with path.open("r") as f:
            return json.load(f)
    except PermissionError as exc:
        raise ReadFileError(f"Permission denied: {file_path}") from exc
    except OSError as exc:
        raise ReadFileError(f"Error reading file: {file_path}") from exc
    except json.JSONDecodeError as exc:
        raise ReadFileError(f"Invalid JSON format "
                            f"in file: {file_path}") from exc


def save_json_file(data, file_path):
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except PermissionError as exc:
        raise ReadFileError(f"Permision denied: {file_path}") from exc
    except OSError as exc:
        raise ReadFileError(f"Error writing file: {file_path}") from exc
    except TypeError as exc:
        raise ReadFileError(f"Data is not JSON "
                            f"serializable: {file_path}") from exc


def is_valid_prefix(partial_text, allowed_strings):
    return any(choise.startswith(partial_text) for choise in allowed_strings)


def get_valid_next_token(partial_text, allowed_strings, vocabulary: Vocabulary):
    valid_token_ids = []
    for token in vocabulary.all_tokens():
        token_text = vocabulary.get_token_text(token)
        candidate_text = partial_text + token_text

        if is_valid_prefix(candidate_text, allowed_strings):
            valid_token_ids.append(token)
    return valid_token_ids

def is_complete_match(text, allowed_strings):
    return text in allowed_strings