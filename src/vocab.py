import json


class Vocabulary:
    def __init__(self, id_to_token: dict[int, str],
                 token_to_ids: dict[str, list[int]]) -> None:
        self.id_to_token = id_to_token
        self.token_to_ids = token_to_ids

    @classmethod
    def from_json_file(cls, path: str) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        id_to_token: dict[int, str] = {}
        token_to_ids: dict[str, list[int]] = {}
        for value, key in data.items():
            # print(f"Processing token_id: {key}, token_text: {value}")
            token_id = int(key)
            token_text = value
            id_to_token[token_id] = token_text
            token_to_ids.setdefault(token_text, []).append(token_id)

        return cls(id_to_token=id_to_token, token_to_ids=token_to_ids)

    def get_token_text(self, token_id: int) -> str:
        return self.id_to_token[token_id]

    def get_token_ids(self, token_text: str) -> list[int]:
        return self.token_to_ids.get(token_text, [])

    def all_tokens(self) -> list[int]:
        return list(self.id_to_token.keys())
