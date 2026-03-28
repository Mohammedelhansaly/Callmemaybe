import json


class Vocabulary:
    def __init__(self, id_to_token, token_to_ids):
        self.id_to_token = id_to_token
        self.token_to_ids = token_to_ids

    @classmethod
    def from_json_file(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        id_to_token = {}
        token_to_ids = {}
        for key, value in data.items():
            token_id = int(key)
            token_text = value
            id_to_token[token_id] = token_text
            token_to_ids.setdefault(token_text, []).append(token_id)

        return cls(id_to_token=id_to_token, token_to_ids=token_to_ids)

    def get_token_text(self, token_id):
        return self.id_to_token[token_id]

    def get_token_ids(self, token_text):
        return self.token_to_ids.get(token_text, [])

    def all_tokens(self):
        return list(self.id_to_token.keys())




