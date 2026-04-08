import json
from .parsing_validator import ParsingValidator


class Parsing:
    def __init__(self, functions: dict, input_data: dict):
        self.functions: dict = functions
        self.input_data: dict = input_data

    @classmethod
    def from_json(cls, functions_path_str: str, input_path_str: str):
        with open(functions_path_str, "r") as f:
            functions: list[dict] = json.loads(f.read())
        with open(input_path_str, "r") as f:
            input_data: list[dict] = json.loads(f.read())
        ParsingValidator(
            functions=functions,
            input_data=input_data
        )
        return cls(functions=functions, input_data=input_data)
