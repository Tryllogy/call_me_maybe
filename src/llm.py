import json
from src.parsing_validator import ParsingValidator
from llm_sdk import Small_LLM_Model


class Llm_interpret:
    def __init__(self, functions: dict, input_data: dict):
        self.functions: dict = functions
        self.input_data: dict = input_data
        self.llm: Small_LLM_Model = Small_LLM_Model()

        with open(self.llm.get_path_to_vocab_file(), "r") as f:
            self.vocab: dict = json.loads(f.read())

        with open(self.llm.get_path_to_merges_file(), "r") as f:
            self.merge: str = f.read()

        with open(self.llm.get_path_to_tokenizer_file(), "r") as f:
            self.tokenizer: dict = json.loads(f.read())

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

    def interpret(self):
        for d in self.input_data:
            ids: list = []
            torch_test = self.llm.encode(d["prompt"])
            for digit in torch_test[0]:
                ids.append(int(digit))

            logic = self.llm.get_logits_from_input_ids(ids)
            decode_test = self.llm.decode(torch_test[0])
            print("LOGIC = ", max(logic))
            print("DECODE = ", decode_test)
            break
