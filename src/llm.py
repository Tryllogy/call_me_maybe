import json
from src.parsing_validator import ParsingValidator
from llm_sdk import Small_LLM_Model
import os
import time
from typing import Any, Optional


class LlmInterpret:
    def __init__(
        self,
        functions: dict,
        input_data: dict,
        name_output_file: str
    ) -> None:
        self.functions: dict = functions
        self.input_data: dict = input_data
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.name_output_file: str = name_output_file
        self.create_output_file()

    @classmethod
    def from_json(
        cls,
        functions_path_str: str,
        input_path_str: str,
        name_output_file: str
    ) -> "LlmInterpret":
        with open(functions_path_str, "r") as f:
            functions: list[dict] = json.loads(f.read())

        with open(input_path_str, "r") as f:
            input_data: list[dict] = json.loads(f.read())

        ParsingValidator(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file
        )

        return cls(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file)

    def create_output_file(self) -> None:
        parsed_file: list[str] = self.name_output_file.split("/")
        path: str = ""
        for i in range(0, len(parsed_file) - 1):
            path += parsed_file[i] + "/"
        os.makedirs(path, exist_ok=True)
        with open(self.name_output_file, "w") as f:
            f.write("[\n")

    def interpret(
            self
    ) -> None:
        base_helper: str = self.make_base_helper()
        tensor: Any = None

        init_time: float = time.time()
        for i, d in enumerate(self.input_data):
            time_input: float = time.time()
            input_helper: str = base_helper
            dump = json.dumps(d["prompt"])
            response: str = "{\n\"prompt\": " + dump + ", \
                    \n  \"name\": \""
            input_helper += f"Prompt:\n{d['prompt']}\nAnswer:\n"
            input_helper += response
            tensor = self.llm.encode(input_helper)
            response += self.generate_response(tensor)
            print(
                f"Total time for prompt {d['prompt']}: "
                f"{time.time() - time_input}")
            with open(self.name_output_file, "a+") as f:
                dict_json: dict = self.make_dict_for_json(
                    response=response
                )
                json.dump(dict_json, f, indent=4)
                if i != len(self.input_data) - 1:
                    f.write(",\n")
                elif i == len(self.input_data) - 1:
                    f.write("\n]")
        print(f"Total time: '{time.time() - init_time}'")

    def make_base_helper(
            self
    ) -> list[int]:
        base_helper: str = json.dumps(self.functions)
        base_helper += "\nReturn exactly one valid JSON object.\n"
        base_helper += "No markdown code fences.\n"
        base_helper += "Do not output any text before or after the JSON object.\n"
        base_helper += "Keys must be exactly: prompt, name, parameters.\n"
        base_helper += "name must be one of the allowed function names.\n"
        base_helper += "parameters must be a JSON object.\n"
        base_helper += "First caracter must be { and last character must be }.\n"
        base_helper += "Example of valid output:\n"
        base_helper += "\"fn_greet\","
        base_helper += "\n  \"parameters\": {\n    \"name\": "
        base_helper += "\"hello\"\n   }\n}\n"
        base_helper += "The functions names are:\n"
        base_helper += self.get_functions_name()
        return base_helper

    def generate_response(self, tensor) -> str:
        response: list[int] = []
        ids: list[int] = list(tensor[0])
        n_brackets: int = 1
        while 1:
            logit: list[float] = self.llm.get_logits_from_input_ids(ids)
            next_token_id: int = logit.index(max(logit))
            ids.append(next_token_id)
            response.append(next_token_id)
            for c in self.llm.decode([next_token_id]):
                if c == "{":
                    n_brackets += 1
                elif c == "}":
                    n_brackets -= 1
            if n_brackets == 0:
                break
        return self.llm.decode(response)

    def make_dict_for_json(
            self,
            response: str,
            prompt: str = "",
    ) -> dict:
        dict_json: dict = {}
        parameters_function: dict = {}
        dict_json.update({"prompt": prompt})
        dict_json.update({"name": ""})
        dict_json.update({"parameters": {}})
        try:
            response_dict: dict = json.loads(response)
            if response_dict.get("prompt"):
                dict_json.update({"prompt": response_dict["prompt"]})
            if response_dict.get("name"):
                dict_json.update({"name": response_dict["name"]})
            if response_dict.get("parameters"):
                parameters_function = response_dict["parameters"]
                dict_json["parameters"] = parameters_function
        except json.JSONDecodeError as e:
            print(e)
            return dict_json
        return dict_json

    def get_functions_name(
            self
    ) -> list:
        names: list[str] = []
        for f in self.functions:
            names.append(f["name"] + "\n")
        return "".join(names)
