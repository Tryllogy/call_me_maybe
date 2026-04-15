import json
from .parsing_validator import ParsingValidator
from .output_validator import OutputValidator
from .llm_sdk import Small_LLM_Model
import os
import time
from typing import Any


class LlmInterpret:
    def __init__(
        self,
        functions: list[dict[str, Any]],
        input_data: list[dict[str, Any]],
        name_output_file: str
    ) -> None:

        ParsingValidator(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file
        )

        self.functions: list[dict[str, Any]] = functions
        self.input_data: list[dict[str, Any]] = input_data
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.name_output_file: str = name_output_file
        self.create_output_file()
        self.valid_names: list[str] = [function["name"]
                                       for function in functions]

    @classmethod
    def from_json(
        cls,
        functions_path_str: str,
        input_path_str: str,
        name_output_file: str
    ) -> "LlmInterpret":

        if not functions_path_str.endswith(".json") or \
                not input_path_str.endswith(".json"):
            raise ValueError(
                "The paths for functions and input data must end with '.json'"
            )

        with open(functions_path_str, "r") as f:
            functions: list[dict[str, Any]] = json.loads(f.read())

        with open(input_path_str, "r") as f:
            input_data: list[dict[str, Any]] = json.loads(f.read())

        return cls(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file)

    def create_output_file(self) -> None:
        path: str = os.path.dirname(self.name_output_file)
        if path:
            os.makedirs(path, exist_ok=True)
        with open(self.name_output_file, "w") as f:
            f.write("[]\n")

    def interpret(
            self
    ) -> None:
        base_helper: str = self.make_base_helper()
        tensor: Any = None
        outputs: list[dict[str, Any]] = []

        init_time: float = time.time()
        for d in self.input_data:
            time_input: float = time.time()
            prompt: str = d["prompt"]
            response_prefix: str = '{"name":"'
            input_helper: str = (
                f"{base_helper}\n"
                f"User request:\n{prompt}\n"
                f"JSON:\n{response_prefix}"
            )
            tensor = self.llm.encode(input_helper)
            response: str = response_prefix + self.generate_response(tensor)
            print(
                f"Total time for prompt '{prompt}': "
                f"{round(time.time() - time_input, 2)} seconds")
            dict_json: dict[str, Any] = self.make_dict_for_json(
                response=response,
                prompt=prompt,
            )
            try:
                OutputValidator(
                    prompt=prompt,
                    name=dict_json["name"],
                    parameters=dict_json["parameters"],
                    valid_names=self.valid_names
                )
            except Exception:
                outputs.append(
                    {
                        "prompt": prompt,
                        "name": "",
                        "parameters": {}
                    }
                )
            else:
                outputs.append(dict_json)
            print(
                f"Output:\n{json.dumps(outputs[-1], indent=4)}\n")

        with open(self.name_output_file, "w") as f:
            json.dump(outputs, f, indent=4)
            f.write("\n")
        print(f"Total time: '{round(time.time() - init_time, 2)}' seconds")

    def make_base_helper(
            self
    ) -> str:
        functions: list[dict[str, Any]] = []
        for function in self.functions:
            functions.append(
                {
                    "name": function["name"],
                    "parameters": function["parameters"],
                }
            )

        base_helper: str = (
            "Choose exactly one function matching the user request.\n"
        )
        base_helper += (
            "Return only one JSON object with keys name and parameters.\n"
        )
        base_helper += "No markdown and no extra text.\n"
        base_helper += "name must be one of the function names below.\n"
        base_helper += (
            "parameters must be a JSON object with matching parameter "
            "names.\n"
        )
        base_helper += "Start with { and end with }.\n"
        base_helper += "Allowed functions:\n"
        base_helper += json.dumps(functions, separators=(",", ":"))
        return base_helper

    def generate_response(
        self,
        tensor: Any
    ) -> str:
        response: list[int] = []
        ids: list[int] = list(tensor[0])
        n_brackets: int = 1
        while True:
            logit: list[float] = self.llm.get_logits_from_input_ids(ids)
            next_token_id: int = logit.index(max(logit))

            ids.append(next_token_id)
            response.append(next_token_id)

            decoded_token: str = self.llm.decode([next_token_id])
            n_brackets += decoded_token.count("{")
            n_brackets -= decoded_token.count("}")

            if n_brackets == 0:
                break
        return self.llm.decode(response)

    def make_dict_for_json(
            self,
            response: str,
            prompt: str = "",
    ) -> dict[str, Any]:
        dict_json: dict[str, Any] = {}
        parameters_function: dict[str, Any] = {}
        dict_json.update({"prompt": prompt})
        dict_json.update({"name": ""})
        dict_json.update({"parameters": {}})
        try:
            response_dict: dict[str, Any] = json.loads(response)
            if response_dict.get("prompt"):
                dict_json.update({"prompt": response_dict["prompt"]})
            if response_dict.get("name"):
                dict_json.update({"name": response_dict["name"]})
            if response_dict.get("parameters"):
                parameters_function = response_dict["parameters"]
                dict_json["parameters"] = parameters_function
        except json.JSONDecodeError:
            return dict_json
        return dict_json
