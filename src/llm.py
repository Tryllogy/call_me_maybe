import json
from src.parsing_validator import ParsingValidator
from llm_sdk import Small_LLM_Model
import os


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

        with open(self.llm.get_path_to_vocab_file(), "r") as f:
            self.vocab: dict = json.loads(f.read())

        with open(self.llm.get_path_to_merges_file(), "r") as f:
            self.merge: str = f.read()

        with open(self.llm.get_path_to_tokenizer_file(), "r") as f:
            self.tokenizer: dict = json.loads(f.read())

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
            input_data=input_data
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
        with open(self.name_output_file, "w+") as f:
            f.write("[\n")

    def torch_tensor_to_ids(
        self,
            tensor,
            ids: list[int] = None
    ) -> list[int]:
        if ids is None:
            ids: list = []
        for digit in tensor[0]:
            ids.append(int(digit))
        return ids

    def encode_dict(
        self,
            dict_to_encode: dict,
            ids: list[int] = None
    ) -> list[int]:
        back_n: int = self.llm.encode("\n")
        if ids is None:
            ids: list[int] = []
        for k, v in dict_to_encode.items():
            if isinstance(v, dict):
                ids = self.encode_dict(v, ids)
            else:
                function_value_encode_dict = self.llm.encode(v)
                ids = self.torch_tensor_to_ids(function_value_encode_dict, ids)
                ids = self.torch_tensor_to_ids(back_n, ids)
            function_key_encode_dict = self.llm.encode(k)
            ids = self.torch_tensor_to_ids(function_key_encode_dict, ids)
            ids = self.torch_tensor_to_ids(back_n, ids)
        return ids

    def interpret(
            self
    ) -> None:
        functions_token: list[int] = []
        for f in self.functions:
            functions_token = self.encode_dict(f, functions_token)

        for i, d in enumerate(self.input_data):
            copy_functions_token = functions_token.copy()
            prompt_encode = self.llm.encode(d["prompt"] + "\n")
            self.torch_tensor_to_ids(prompt_encode, copy_functions_token)
            self.add_response_helper(copy_functions_token)
            logit: list[float] = self.llm.get_logits_from_input_ids(
                copy_functions_token)

            response: str = self.make_response(logit, copy_functions_token)

            parameters_ids: list[int] = functions_token.copy()
            self.torch_tensor_to_ids(
                self.llm.encode(d["prompt"] + "\n"),
                parameters_ids
            )
            self.add_parameters_helper(
                parameters_ids, response, d["prompt"])
            logit = self.llm.get_logits_from_input_ids(parameters_ids)
            parameters: list = self.make_parameters(
                logit, parameters_ids)

            with open(self.name_output_file, "a+") as f:
                json.dump(self.make_dict_for_json(
                    d["prompt"], response, parameters), f, indent=4)
                if i < len(self.input_data) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")

        with open(self.name_output_file, "a+") as f:
            f.write("]")

    def make_dict_for_json(
            self,
            prompt: str,
            response: str,
            parameters: list
    ) -> dict:
        dict_json: dict = {}
        parameters_function: dict = {}
        dict_json.update({"prompt": prompt})
        dict_json.update({"name": response})
        dict_json.update({"parameters": {}})
        for f in self.functions:
            if f["name"] == response:
                parameters_function = f["parameters"]
            for i, (k, _) in enumerate(parameters_function.items()):
                try:
                    dict_json["parameters"].update({k: parameters[i]})
                except Exception:
                    continue
        return dict_json

    def add_response_helper(self, ids: list[int]) -> None:
        self.torch_tensor_to_ids(
            self.llm.encode(
                "Answer exactly like: <function_name> without " +
                "any other text\n"
            ),
            ids
        )
        self.torch_tensor_to_ids(self.llm.encode(
            "Allowed names:\n"), ids)
        self.add_functions_name_token(ids)
        self.torch_tensor_to_ids(self.llm.encode(
            "Answer:\n"), ids)

    def add_parameters_helper(
        self,
            ids: list[int],
            response: str,
            prompt: str
    ) -> None:
        parameters_function: dict = {}
        for f in self.functions:
            if f["name"] == response:
                parameters_function = f["parameters"]
        if len(parameters_function) == 0:
            parameters_function.update({"None": {"type": "None"}})
        self.torch_tensor_to_ids(
            self.llm.encode(
                f"\nGive the parameters of function {response}, "
                "values only, in order, comma-separated, never parameter names\n"
            ),
            ids
        )
        self.torch_tensor_to_ids(
            self.llm.encode(
                "Parameters available and types: "
            ),
            ids
        )
        for k, v in parameters_function.items():
            self.torch_tensor_to_ids(
                self.llm.encode(
                    f"{k}: {v['type']}, "
                ),
                ids
            )
        self.torch_tensor_to_ids(
            self.llm.encode(
                f"\nSentence to search in: \n{prompt}"
            ),
            ids
        )
        self.torch_tensor_to_ids(self.llm.encode("\nAnswer:\n"), ids)

    def make_parameters(
            self,
            logit: list[float],
            ids: list[int]
    ) -> list:
        parameters: list = []
        last_token: str = ""
        response: str = ""
        while "\n" not in last_token:
            next_token_id = logit.index(max(logit))
            ids.append(next_token_id)
            logit = self.llm.get_logits_from_input_ids(
                ids)
            last_token = self.llm.decode(ids[-1])
            response += self.llm.decode(ids[-1])
            print(self.llm.decode(ids))

        response = response.replace("\n", "").replace('"', "").strip()
        if response == "":
            return parameters

        if "," in response:
            return [p.strip() for p in response.split(",") if p.strip() != ""]

        parameters.append(response)

        return parameters

    def make_response(
            self,
            logit: list[float],
            ids: list[int]
    ) -> str:
        functions_names: list[str] = []

        for f in self.functions:
            functions_names.append(f["name"])
        last_token: str = ""
        response: str = ""
        while "\n" not in last_token and response not in functions_names:
            next_token_id = logit.index(max(logit))
            ids.append(next_token_id)
            logit = self.llm.get_logits_from_input_ids(
                ids)
            last_token = self.llm.decode(ids[-1])
            response += self.llm.decode(ids[-1])
        response = response.replace("\n", "")
        response = response.replace('"', "")
        return response

    def add_functions_name_token(
            self,
            ids: list[int] = None
    ) -> list[int]:
        if ids is None:
            ids = []
        for f in self.functions:
            name_encode = self.llm.encode(f["name"] + "\n")
            self.torch_tensor_to_ids(name_encode, ids)
