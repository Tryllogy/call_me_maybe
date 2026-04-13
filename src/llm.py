import json
from src.parsing_validator import ParsingValidator
from llm_sdk import Small_LLM_Model
import os
import time


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
        with open(self.name_output_file, "w") as f:
            f.write("")

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

    def interpret(
            self
    ) -> None:
        first_time = time.time()
        functions_token: list[int] = self.make_base_helper()
        responses: list[str] = []

        for i, d in enumerate(self.input_data):
            loop_time = time.time()
            copy_functions_token = functions_token.copy()

            self.torch_tensor_to_ids(self.llm.encode(
                f"Prompt:\n{d['prompt']}\nAnswer:\n"), copy_functions_token)
            self.torch_tensor_to_ids(self.llm.encode("{\n\"prompt\": \"" + d['prompt'] + "\",\n  \"name\": \""),
                                     copy_functions_token)
            response = self.generate_response(
                copy_functions_token,
                self.llm.get_logits_from_input_ids(copy_functions_token),
                d['prompt'])
            responses.append(response)
            print("Time for prompt n°", i, ":",
                  time.time() - loop_time, "seconds")
        print("Total time:", time.time() - first_time, "seconds")
        with open(self.name_output_file, "a+") as f:
            f.write("[\n")
            for i, response in enumerate(responses):
                dict_json: dict = self.make_dict_for_json(
                    response=response
                )
                json.dump(dict_json, f, indent=4)
                if i != len(self.input_data) - 1:
                    f.write(",\n")
            f.write("\n]")

    def make_base_helper(
            self
    ) -> list[int]:
        functions_token: list[int] = []
        functions_token = self.torch_tensor_to_ids(self.llm.encode(
            json.dumps(self.functions)), functions_token)
        self.torch_tensor_to_ids(self.llm.encode(
            "Return exactly one valid JSON object.\n"
            "No markdown code fences.\n"
            "Do not output any text before or after the JSON object.\n"
            "Keys must be exactly: prompt, name, parameters.\n"
            "name must be one of the allowed function names.\n"
            "parameters must be a JSON object.\n"
            "First caracter must be { and last character must be }.\n"
            "Example of valid output:\n"),
            functions_token)
        self.torch_tensor_to_ids(self.llm.encode(
            "{\n \"prompt\": \"Welcome hello\","
            "\n  \"name\": \"fn_greet\","
            "\n  \"parameters\": {\n    \"name\": "
            "\"hello\"\n   }\n}\n"), functions_token)
        self.add_functions_name_token(functions_token)
        return functions_token

    def generate_response(self, ids, logit, prompt: str) -> None:
        braket_opens: int = 1
        prompt = json.dumps(prompt)
        if "\"" in prompt:
            response: str = "{\n\"prompt\": " + \
                prompt + ",\n  \"name\": \""
        else:
            response: str = "{\n\"prompt\": \"" + prompt + "\",\n  \"name\": \""
        last_token: str = ""
        while 1:
            next_token_id = logit.index(max(logit))
            ids.append(next_token_id)
            logit = self.llm.get_logits_from_input_ids(
                ids)
            last_token = self.llm.decode(ids[-1])
            if "{" in last_token:
                for c in last_token:
                    if c == "{":
                        braket_opens += 1
            elif "}" in last_token:
                for c in last_token:
                    if c == "}":
                        braket_opens -= 1
            response += self.llm.decode(ids[-1])
            if braket_opens == 0:
                break
        return response

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

    def add_functions_name_token(
            self,
            ids: list[int] = None
    ) -> list[int]:
        if ids is None:
            ids = []
        self.torch_tensor_to_ids(self.llm.encode(
            "The functions names are:\n"), ids)
        for f in self.functions:
            name_encode = self.llm.encode(f["name"] + "\n")
            self.torch_tensor_to_ids(name_encode, ids)
