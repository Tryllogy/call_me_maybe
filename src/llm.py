import json
import os
import time
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from llm_sdk import Small_LLM_Model
from .output_validator import OutputValidator
from .parsing_validator import ParsingValidator
from .prompt_validator import PromptValidator


class LlmInterpret:
    """Interpret user prompts and produce validated function calls.

    Attributes:
        functions: Function definitions loaded from the input JSON file.
        input_data: Prompt payloads to process.
        llm: Loaded lightweight LLM backend.
        name_output_file: Destination JSON path for generated outputs.
        valid_names: Allowed function names extracted from definitions.
    """

    def __init__(
        self,
        functions: list[dict[str, Any]],
        input_data: list[dict[str, Any]],
        name_output_file: str
    ) -> None:
        """Initialize the interpreter with validated input payloads.

        Args:
            functions: Function specification list.
            input_data: List of prompt dictionaries.
            name_output_file: JSON output path.
        """

        ParsingValidator(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file
        )

        self.functions: list[dict[str, Any]] = functions
        self.input_data: list[dict[str, Any]] = input_data
        self.llm: Small_LLM_Model = Small_LLM_Model()
        self.name_output_file: str = name_output_file
        self.valid_names: list[str] = [function["name"]
                                       for function in functions]
        self.create_output_file()

    @classmethod
    def from_json(
        cls,
        functions_path_str: str,
        input_path_str: str,
        name_output_file: str
    ) -> "LlmInterpret":
        """Build an interpreter instance from JSON files.

        Args:
            functions_path_str: Path to function definitions JSON file.
            input_path_str: Path to prompt input JSON file.
            name_output_file: Output file path for generated results.

        Returns:
            A configured ``LlmInterpret`` instance.

        Raises:
            ValueError: If one of the paths does not end with ``.json``.
        """

        if (
            not functions_path_str.endswith(".json")
            or not input_path_str.endswith(".json")
        ):
            raise ValueError(
                "The paths for functions and input data must end with '.json'"
            )

        with open(functions_path_str, "r", encoding="utf-8") as file:
            functions: list[dict[str, Any]] = json.loads(file.read())

        with open(input_path_str, "r", encoding="utf-8") as file:
            input_data: list[dict[str, Any]] = json.loads(file.read())

        return cls(
            functions=functions,
            input_data=input_data,
            name_output_file=name_output_file,
        )

    def create_output_file(self) -> None:
        """Create the output JSON file and initialize it with an empty list."""
        path: str = os.path.dirname(self.name_output_file)
        if path:
            os.makedirs(path, exist_ok=True)
        with open(self.name_output_file, "w", encoding="utf-8") as file:
            file.write("[]\n")

    def interpret(self) -> None:
        """Generate and validate one function call per input prompt."""
        base_helper: str = self.make_base_helper()
        tensor: Any = None
        outputs: list[dict[str, Any]] = []

        init_time: float = time.time()
        for data in self.input_data:
            prompt: str = "Invalid prompt"
            dict_json: dict[str, Any] = {
                "prompt": prompt,
                "name": "Unknown function name",
                "parameters": {},
            }
            try:
                PromptValidator(prompt=data["prompt"])
            except Exception:
                prompt = "Invalid prompt"
            else:
                time_input: float = time.time()
                prompt = data["prompt"]
                response_prefix: str = '{"name":"'
                input_helper: str = (
                    f"{base_helper}\n"
                    f"User request:\n{prompt}\n"
                    f"JSON:\n{response_prefix}"
                )
                tensor = self.llm.encode(input_helper)
                response: str = response_prefix + \
                    self.generate_response(tensor, prompt)
                print(
                    f"Total time for prompt '{prompt}': "
                    f"{round(time.time() - time_input, 2)} seconds"
                )
                dict_json = self.make_dict_for_json(
                    response=response,
                    prompt=prompt,
                )
                try:
                    OutputValidator(
                        prompt=prompt,
                        name=dict_json["name"],
                        parameters=dict_json["parameters"],
                        valid_names=self.valid_names,
                    )
                except Exception:
                    dict_json = {
                        "prompt": prompt,
                        "name": "Unknown function name",
                        "parameters": {},
                    }
                else:
                    for key, value in dict_json["parameters"].items():
                        if isinstance(value, int):
                            dict_json["parameters"][key] = float(value)
            outputs.append(dict_json)
            print(
                f"Output:\n{json.dumps(outputs[-1], indent=4)}\n"
            )

        with open(self.name_output_file, "w", encoding="utf-8") as file:
            json.dump(outputs, file, indent=4)
            file.write("\n")
        print(f"Total time: '{round(time.time() - init_time, 2)}' seconds")

    def make_base_helper(self) -> str:
        """Create the fixed instruction prefix sent to the model."""
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
            "Output must be valid JSON.\n"
            "Return exactly one JSON object with exactly two keys: "
            "\"name\" and \"parameters\".\n"
            "No markdown, no prose, no extra keys.\n"
            "The object must start with { and end with }.\n"
            "The \"name\" value must match one allowed function.\n"
            "The \"parameters\" value must be an object using only "
            "parameters of the chosen function.\n"
            "Example output:\n"
            "{\n"
            '  "name": "fn_greet",\n'
            '  "parameters": {\n'
            '    "name": "Josh"\n'
            "  }\n"
            "}\n"
            "Choose the best function from the list below.\n"
            "Allowed functions:\n"
        )
        base_helper += json.dumps(functions, separators=(",", ":"))
        return base_helper

    def generate_response(
        self,
        tensor: Any,
        prompt: str,
        max_tokens: int = 200
    ) -> str:
        """Generate a constrained JSON response for one prompt.

        Args:
            tensor: Tokenized prompt prefix produced by the model encoder.
            prompt: Original user prompt.
            max_tokens: Maximum number of decoding steps to perform.

        Returns:
            The decoded JSON-like function call.
        """
        response: list[int] = []
        ids: list[int] = list(tensor[0])
        tokens: int = 0
        state: str = "name_value"
        name: list[int] = []
        chosen_func: str = ""
        param_key: list[int] = []
        param_value: list[int] = []
        av_tok_name: list[int] = self.get_available_tokens_name()
        params_count: int = 0
        next_token_id: int = 0
        after_name_ids: list[int] = list(self.llm.encode('",')[0])
        parameters_key_ids: list[int] = list(
            self.llm.encode('"parameters":{"')[0]
        )
        av_tok_param_key: Optional[list[int]] = None
        colon_ids: list[int] = list(self.llm.encode(':"')[0])
        av_tok_param_value: Optional[list[int]] = None
        end_json_ids: list[int] = list(self.llm.encode("}}")[0])
        while tokens <= max_tokens:
            match state:
                case "name_value":
                    next_token_id = self.get_constrained_logit(
                        av_tok_name, ids)
                    name.append(next_token_id)
                    ids.append(next_token_id)
                    tokens += 1
                    decoded_name: str = self.llm.decode(name)
                    for valid_name in self.valid_names:
                        if valid_name == decoded_name:
                            response += name
                            chosen_func = valid_name
                            state = "after_name"
                            break
                case "after_name":
                    ids += after_name_ids
                    response += after_name_ids
                    tokens += len(after_name_ids)
                    state = "parameters_start"
                case "parameters_start":
                    ids += parameters_key_ids
                    response += parameters_key_ids
                    tokens += len(parameters_key_ids)
                    state = "parameters_key"
                case "parameters_key":
                    if av_tok_param_key is None:
                        av_tok_param_key = \
                            self.get_available_tokens_param_key(chosen_func)
                    next_token_id = self.get_constrained_logit(
                        av_tok_param_key, ids)
                    param_key.append(next_token_id)
                    ids.append(next_token_id)
                    tokens += 1
                    decoded_param_key: str = self.llm.decode(param_key)
                    if decoded_param_key.endswith('"'):
                        response += param_key
                        state = "parameters_colon"
                case "parameters_colon":
                    ids += colon_ids
                    response += colon_ids
                    tokens += len(colon_ids)
                    state = "parameters_value"
                case "parameters_value":
                    if av_tok_param_value is None:
                        av_tok_param_value = \
                            self.get_available_tokens_param_value(prompt)
                    next_token_id = self.get_constrained_logit(
                        av_tok_param_value, ids)
                    param_value.append(next_token_id)
                    ids.append(next_token_id)
                    tokens += 1
                    decoded_param_value: str = self.llm.decode(param_value)
                    if decoded_param_value.endswith('"'):
                        response += param_value
                        params_count += 1
                        state = "check_end"
                case "check_end":
                    max_params: int = self.check_all_param_here(chosen_func)
                    if params_count >= max_params:
                        state = "end"
                    else:
                        comma_sep_ids: list[int] = list(
                            self.llm.encode(',"')[0])
                        ids += comma_sep_ids
                        response += comma_sep_ids
                        tokens += len(comma_sep_ids)
                        param_key.clear()
                        param_value.clear()
                        state = "parameters_key"
                case "end":
                    response += end_json_ids
                    break
        return str(self.llm.decode(response))

    def check_all_param_here(self, func_name: str) -> int:
        """Return the number of parameters expected by a function.

        Args:
            func_name: Name of the function being generated.

        Returns:
            The number of parameters for the matching function, or 0.
        """
        for function in self.functions:
            if function["name"] == func_name:
                return len(function["parameters"])
        return 0

    def get_constrained_logit(
            self,
            av_tokens: list[int],
            ids: list[int]
    ) -> int:
        """Select the highest-scoring token among the allowed tokens.

        Args:
            av_tokens: Token ids allowed at the current decoding step.
            ids: Current token sequence.

        Returns:
            The selected token id.
        """
        logits: NDArray[Any] = np.array(
            self.llm.get_logits_from_input_ids(ids))
        constrained_logits: NDArray[Any] = np.full(
            len(logits),
            -np.inf,
            dtype=float,
        )
        for token in av_tokens:
            constrained_logits[token] = logits[token]
        next_token_id: np.int64 = constrained_logits.argmax()
        return int(next_token_id)

    def get_available_tokens_param_key(
        self,
        function_name: str,
    ) -> list[int]:
        """Return token ids allowed for a parameter key.

        Args:
            function_name: Name of the selected function.

        Returns:
            A deduplicated list of token ids.
        """
        available_tokens: list[int] = self.llm.encode('"')[0].tolist()

        for function in self.functions:
            if function["name"] == function_name:
                for key, _ in function["parameters"].items():
                    available_tokens.extend(self.llm.encode(key)[0].tolist())
                break
        return list(set(available_tokens))

    def get_available_tokens_param_value(
            self,
            prompt: str
    ) -> list[int]:
        """Return token ids allowed for a parameter value.

        Args:
            prompt: Original user prompt.

        Returns:
            A deduplicated list of token ids.
        """
        available_tokens: list[int] = self.llm.encode('"[]\\/')[0].tolist()
        available_tokens.extend(self.llm.encode(" ")[0].tolist())
        available_tokens.extend(self.llm.encode(prompt)[0].tolist())
        split_prompt: list[str] = prompt.split()
        for word in split_prompt:
            available_tokens.extend(self.llm.encode(word)[0].tolist())
            available_tokens.extend(self.llm.encode(f" {word}")[0].tolist())

        return list(set(available_tokens))

    def get_available_tokens_name(self) -> list[int]:
        """Return token ids allowed for a function name.

        Returns:
            A deduplicated list of token ids.
        """
        available_tokens: list[int] = self.llm.encode('",:{} ')[0].tolist()

        for func_name in self.valid_names:
            available_tokens.extend(self.llm.encode(func_name)[0].tolist())
        return list(set(available_tokens))

    def make_dict_for_json(
            self,
            response: str,
            prompt: str = "",
    ) -> dict[str, Any]:
        """Convert model output text into the canonical output dictionary.

        Args:
            response: Raw JSON-like model response string.
            prompt: Original user prompt.

        Returns:
            A dictionary with ``prompt``, ``name``, and ``parameters`` keys.
        """
        dict_json: dict[str, Any] = {
            "prompt": prompt,
            "name": "",
            "parameters": {},
        }
        try:
            response_dict: dict[str, Any] = json.loads(response)
            if response_dict.get("prompt"):
                dict_json.update({"prompt": response_dict["prompt"]})
            if response_dict.get("name"):
                dict_json.update({"name": response_dict["name"]})
            if response_dict.get("parameters"):
                dict_json["parameters"] = response_dict["parameters"]
        except json.JSONDecodeError:
            return dict_json
        return dict_json
