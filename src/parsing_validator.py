from pydantic import BaseModel, Field, field_validator


class ParsingValidator(BaseModel):
    functions: list[dict] = Field(...)
    input_data: list[dict] = Field(...)
    name_output_file: str = Field(...)

    @field_validator("functions", mode="after")
    @classmethod
    def check_functions(cls, data: list[dict]) -> None:
        for func in data:
            if not isinstance(func, dict):
                raise TypeError("Each function must be a dictionary")
            if "name" not in func:
                raise ValueError("Each function must have a 'name' key")
            if "description" not in func:
                raise ValueError("Each function must have a 'description' key")
            if "parameters" not in func:
                raise ValueError("Each function must have a 'parameters' key")
            if not isinstance(func["parameters"], dict):
                raise ValueError("Each function's 'parameters' "
                                 "must be a dictionary")
            for p in func["parameters"]:
                if "type" not in func["parameters"][p]:
                    raise ValueError(f"Parameter '{p}' must have a 'type' key")
            if "returns" not in func:
                raise ValueError("Each function must have a 'returns' key")
            if not isinstance(func["returns"], dict):
                raise ValueError("Each function's 'returns' "
                                 "must be a dictionary")
            if "type" not in func["returns"]:
                raise ValueError(
                    "Each function's 'returns' must have a 'type' key")

    @field_validator("input_data", mode="after")
    @classmethod
    def check_input_data(cls, data: list[dict]) -> None:
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Each input data item must be a dictionary")
            if "prompt" not in item:
                raise ValueError("Each input data item must have "
                                 "a 'prompt' key")

    @field_validator("name_output_file", mode="after")
    @classmethod
    def check_name_output_file(cls, data: str) -> None:
        if not isinstance(data, str):
            raise TypeError("The name of the output file must be a string")
        if not data.endswith(".json"):
            raise ValueError(
                "The name of the output file must end with '.json'")
