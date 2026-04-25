from typing import Any

from pydantic import BaseModel, Field, field_validator


class ParsingValidator(BaseModel):
    """Validate structure and required keys for project input payloads.

    Attributes:
        functions: Function definitions loaded from JSON.
        input_data: Prompt entries loaded from JSON.
        name_output_file: Target output file path.
    """

    functions: list[dict[str, Any]] = Field(
        ..., description="Function definition objects."
    )
    input_data: list[dict[str, Any]] = Field(
        ..., description="Prompt entries to process."
    )
    name_output_file: str = Field(..., description="Output JSON file path.")

    @field_validator("functions", mode="after")
    @classmethod
    def check_functions(
        cls,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate the function definition schema.

        Args:
            data: Function definition dictionaries.

        Returns:
            The validated function definitions.

        Raises:
            TypeError: If a function item is not a dictionary.
            ValueError: If required keys or nested fields are missing.
        """
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
                raise ValueError(
                    "Each function's 'parameters' must be a dictionary"
                )
            for parameter_name, parameter_data in func["parameters"].items():
                if not isinstance(parameter_data, dict):
                    raise TypeError(
                        "Each parameter definition must be a dictionary"
                    )
                if "type" not in parameter_data:
                    raise ValueError(
                        f"Parameter '{parameter_name}' must have a 'type' key"
                    )
            if "returns" not in func:
                raise ValueError("Each function must have a 'returns' key")
            if not isinstance(func["returns"], dict):
                raise ValueError(
                    "Each function's 'returns' must be a dictionary"
                )
            if "type" not in func["returns"]:
                raise ValueError(
                    "Each function's 'returns' must have a 'type' key"
                )
        return data

    @field_validator("input_data", mode="after")
    @classmethod
    def check_input_data(
        cls,
        data: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Validate prompt input payloads.

        Args:
            data: Prompt dictionaries.

        Returns:
            The validated prompt payloads.

        Raises:
            TypeError: If an input item is not a dictionary.
            ValueError: If an input item does not define ``prompt``.
        """
        for item in data:
            if not isinstance(item, dict):
                raise TypeError("Each input data item must be a dictionary")
            if "prompt" not in item:
                raise ValueError(
                    "Each input data item must have a 'prompt' key"
                )
        return data

    @field_validator("name_output_file", mode="after")
    @classmethod
    def check_name_output_file(cls, data: str) -> str:
        """Validate the output file name.

        Args:
            data: Output file path string.

        Returns:
            The validated output file path.

        Raises:
            TypeError: If the output path is not a string.
            ValueError: If the output path does not end with ``.json``.
        """
        if not isinstance(data, str):
            raise TypeError("The name of the output file must be a string")
        if not data.endswith(".json"):
            raise ValueError(
                "The name of the output file must end with '.json'"
            )
        return data
