from pydantic import BaseModel, Field, model_validator
from typing import Any


class OutputValidator(BaseModel):
    """Validate one generated function-call payload.

    Attributes:
        prompt: Original user prompt.
        name: Generated function name.
        parameters: Generated function arguments.
        valid_names: Allowed function names.
    """

    prompt: str = Field(..., description="Original user prompt.")
    name: str = Field(..., description="Generated function name.")
    parameters: dict[str, Any] = Field(
        ..., description="Generated function parameters."
    )
    valid_names: list[str] = Field(
        ..., description="List of allowed function names."
    )

    @model_validator(mode="after")
    def validate_name(self) -> "OutputValidator":
        """Ensure the generated name belongs to the allowed function names.

        Returns:
            The validated model instance.

        Raises:
            ValueError: If ``name`` is not present in ``valid_names``.
        """
        if self.name not in self.valid_names:
            raise ValueError(
                "The 'name' field must be one of the following: "
                f"{self.valid_names}"
            )
        return self
