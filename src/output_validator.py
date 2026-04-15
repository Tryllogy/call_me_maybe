from pydantic import BaseModel, Field, model_validator
from typing import Any


class OutputValidator(BaseModel):
    prompt: str = Field(...)
    name: str = Field(...)
    parameters: dict[str, Any] = Field(...)
    valid_names: list[str] = Field(...)

    @model_validator(mode="after")
    def validate_name(self) -> "OutputValidator":
        if self.name not in self.valid_names:
            raise ValueError(
                f"The 'name' field must be one of \
                the following: {self.valid_names}"
            )
        return self
