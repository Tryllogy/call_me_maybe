from pydantic import BaseModel, field_validator


class PromptValidator(BaseModel):
    """Validate a single user prompt.

    Attributes:
        prompt: The prompt text received from the user.
    """

    prompt: str

    @field_validator("prompt", mode="after")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Ensure the prompt is not empty.

        Args:
            value: The prompt string.

        Returns:
            The validated prompt.

        Raises:
            ValueError: If the prompt is empty.
        """
        if len(value) == 0:
            raise ValueError("Prompt cannot be empty")
        return value
