from pydantic import BaseModel, field_validator


class PromptValidator(BaseModel):
    prompt: str

    @field_validator('prompt', mode='after')
    def validate_prompt(cls, v: str) -> None:
        """
        Validate the prompt.

        Args:
            v: The prompt string.

        Returns:
            None.

        Raises:
            ValueError: If the prompt is empty.
        """
        if len(v) == 0:
            raise ValueError("Prompt cannot be empty")
