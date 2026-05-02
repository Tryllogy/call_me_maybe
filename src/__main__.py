import argparse
import sys

from .llm import LlmInterpret


def main() -> None:
    """Run the command-line workflow for LLM function calling."""
    llm: LlmInterpret
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
        )
        parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calls.json",
        )
        arguments: argparse.Namespace = parser.parse_args()
        llm = LlmInterpret.from_json(
            functions_path_str=arguments.functions_definition,
            input_path_str=arguments.input,
            name_output_file=arguments.output,
        )
    except Exception as exc:
        if hasattr(exc, "errors"):
            for error in exc.errors():
                print(error["msg"])
        else:
            print("Error:", exc)
        sys.exit(1)

    try:
        llm.interpret()
    except Exception as exc:
        print(f"Error: {exc}")
    sys.exit(0)


if __name__ == "__main__":
    main()
