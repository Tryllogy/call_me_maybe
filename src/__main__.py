from .llm import LlmInterpret
import argparse
import sys


def main() -> None:
    """Run the command-line workflow for LLM function calling.

    The program expects exactly three flag/value pairs:
    ``--functions_definition``, ``--input``, and ``--output``.
    Base arguments are provided.

    Returns:
        None.
    """
    llm: LlmInterpret
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--functions_definition", type=str,
                            default="data/input/functions_definition.json")
        parser.add_argument("--input", type=str,
                            default="data/input/function_calling_tests.json")
        parser.add_argument("--output", type=str,
                            default="data/output/function_calls.json")
        arguments = parser.parse_args()
        print(arguments.functions_definition)
        llm = LlmInterpret.from_json(
            functions_path_str=arguments.functions_definition,
            input_path_str=arguments.input,
            name_output_file=arguments.output
        )
    except Exception as e:
        if hasattr(e, "errors"):
            for error in e.errors():
                print(error["msg"])
        else:
            print("Error:", e)
        sys.exit(1)

    try:
        llm.interpret()
    except Exception as e:
        print(f"Error: {e}")
    sys.exit(0)


if __name__ == "__main__":
    main()
