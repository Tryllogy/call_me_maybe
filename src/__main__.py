from .parsing import Parsing
import sys


def main() -> None:
    if len(sys.argv) != 7:
        print("Too few arguments, try", end=" ")
        print("--functions_definition <functions_definition>", end=" ")
        print("--input <input_data>", end=" ")
        print("--output <output_file>")
        return
    try:
        parser: Parsing = Parsing.from_json(sys.argv[2], sys.argv[4])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
