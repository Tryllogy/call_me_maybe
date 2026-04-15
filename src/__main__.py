from .llm import LlmInterpret
import sys


def main() -> None:
    if len(sys.argv) != 7:
        print("Too few arguments, try", end=" ")
        print("--functions_definition <functions_definition>", end=" ")
        print("--input <input_data>", end=" ")
        print("--output <output_file>")
        return
    try:
        print("Mounting LlmInterpret")
        llm: LlmInterpret = LlmInterpret.from_json(sys.argv[2],
                                                   sys.argv[4], sys.argv[6])
        print("Done")
    except Exception as e:
        if hasattr(e, "errors"):
            for error in e.errors():
                print(error["msg"])
        else:
            print("Error:", e)
        sys.exit(1)

    llm.interpret()
    sys.exit(0)


if __name__ == "__main__":
    main()
