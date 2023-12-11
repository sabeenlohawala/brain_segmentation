import argparse


def test():
    """
    This is the main function of the program.
    It creates an argument parser and parses the command line arguments.
    Returns the parsed arguments.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args


if __name__ == "__main__":
    test()
