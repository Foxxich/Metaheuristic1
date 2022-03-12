import argparse

from tsplib_parser.tsp_file_parser import TSPParser


def parse_boolean(value: str) -> bool:
    value = value.lower()

    if value in ["true", "yes", "y", "1", "t"]:
        return True
    elif value in ["false", "no", "n", "0", "f"]:
        return False

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="tsp file to parse.", type=str)
    args = parser.parse_args()
    file_name: str = args.file
    should_plot: bool = True
    if file_name is not None:
        f = open(file_name)
        lines = f.readlines()
        if "EUC_2D" in lines[4]:  # lines with given type
            TSPParser(filename=file_name, plot_tsp=should_plot)
        # else if in lines[5] : # lines with given type
