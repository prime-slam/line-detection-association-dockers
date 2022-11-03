import argparse

from adapter import Adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python run.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--imgs", "-i", metavar="imgs", help="path to images", default="input/"
    )

    parser.add_argument(
        "--output", "-o", metavar="output", help="output path", default="output/"
    )

    args = parser.parse_args()
    Adapter(args.imgs, args.output).run()
