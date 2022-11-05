import argparse

from os import path
from adapter import Adapter, Device


def positive_int(value: str) -> int:
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value} is not an integer") from error
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python run.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--imgs", "-i", metavar="PATH", help="path to images", default="input/"
    )

    parser.add_argument(
        "--output", "-o", metavar="PATH", help="output path", default="output/"
    )

    parser.add_argument(
        "--lines-dir",
        "-l",
        metavar="STRING",
        dest="lines_dir",
        help="name of lines output directory",
        default="lines",
    )

    parser.add_argument(
        "--scores-dir",
        "-s",
        metavar="STRING",
        dest="scores_dir",
        help="name of scores output directory",
        default="scores",
    )

    parser.add_argument(
        "--batch",
        "-b",
        metavar="NUM",
        help="dataloader batch size",
        default=1,  # TODO: set 1
        type=positive_int,
    )

    parser.add_argument(
        "--base-config",
        "-B",
        metavar="PATH",
        dest="base_config",
        help="base model configuration path",
        default=path.join(path.dirname(__file__), "config/base.yaml"),
    )

    parser.add_argument(
        "--model-config",
        "-m",
        metavar="PATH",
        help="pretrained model configuration path",
        default=path.join(path.dirname(__file__), "config/fclip_HR.yaml"),
    )

    parser.add_argument(
        "--model",
        "-M",
        metavar="PATH",
        help="pretrained model path",
        default=path.join(path.dirname(__file__), "pretrained/HR/checkpoint.pth.tar"),
    )

    parser.add_argument(
        "--device",
        "-d",
        metavar="STRING",
        choices=list(map(lambda c: c.name, Device)),
        help="name of desired execution device",
        default="cuda",
    )

    args = parser.parse_args()
    Adapter(
        image_path=args.imgs,
        output_path=args.output,
        lines_output_directory=args.lines_dir,
        scores_output_directory=args.scores_dir,
        base_config_path=args.base_config,
        model_config_path=args.model_config,
        pretrained_model_path=args.model,
        device=Device[args.device],
        batch_size=args.batch,
    ).run()
