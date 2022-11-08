# Copyright (c) 2022, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

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
        default=1,
        type=positive_int,
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
        image_path=Path(args.imgs),
        output_path=Path(args.output),
        lines_output_directory=Path(args.lines_dir),
        scores_output_directory=Path(args.scores_dir),
        batch_size=args.batch,
    ).run()
