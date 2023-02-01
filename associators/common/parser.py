# Copyright (c) 2023, Kirill Ivanov, Anastasiia Kornilova and Dmitrii Iarosh
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

from numbers import Number

from common.device import Device


def positive_number(value: str, numeric_type: type) -> Number:
    try:
        value = numeric_type(value)
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"{value} is not a positive {numeric_type.__name__}"
            )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value} is not {numeric_type.__name__}"
        ) from error
    return value


def positive_int(value: str) -> Number:
    return positive_number(value, int)


def positive_float(value: str) -> Number:
    return positive_number(value, float)


def create_base_parser(with_score_directory: bool = True):
    parser = argparse.ArgumentParser(
        prog="python run.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--imgs", "-i", metavar="PATH", help="path to images", default="input/"
    )

    parser.add_argument(
        "--lines",
        "-l",
        metavar="PATH",
        help="path to predicted lines",
        default="lines/",
    )

    parser.add_argument(
        "--output", "-o", metavar="PATH", help="output path", default="output/"
    )

    parser.add_argument(
        "--associations-dir",
        "-a",
        metavar="STRING",
        help="name of associations output directory",
        default="associations",
    )

    if with_score_directory:
        parser.add_argument(
            "--scores-dir",
            "-S",
            metavar="STRING",
            help="name of scores output directory",
            default="scores",
        )

    parser.add_argument(
        "--pairs",
        "-p",
        metavar="PATH",
        help="path to a file with pairs of frames for which associations will be calculated",
        default=None,
    )

    parser.add_argument(
        "--step",
        metavar="NUM",
        help="step with which associations between "
        "frames will be calculated (does not work if a file with pairs is provided)",
        default=1,
        type=positive_int,
    )
    return parser


def create_dl_base_parser(with_score_directory: bool = True):
    parser = create_base_parser(with_score_directory)
    parser.add_argument(
        "--device",
        "-d",
        metavar="STRING",
        choices=list(map(lambda c: c.name, Device)),
        help="name of desired execution device",
        default="gpu",
    )

    return parser
