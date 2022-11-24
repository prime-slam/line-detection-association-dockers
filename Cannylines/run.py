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

from pathlib import Path

from adapter import Adapter
from common.parser import create_base_parser, positive_int, positive_float

if __name__ == "__main__":
    parser = create_base_parser(with_score_directory=False)

    parser.add_argument(
        "--kernel",
        "-k",
        metavar="NUM",
        help="convolutional kernel size",
        default=3,
        type=positive_int,
    )

    parser.add_argument(
        "--sigma",
        "-S",
        metavar="NUM",
        help="normal distribution sigma",
        default=8,
        type=positive_float,
    )

    args = parser.parse_args()
    Adapter(
        image_path=Path(args.imgs),
        output_path=Path(args.output),
        lines_output_directory=Path(args.lines_dir),
        scores_output_directory=None,
        sigma=args.sigma,
        kernel_size=args.kernel,
    ).run()
