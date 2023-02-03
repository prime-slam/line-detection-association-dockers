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

from pathlib import Path

from adapter import Adapter
from common.parser import positive_int, create_dl_base_parser

if __name__ == "__main__":
    parser = create_dl_base_parser(with_score_directory=False)

    parser.add_argument(
        "--batch",
        "-b",
        metavar="NUM",
        help="dataloader batch size",
        default=1,
        type=positive_int,
    )

    parser.add_argument(
        "--config",
        "-c",
        metavar="PATH",
        help="base model configuration path",
        default=Path(__file__).resolve().parent / "config/base.yaml",
    )

    parser.add_argument(
        "--model",
        "-m",
        metavar="PATH",
        help="model path",
        default=Path(__file__).resolve().parent / "model/model-4501600.index",
    )

    args = parser.parse_args()

    Adapter(
        images_path=Path(args.imgs),
        lines_path=Path(args.lines),
        associations_dir=args.associations_dir,
        output_path=Path(args.output),
        frames_step=args.step,
        config_path=Path(args.config),
        pairs_file=args.pairs,
        model_path=Path(args.model),
        device=args.device,
    ).run()
