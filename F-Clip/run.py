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

from adapter import Adapter, Device
from common.parser import create_base_parser, positive_int

if __name__ == "__main__":
    parser = create_base_parser()

    parser.add_argument(
        "--batch",
        "-b",
        metavar="NUM",
        help="dataloader batch size",
        default=1,
        type=positive_int,
    )

    parser.add_argument(
        "--base-config",
        "-B",
        metavar="PATH",
        dest="base_config",
        help="base model configuration path",
        default=Path(__file__).resolve().parent / "config/base.yaml",
    )

    parser.add_argument(
        "--model-config",
        "-m",
        metavar="PATH",
        help="pretrained model configuration path",
        default=Path(__file__).resolve().parent / "config/fclip_HR.yaml",
    )

    parser.add_argument(
        "--model",
        "-M",
        metavar="PATH",
        help="pretrained model path",
        default=Path(__file__).resolve().parent / "pretrained/HR/checkpoint.pth.tar",
    )

    args = parser.parse_args()
    Adapter(
        image_path=Path(args.imgs),
        output_path=Path(args.output),
        lines_output_directory=Path(args.lines_dir),
        scores_output_directory=Path(args.scores_dir),
        base_config_path=Path(args.base_config),
        model_config_path=Path(args.model_config),
        pretrained_model_path=Path(args.model),
        device=Device[args.device],
        batch_size=args.batch,
    ).run()
