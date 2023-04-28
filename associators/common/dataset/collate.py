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

from torch.utils.data import default_collate

from common.frames_pair import FramesPair


def collate_frames_pair(pair: FramesPair):
    return FramesPair(
        lines_pair=pair.lines_pair,
        images_metadata_pair=pair.images_metadata_pair,
        images_pair=default_collate(pair.images_pair),
    )


def collate(batch):
    return [collate_frames_pair(elem) for elem in batch]
