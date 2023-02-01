from typing import List
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from common.adapter.tensorflow_adapter import TensorflowAdapter
from common.device import Device
from common.image_metadata import ImageMetadata
from common.dataset.line_dataset import LineDataset
from common.prediction import Prediction


class Adapter(TensorflowAdapter):
    def __init__(
        self,
        image_path: Path,
        output_path: Path,
        lines_output_directory: Path,
        scores_output_directory: Path,
        pretrained_model_path: Path,
        device: Device,
    ):
        super().__init__(
            image_path,
            output_path,
            lines_output_directory,
            scores_output_directory,
            device,
        )
        self.model_path = pretrained_model_path
        self.model_input_size = (512, 512)
        self.model_output_size = (256, 256)
        self.score_threshold = 0.01
        self.endpoints_distance_threshold = 2.0

    def _predict(self, model, image):
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        model.set_tensor(input_details[0]["index"], image)
        model.invoke()

        centers = model.get_tensor(output_details[0]["index"])[0]
        centers_score = model.get_tensor(output_details[1]["index"])[0]
        endpoints = model.get_tensor(output_details[2]["index"])[0]

        start = endpoints[:, :, :2]
        end = endpoints[:, :, 2:]

        distances = np.sqrt(np.sum((start - end) ** 2, axis=-1))

        lines = []
        scores = []

        for center, score in zip(centers, centers_score):
            y, x = center
            distance = distances[y, x]
            if (
                score > self.score_threshold
                and distance > self.endpoints_distance_threshold
            ):
                x_start_shift, y_start_shift, x_end_shift, y_end_shift = endpoints[
                    y, x, :
                ]

                x_start = x + x_start_shift
                y_start = y + y_start_shift
                x_end = x + x_end_shift
                y_end = y + y_end_shift

                lines.append([x_start, y_start, x_end, y_end])
                scores.append(score)

        return np.array(lines), np.array(scores)

    def _create_imageloader(self):
        return LineDataset(self.image_path, self._transform_image)

    def _transform_image(self, image: np.ndarray):
        transformed = cv2.resize(
            image, self.model_input_size, interpolation=cv2.INTER_AREA
        )
        # add 4th channel to each pixel
        transformed = np.concatenate(
            [transformed, np.ones([*self.model_input_size, 1])], axis=-1
        )
        transformed = transformed[np.newaxis].astype("float32")
        return transformed

    def _build_model(self):
        model = tf.lite.Interpreter(model_path=str(self.model_path))
        model.allocate_tensors()
        return model

    def _postprocess_predictions(
        self, raw_predictions, metadata: ImageMetadata
    ) -> List[Prediction]:
        lines, scores = raw_predictions
        output_height, output_width = self.model_output_size

        x_scale = metadata.width / output_width
        y_scale = metadata.height / output_height

        x_index = [0, 2]
        y_index = [1, 3]

        lines[:, x_index] *= x_scale
        lines[:, y_index] *= y_scale

        return [Prediction(lines=lines, scores=scores, metadata=metadata)]
