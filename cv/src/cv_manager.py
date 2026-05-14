"""Manages the CV model."""

import io
from PIL import Image
from ultralytics import YOLO
from typing import Any


class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO
        self.model = YOLO('best.pt')

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        # Your inference code goes here.
        # TODO

        img = Image.open(io.BytesIO(image))

        results = self.model.predict(source=img, verbose=False, conf=0.5)

        formatted_predictions = []
        
        for result in results:

            result.save("debug_vision.jpg")
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                bbox_coords = box.xyxy[0].tolist()

                formatted_predictions.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox_coords
                })

        return formatted_predictions
