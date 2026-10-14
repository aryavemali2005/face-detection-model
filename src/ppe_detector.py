"""
PPE detection using a YOLOv8 model.

The model file (models/ppe_yolov8.pt) will be trained in Phase 4 once PPE
images are annotated.  Until then, constructing a PPEDetector raises
FileNotFoundError — main.py handles that gracefully and disables PPE checks.
"""

from pathlib import Path
from ultralytics import YOLO
from src.config import PPE_CONFIDENCE_THRESHOLD, REQUIRED_PPE_CLASSES


class PPEDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"PPE model not found: {model_path}\n"
                "Train the YOLOv8 model on annotated PPE data (Phase 4) and place\n"
                "the resulting weights file at that path."
            )
        self._model = YOLO(str(model_path))

    def detect(self, frame) -> list[str]:
        """
        Run inference on `frame` (BGR numpy array from OpenCV).
        Returns a list of class-name strings for every detection that meets
        the confidence threshold.  Duplicates are kept — two gloves still count
        as two detections.
        """
        results = self._model(frame, verbose=False)[0]
        detected = []
        for box in results.boxes:
            if float(box.conf) >= PPE_CONFIDENCE_THRESHOLD:
                detected.append(self._model.names[int(box.cls)])
        return detected

    def missing_items(self, detected: list[str]) -> list[str]:
        """
        Compare detected items against REQUIRED_PPE_CLASSES and return
        whatever is absent.  An empty list means full compliance.
        """
        detected_set = set(detected)
        return [item for item in REQUIRED_PPE_CLASSES if item not in detected_set]
