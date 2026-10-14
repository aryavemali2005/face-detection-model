"""
Face identification module.

Loads pre-computed staff encodings from models/face_encodings.pkl (built by
running src/face_encoder.py) and identifies faces in each video frame.
"""

import pickle
import cv2
import numpy as np
import face_recognition
from dataclasses import dataclass
from pathlib import Path
from src.config import FACE_MATCH_TOLERANCE, FRAME_SCALE_FACTOR


@dataclass
class DetectedPerson:
    name: str           # staff name, or "Intruder" for unknowns
    is_known: bool
    face_location: tuple  # (top, right, bottom, left) in original-frame pixels


class FaceIdentifier:
    def __init__(self, encodings_path: Path):
        if not encodings_path.exists():
            raise FileNotFoundError(
                f"Face encodings file not found: {encodings_path}\n"
                "Run `python src/face_encoder.py` after adding staff photos."
            )
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        self._names: list[str] = data["names"]
        self._encodings: list = data["encodings"]
        print(f"Loaded encodings for: {sorted(set(self._names))}")

    def identify(self, frame) -> list[DetectedPerson]:
        """
        Detect and identify all faces in `frame` (BGR numpy array).

        Scales the frame down by FRAME_SCALE_FACTOR before running face
        detection — this is the biggest performance lever for real-time video.
        Detected locations are scaled back to original-frame coordinates.
        """
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        people = []
        scale_inv = 1.0 / FRAME_SCALE_FACTOR

        for enc, loc in zip(encodings, locations):
            name, is_known = "Intruder", False

            if self._encodings:
                matches = face_recognition.compare_faces(
                    self._encodings, enc, tolerance=FACE_MATCH_TOLERANCE
                )
                if True in matches:
                    distances = face_recognition.face_distance(self._encodings, enc)
                    best_idx = int(np.argmin(distances))
                    name, is_known = self._names[best_idx], True

            # Scale bounding box back to original resolution
            top, right, bottom, left = loc
            scaled = (
                int(top * scale_inv), int(right * scale_inv),
                int(bottom * scale_inv), int(left * scale_inv),
            )
            people.append(DetectedPerson(name=name, is_known=is_known, face_location=scaled))

        return people
