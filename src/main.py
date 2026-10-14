"""
Smart Hygiene & Intrusion Monitoring System — main entry point.

Run from the project root:
    python src/main.py

Prerequisites:
    1. Staff photos added to data/staff_photos/<name>/
    2. Face encodings built:  python src/face_encoder.py
    3. (Phase 4) PPE model trained and placed at models/ppe_yolov8.pt

Press Q in the video window to quit.
"""

import sys
import cv2
from pathlib import Path

# Allow `python src/main.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    VIDEO_SOURCE, FACE_ENCODINGS_PATH, PPE_MODEL_PATH,
    FACE_DETECTION_INTERVAL,
)
from src.intrusion_detector import FaceIdentifier, DetectedPerson
from src.ppe_detector import PPEDetector
from src.alert import AlertEngine
from src.productivity_tracker import ProductivityTracker

# ── Annotation colours (BGR) ──────────────────────────────────────────────────
GREEN  = (0, 200, 0)    # compliant known staff
YELLOW = (0, 200, 255)  # known staff with PPE violation
RED    = (0, 0, 220)    # intruder
WHITE  = (255, 255, 255)


def annotate_frame(frame, person: DetectedPerson, violations: list[str]) -> None:
    """Draw a labelled bounding box and any PPE violations onto frame in-place."""
    top, right, bottom, left = person.face_location

    if not person.is_known:
        colour, label = RED, "INTRUDER"
    elif violations:
        colour, label = YELLOW, person.name
    else:
        colour, label = GREEN, person.name

    # Face box
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
    # Filled label bar below box
    cv2.rectangle(frame, (left, bottom - 22), (right, bottom), colour, cv2.FILLED)
    cv2.putText(frame, label, (left + 4, bottom - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    # Missing PPE items listed above the box
    for i, item in enumerate(violations):
        cv2.putText(frame, f"Missing: {item}", (left, top - 10 - i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)


def main():
    # ── Load face identifier ──────────────────────────────────────────────────
    try:
        identifier = FaceIdentifier(FACE_ENCODINGS_PATH)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    # ── Load PPE detector (optional until Phase 4) ────────────────────────────
    ppe = None
    try:
        ppe = PPEDetector(PPE_MODEL_PATH)
        print("PPE detector loaded.")
    except FileNotFoundError:
        print("[INFO] PPE model not found — PPE checking is disabled until Phase 4.")

    alert   = AlertEngine()
    tracker = ProductivityTracker()

    # ── Open video source ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {VIDEO_SOURCE}")
        sys.exit(1)

    print("Monitoring started. Press Q to quit.")

    frame_idx = 0
    last_people: list[DetectedPerson] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── Face detection (throttled for performance) ────────────────────────
        if frame_idx % FACE_DETECTION_INTERVAL == 0:
            last_people = identifier.identify(frame)

        # ── PPE check (full-frame; per-person association added in Phase 4) ───
        global_violations: list[str] = []
        if ppe is not None:
            detected_ppe = ppe.detect(frame)
            global_violations = ppe.missing_items(detected_ppe)

        # ── Annotate + alert ──────────────────────────────────────────────────
        for person in last_people:
            violations = global_violations
            annotate_frame(frame, person, violations)

            tracker.update(person, frame)

            if not person.is_known:
                if violations:
                    alert.speak("Unknown person detected. Please wear the required safety gear.", person.name)
            elif violations:
                missing = " and ".join(violations)
                alert.speak(
                    f"{person.name.replace('_', ' ')}, please wear your {missing}.",
                    person.name,
                )

        cv2.imshow("Hygiene Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    alert.shutdown()

    report = tracker.save_report()
    if report:
        print(f"\nProductivity report saved → {report}")


if __name__ == "__main__":
    main()
