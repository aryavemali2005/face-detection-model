"""
Productivity tracker — measures presence time and activity level per staff member.

Presence : total seconds the person's face is detected on camera.
Activity : how much the body region below their face changes between frames
           (still = low score, moving/working = high score).
Combined : weighted blend of presence % and activity score, normalised 0-100.

Usage:
    tracker = ProductivityTracker()
    # inside the frame loop:
    tracker.update(person, frame)
    # on shutdown:
    path = tracker.save_report()
"""

import csv
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.config import LOGS_DIR, ACTIVITY_WEIGHT
from src.intrusion_detector import DetectedPerson


@dataclass
class _PersonRecord:
    first_seen_mono: float = field(default_factory=time.monotonic)
    last_seen_mono: float  = field(default_factory=time.monotonic)
    first_seen_wall: str   = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    last_seen_wall: str    = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    presence_seconds: float = 0.0
    raw_motion_total: float = 0.0
    motion_samples: int     = 0
    prev_crop: np.ndarray | None = None


class ProductivityTracker:
    # Gaps larger than this (seconds) between detections are NOT counted as
    # presence — the person left the camera view and came back.
    _MAX_GAP = 2.0

    # A mean pixel difference of this value maps to a 100% activity score.
    # Typical values: ~5 = very slight motion, ~15 = active movement.
    _MOTION_CEILING = 15.0

    def __init__(self) -> None:
        self._records: dict[str, _PersonRecord] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, person: DetectedPerson, frame: np.ndarray) -> None:
        """Call once per frame for every known staff member in last_people."""
        if not person.is_known:
            return

        name = person.name
        now  = time.monotonic()

        if name not in self._records:
            self._records[name] = _PersonRecord(
                first_seen_mono=now,
                last_seen_mono=now,
                first_seen_wall=datetime.now().strftime("%H:%M:%S"),
                last_seen_wall=datetime.now().strftime("%H:%M:%S"),
            )
            return  # no elapsed time to count on first sighting

        rec = self._records[name]
        gap = now - rec.last_seen_mono

        if gap < self._MAX_GAP:
            rec.presence_seconds += gap

        rec.last_seen_mono = now
        rec.last_seen_wall = datetime.now().strftime("%H:%M:%S")

        # Motion: compare grayscale body crop to the previous frame's crop
        crop = self._body_crop(frame, person.face_location)
        if crop is not None and rec.prev_crop is not None:
            if rec.prev_crop.shape == crop.shape:
                diff = cv2.absdiff(crop, rec.prev_crop)
                rec.raw_motion_total += float(np.mean(diff))
                rec.motion_samples   += 1
        if crop is not None:
            rec.prev_crop = crop

    def save_report(self) -> Path | None:
        """
        Write a CSV to LOGS_DIR and return the file path.
        Returns None if no staff were tracked this session.
        """
        if not self._records:
            return None

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        today    = datetime.now().strftime("%Y-%m-%d")
        out_path = LOGS_DIR / f"productivity_{today}.csv"

        shift_start = min(r.first_seen_mono for r in self._records.values())
        shift_end   = max(r.last_seen_mono  for r in self._records.values())
        shift_minutes = (shift_end - shift_start) / 60.0

        rows = [
            self._row(name, rec, shift_minutes)
            for name, rec in sorted(self._records.items())
        ]

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        return out_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _body_crop(
        self, frame: np.ndarray, face_location: tuple
    ) -> np.ndarray | None:
        """Grayscale crop of the region below the face (captures torso movement)."""
        h, w = frame.shape[:2]
        top, right, bottom, left = face_location
        face_h = max(bottom - top, 1)

        crop_top    = top
        crop_bottom = min(h, bottom + face_h * 3)
        crop_left   = max(0, left)
        crop_right  = min(w, right)

        if crop_bottom <= crop_top or crop_right <= crop_left:
            return None

        region = frame[crop_top:crop_bottom, crop_left:crop_right]
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    def _motion_score(self, rec: _PersonRecord) -> float:
        """0-100 normalised motion score."""
        if rec.motion_samples == 0:
            return 0.0
        avg = rec.raw_motion_total / rec.motion_samples
        return min(100.0, (avg / self._MOTION_CEILING) * 100.0)

    def _combined_score(self, rec: _PersonRecord, shift_minutes: float) -> float:
        """0-100 weighted blend of presence % and activity score."""
        if shift_minutes <= 0:
            return 0.0
        presence_pct = min(100.0, (rec.presence_seconds / (shift_minutes * 60)) * 100.0)
        motion_pct   = self._motion_score(rec)
        return round(
            (1 - ACTIVITY_WEIGHT) * presence_pct + ACTIVITY_WEIGHT * motion_pct, 1
        )

    def _row(self, name: str, rec: _PersonRecord, shift_minutes: float) -> dict:
        return {
            "name":             name.replace("_", " ").title(),
            "date":             datetime.now().strftime("%Y-%m-%d"),
            "first_seen":       rec.first_seen_wall,
            "last_seen":        rec.last_seen_wall,
            "presence_seconds": round(rec.presence_seconds, 1),
            "motion_score":     round(self._motion_score(rec), 1),
            "combined_score":   self._combined_score(rec, shift_minutes),
        }
