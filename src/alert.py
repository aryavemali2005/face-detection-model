"""
Audio alert engine.  Uses pyttsx3 for offline text-to-speech — no internet needed.

Cooldown per person_id prevents the same alert from firing on every video frame.
"""

import time
import pyttsx3
from src.config import ALERT_COOLDOWN_SECONDS


class AlertEngine:
    def __init__(self):
        self._engine = pyttsx3.init()
        # 160 words/min is clearer than the default ~200 for facility announcements
        self._engine.setProperty("rate", 160)
        self._last_spoken: dict[str, float] = {}

    def speak(self, text: str, person_id: str = "global") -> bool:
        """
        Speak `text` aloud unless the same person_id was alerted within
        ALERT_COOLDOWN_SECONDS.  Returns True if the alert was actually spoken.
        """
        now = time.monotonic()
        if now - self._last_spoken.get(person_id, 0.0) < ALERT_COOLDOWN_SECONDS:
            return False

        self._last_spoken[person_id] = now
        self._engine.say(text)
        self._engine.runAndWait()
        return True

    def reset_cooldown(self, person_id: str) -> None:
        """Force the next speak() call for this person to fire immediately."""
        self._last_spoken.pop(person_id, None)

    def shutdown(self) -> None:
        self._engine.stop()
