"""
Quick sanity check — run this once after setting up the venv to confirm
every required library imports correctly and reports its version.
"""

import sys

print(f"Python {sys.version}\n")

checks = []

try:
    import cv2
    checks.append(("opencv-python", cv2.__version__, True))
except ImportError as e:
    checks.append(("opencv-python", str(e), False))

try:
    import ultralytics
    checks.append(("ultralytics", ultralytics.__version__, True))
except ImportError as e:
    checks.append(("ultralytics", str(e), False))

try:
    import face_recognition
    checks.append(("face_recognition", "1.3.0", True))
except ImportError as e:
    checks.append(("face_recognition", str(e), False))

try:
    import numpy as np
    checks.append(("numpy", np.__version__, True))
except ImportError as e:
    checks.append(("numpy", str(e), False))

try:
    import pyttsx3
    import importlib.metadata as _meta
    checks.append(("pyttsx3", _meta.version("pyttsx3"), True))
except ImportError as e:
    checks.append(("pyttsx3", str(e), False))

try:
    import dotenv
    import importlib.metadata
    checks.append(("python-dotenv", importlib.metadata.version("python-dotenv"), True))
except ImportError as e:
    checks.append(("python-dotenv", str(e), False))

all_ok = True
for name, version, ok in checks:
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name:<20} {version}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("All libraries OK")
else:
    print("One or more libraries failed to import — see FAIL lines above.")
    sys.exit(1)
