# Smart Hygiene & Intrusion Monitoring System

Camera-based monitoring that recognises staff by face, flags unknown people (intrusion),
checks PPE (head cap, gloves, apron, slip-ins) with a YOLOv8 model, gives spoken alerts,
and tracks presence/activity per staff member.

## Layout

```
src/
  main.py                  entry point (video loop, press Q to quit)
  config.py                all tunable settings (overridable via .env)
  face_encoder.py          builds models/face_encodings.pkl from data/staff_photos/
  intrusion_detector.py    face identification
  ppe_detector.py          YOLOv8 PPE detection (needs models/ppe_yolov8.pt)
  alert.py                 text-to-speech alerts with cooldown
  productivity_tracker.py  presence + activity scoring, CSV report
  check_install.py         verifies the required libraries import
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/check_install.py
```

1. Add staff photos to `data/staff_photos/<name>/`
2. Build face encodings: `python src/face_encoder.py`
3. (Optional) place a trained PPE model at `models/ppe_yolov8.pt`
4. Run: `python src/main.py`
