"""
One-time utility: scan data/staff_photos/, compute face encodings, save to
models/face_encodings.pkl.

Run from the project root:
    python src/face_encoder.py

Re-run whenever you add or remove staff photos.
"""

import sys
import pickle
from pathlib import Path

# Allow running as `python src/face_encoder.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import face_recognition
from src.config import STAFF_PHOTOS_DIR, FACE_ENCODINGS_PATH

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_encodings() -> dict:
    """
    Walk STAFF_PHOTOS_DIR, encode every face image, and return
    {"names": [...], "encodings": [...]}.  The two lists are parallel —
    names[i] is the person whose encoding is in encodings[i].
    """
    if not STAFF_PHOTOS_DIR.exists():
        raise FileNotFoundError(
            f"Staff photos folder not found: {STAFF_PHOTOS_DIR}\n"
            "Create it and add one sub-folder per person."
        )

    person_dirs = sorted(d for d in STAFF_PHOTOS_DIR.iterdir() if d.is_dir())
    if not person_dirs:
        raise ValueError(
            "No person sub-folders found inside staff_photos/.\n"
            "Expected layout:\n"
            "  staff_photos/\n"
            "    john_doe/\n"
            "      photo1.jpg\n"
        )

    names, encodings = [], []

    for person_dir in person_dirs:
        person_name = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

        if not images:
            print(f"  [SKIP] {person_name} — no image files found")
            continue

        print(f"  Encoding {person_name} ({len(images)} image(s))...")
        for img_path in sorted(images):
            img = face_recognition.load_image_file(str(img_path))
            found = face_recognition.face_encodings(img)

            if len(found) == 0:
                print(f"    [WARN] No face detected in {img_path.name} — skipping")
                continue
            if len(found) > 1:
                print(f"    [WARN] {len(found)} faces in {img_path.name} — using the first one")

            names.append(person_name)
            encodings.append(found[0])

    return {"names": names, "encodings": encodings}


def main():
    print("=" * 50)
    print("Staff Face Encoder")
    print("=" * 50)

    data = build_encodings()

    if not data["encodings"]:
        print("\n[ERROR] No encodings were produced. Check the warnings above.")
        sys.exit(1)

    FACE_ENCODINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FACE_ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

    unique_people = len(set(data["names"]))
    print(f"\nDone. {len(data['encodings'])} encoding(s) for {unique_people} person(s).")
    print(f"Saved → {FACE_ENCODINGS_PATH}")


if __name__ == "__main__":
    main()
