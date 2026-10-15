# claim-photo-toolkit

Check car-damage photos submitted with insurance claims.

People editing a claim can reuse a photo from somewhere else, run it through an editor or a generator, or strip the camera metadata before upload. This toolkit runs the same four checks on every file and returns one per-photo and per-claim report. The report is a triage signal for a human reviewer. It is not a fraud decision.

## Install

The package is not published on PyPI. Install it from a checkout:

```bash
git clone https://github.com/saiobuladinne/claim-photo-toolkit.git
cd claim-photo-toolkit
python -m pip install -e .
```

That install is enough for duplicate detection, EXIF triage, and the damage-model interface. The integrity classifier is optional:

```bash
python -m pip install -e ".[ml]"
```

`[ml]` adds `torch` and `transformers`. The core package depends only on Pillow and ImageHash. Python 3.10 or newer is required.

## Quickstart

```python
from claim_photo_toolkit import inspect

report = inspect(["front.jpg", "side.jpg"], claim_date="2026-03-01")
for photo in report.photos:
    if photo.flagged:
        print(photo.path, ", ".join(photo.flags))
print(report.to_dict())
```

A string is one path. `claim_date` accepts a `date`, a `datetime`, or an ISO-8601 string. Photos taken after that date are flagged.

Without the `[ml]` extra, `inspect` still returns a report. Each photo's integrity section has `"status": "unavailable"` and the other checks run as usual. Pass `run_integrity=False` to skip the classifier on purpose.

Importing the package does not download the model. The first integrity check loads `obuladinnesai/claim-photo-integrity-detector-v5` and keeps it in memory for later calls. `reset_integrity_model()` drops that in-memory copy. Hugging Face's own disk cache is separate.

## CLI

```bash
claim-photo inspect front.jpg side.jpg --claim-date 2026-03-01 --json
```

`--json` prints the report to stdout. Without it, the command prints a short text summary. Other flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--claim-date` | none | ISO date or datetime. Later capture times are flagged. |
| `--integrity-threshold` | `0.5` | Flag when the `ai_generated` score is at least this value. |
| `--duplicate-distance` | `8` | Maximum perceptual-hash Hamming distance treated as the same photo. |
| `--skip-integrity` | off | Do not load the classifier. |
| `--json` | off | Print JSON instead of text. |

Exit code `0` means a report was produced, including when photos are flagged or unreadable. Exit code `2` means the arguments were rejected. `claim-photo --version` prints the toolkit version.

Example for one JPEG whose software tag is Photoshop and whose capture time is after the claim date (`--skip-integrity`):

```json
{
  "version": "0.1.0",
  "claim_date": "2026-03-01",
  "photo_count": 1,
  "flagged_count": 1,
  "error_count": 0,
  "duplicate_groups": [],
  "photos": [
    {
      "path": "front.jpg",
      "flagged": true,
      "flags": ["metadata:editing_software", "metadata:timestamp_after_claim"],
      "error": null,
      "integrity": {
        "status": "skipped",
        "flagged": false,
        "label": null,
        "scores": {},
        "threshold": null,
        "model_id": "obuladinnesai/claim-photo-integrity-detector-v5",
        "flags": [],
        "detail": "integrity check skipped"
      },
      "damage": {
        "status": "unavailable",
        "backend": "unavailable",
        "parts": [],
        "summary": null,
        "detail": "No damage-assessment backend is configured. Pass damage_backend= to inspect()."
      },
      "duplicates": {
        "status": "ok",
        "flagged": false,
        "algorithm": "phash",
        "hash": "8000000000000000",
        "distance_threshold": 8,
        "matches": [],
        "detail": null
      },
      "metadata": {
        "status": "flagged",
        "flagged": true,
        "timestamp": "2026-04-02T15:04:05",
        "gps": {"latitude": 37.775, "longitude": -122.419444},
        "device": "Apple iPhone 15",
        "software": "Adobe Photoshop 25.0",
        "image_format": "JPEG",
        "flags": ["editing_software", "timestamp_after_claim"],
        "detail": null
      }
    }
  ]
}
```

When a second file is the same photo, `duplicate_groups` contains `{ "paths": ["front.jpg", "front-copy.jpg"], "max_distance": 0 }` and each photo's `matches` lists the other path. When the integrity model runs, `scores` contains `real` and `ai_generated`, `label` is the higher score, and `flags` is `["ai_generated"]` when that score clears the threshold. GPS is `null` when the file has no coordinates.

## Checks

`inspect` always returns one `ClaimReport`. A photo is `flagged` when any suspicion flag below is set. Damage findings do not flag a photo. A file Pillow cannot open is counted in `error_count` and is not treated as suspicious.

### Integrity

Wraps the public ViT checkpoint at the root of [`obuladinnesai/claim-photo-integrity-detector-v5`](https://huggingface.co/obuladinnesai/claim-photo-integrity-detector-v5). The model card's usage for that checkpoint is:

```python
from transformers import pipeline

pipe = pipeline(
    "image-classification",
    model="obuladinnesai/claim-photo-integrity-detector-v5",
)
pipe("path/to/car_damage_photo.jpg")
```

The toolkit calls that pipeline, lazily, once per process. The head has two labels, `real` and `ai_generated`. A photo is flagged when `ai_generated >= integrity_threshold` (default `0.5`). The card does not publish an operating point for this checkpoint, so the threshold is an argument rather than a calibrated false-alarm rate.

### Damage

There is no damage model in this package. `inspect` uses `UnavailableDamageBackend` until you pass `damage_backend`. See [Plugging in a damage model](#plugging-in-a-damage-model). A configured backend returns part names, a severity (`none`, `minor`, `moderate`, `severe`), an optional confidence from 0 to 1, and an optional summary. Part names belong to the backend.

### Duplicates

Each readable photo gets a 64-bit perceptual hash (`imagehash.phash`, `hash_size=8`). Two photos match when the Hamming distance is at most `duplicate_distance` (default 8). Each photo lists its matches. `duplicate_groups` lists connected groups of two or more photos, in claim order, with the largest direct distance inside the group. Pass `hash_size` to `inspect` if you want a longer hash; the distance threshold is in bits and should be reconsidered when the hash size changes.

### Metadata

EXIF is read with Pillow: capture time (`DateTimeOriginal`, then `DateTimeDigitized`, then `DateTime`), GPS, camera make and model, and software.

| Flag | Meaning |
| --- | --- |
| `missing_exif` | The file has no EXIF metadata. |
| `stripped_metadata` | JPEG, TIFF, or MPO with neither a capture time nor a camera make/model. |
| `editing_software` | The software tag names a photo editor or image generator (for example Photoshop, GIMP, Snapseed, Stable Diffusion). |
| `timestamp_after_claim` | The capture time is later than `claim_date`. A date compares calendar days. A datetime compares the clock time as recorded, with timezone offsets ignored. |
| `unreadable_timestamp` | A timestamp tag is present but not in EXIF's `YYYY:MM:DD HH:MM:SS` form. |
| `invalid_gps` | GPS tags are present but not a usable WGS84 point. |

Photo-level flags prefix these codes with `metadata:`. A clean phone JPEG with a capture time and device is not flagged. A PNG with no EXIF is `missing_exif` only; `stripped_metadata` is reserved for formats that normally carry camera metadata.

## Design and architecture

```
inspect(paths, claim_date=...)
├── integrity.py    optional ViT pipeline, loaded once and cached
├── damage.py       DamageBackend protocol; unavailable until you plug one in
├── duplicates.py   phash, Hamming distance, connected groups
└── metadata.py     Pillow EXIF to review flags
        │
        ▼
ClaimReport.to_dict()     claim-photo inspect --json
```

The library layout is a `src` package, `claim_photo_toolkit`. Result records are frozen dataclasses. `inspect` is the only call a claim needs. Checks are independent except duplicates, which compare the whole set. An unreadable file, a backend exception, or a bad model response is stored on that photo and the rest of the claim continues.

`transformers` is imported inside the loader, so installing the core package does not import PyTorch. The loader uses a lock and builds the pipeline at most once. Tests inject a callable with the same label/score shape and never download the checkpoint.

Damage stays behind a protocol on purpose. Shipping a pretend part-and-severity model would report findings the toolkit cannot support. The CLI has no damage backend; from the command line the damage section is `unavailable`.

## Plugging in a damage model

Implement `name` and `assess`. `assess` receives an RGB `PIL.Image` and the photo path, and returns a `DamageAssessment`.

```python
from pathlib import Path

from PIL import Image

from claim_photo_toolkit import DamageAssessment, PartFinding, inspect


class PrivateDamageModel:
    name = "private-damage-model"

    def assess(self, image: Image.Image, path: Path) -> DamageAssessment:
        prediction = your_model.predict(image)
        return DamageAssessment(
            parts=(
                PartFinding(
                    part=prediction.part,
                    severity=prediction.severity,
                    confidence=prediction.confidence,
                ),
            ),
            summary=prediction.summary,
        )


report = inspect(
    ["front.jpg"],
    claim_date="2026-03-01",
    damage_backend=PrivateDamageModel(),
)
```

`severity` must be `none`, `minor`, `moderate`, or `severe`. `confidence`, when set, must be between 0 and 1. `parts` is a tuple of `PartFinding`. If `assess` raises `DamageModelUnavailable`, the photo's damage status is `unavailable`. Any other exception is stored as a damage error and the claim continues. `UnavailableDamageBackend` is the default and raises `DamageModelUnavailable` instead of inventing findings.

## Limitations

- Flags are review hints, not proof of editing, reuse, fraud, or authenticity.
- The wrapped checkpoint is the v5 ViT classifier (`real` vs `ai_generated`) documented on the model card. That card reports the checkpoint near chance on localized inpainting and marks it for research use. The same Hugging Face repo also hosts a later localization model that does not load with `transformers.pipeline`. This package does not wrap that model.
- The default integrity threshold is 0.5 because the card does not publish a false-alarm threshold for the v5 classifier.
- The first integrity run downloads the checkpoint from the Hugging Face Hub (the root `model.safetensors` is about 343 MB) and needs network access unless the Hub cache already has it.
- Accuracy depends on the generators and edits the checkpoint was trained on. Newer ones can look different.
- EXIF is easy to forge, delete, or rewrite. Messaging apps often strip it, and screenshots never had it, so `missing_exif` will fire on ordinary files.
- Timestamps are compared as stored. A date-only claim date flags a photo only when its capture calendar day is later.
- A perceptual hash misses heavy crops, overlays, and some recompression, and unrelated images can land within the distance threshold.
- No damage model ships here. Empty or unavailable damage means no backend ran, not that the car is undamaged.
- Files Pillow cannot decode (for example HEIC without an extra codec) come back as read errors. Pillow's decompression-bomb limit still applies to very large images.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
```

Those are the same commands GitHub Actions runs on Python 3.10 and 3.12. The CI install does not include `[ml]`.

## License

MIT. See [LICENSE](LICENSE).
