"""Score car photos with the Car Damage Photo Integrity v9 model and save tamper heatmaps.

    python inference.py --image examples/car.jpg
    python inference.py --image examples/car.jpg --heatmap out/heat.png --overlay out/overlay.png
    python inference.py --image a.jpg b.jpg --condition jpeg_q90 --device cpu

The weights and config are downloaded from the Hugging Face Hub at a pinned revision (no token
needed). The model code in v9/ is an unmodified copy of the files at that same revision; run
scripts/verify_vendored.py to check. Prints one JSON line per image.

Output is a triage signal for human review, not proof of editing, fraud or authenticity.
"""
import argparse, importlib.util, json, os, sys
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# v9/inference.py from the pinned Hub revision, loaded under its own name so it cannot clash
# with this file (it adds v9/ to sys.path itself to import model_v9).
_spec = importlib.util.spec_from_file_location(
    "v9_inference", os.path.join(os.path.dirname(os.path.abspath(__file__)), "v9", "inference.py"))
v9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v9)

REPO_ID = "obuladinnesai/car-damage-photo-integrity-v9"
REVISION = "4b7b6ea59cc125119794b2f5a49efb966ec3e5ff"
# Without resizing, peak CPU memory grows with resolution (about 14 GB for a 12 MP photo on
# jpeg_q90/native in our test), so warn above this many pixels.
LARGE_IMAGE_PIXELS = 4_000_000


def overlay(im, heat, alpha=0.6):
    """Blend the [0, 1] heatmap over the preprocessed photo in red."""
    base = np.asarray(im.convert("RGB"), dtype=np.float32)
    a = (np.clip(heat, 0, 1) * alpha)[..., None]
    red = np.array([255, 0, 0], dtype=np.float32)
    return Image.fromarray((base * (1 - a) + red * a).astype(np.uint8))


def _save(img, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", nargs="+", required=True, help="one or more JPEG/PNG files")
    ap.add_argument("--condition", default="upload", choices=["upload", "jpeg_q90", "native"],
                    help="preprocessing + matching threshold (default: upload = max side 1600 px, JPEG q85)")
    ap.add_argument("--device", help="cpu, cuda or mps (default: best available)")
    ap.add_argument("--heatmap", help="save the grayscale heatmap PNG here (single image only)")
    ap.add_argument("--overlay", help="save the heatmap blended over the photo here (single image only)")
    ap.add_argument("--weights", help="local model.safetensors (skips the Hub download)")
    ap.add_argument("--config", help="local config.json (skips the Hub download)")
    a = ap.parse_args()
    if (a.heatmap or a.overlay) and len(a.image) > 1:
        ap.error("--heatmap/--overlay need exactly one --image")

    weights = a.weights or hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION)
    config = a.config or hf_hub_download(REPO_ID, "config.json", revision=REVISION)
    net, cfg, dev = v9.load_model(weights, config, a.device)

    for path in a.image:
        im = Image.open(path)
        if a.condition != "upload" and im.width * im.height > LARGE_IMAGE_PIXELS:
            print(f"warning: {path} is {im.width}x{im.height}; '{a.condition}' scores it at full size, "
                  "which can need >10 GB RAM. Use --condition upload for phone photos.", file=sys.stderr)
        r = v9.score_image(net, cfg, dev, im, a.condition)
        print(json.dumps({"image": os.path.basename(path), "score": round(r["score"], 4),
                          "threshold": round(r["threshold"], 4), "flagged": r["flagged"],
                          "condition": a.condition, "max_heat": round(float(r["heatmap"].max()), 4),
                          "revision": REVISION[:10]}))
        if a.heatmap:
            _save(Image.fromarray((r["heatmap"] * 255).astype(np.uint8)), a.heatmap)
        if a.overlay:
            _save(overlay(v9.preprocess(Image.open(path), a.condition), r["heatmap"]), a.overlay)


if __name__ == "__main__":
    main()
