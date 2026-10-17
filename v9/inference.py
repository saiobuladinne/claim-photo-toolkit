"""Minimal inference for the released v9 car-damage photo-integrity model.

Loads model.safetensors + config.json (next to this file if present, else from the Hub) and
returns an image-level score plus a per-pixel edit heatmap.

    python inference.py photo.jpg                       # local weights, else downloads from the Hub
    python inference.py photo.jpg --weights model.safetensors --config config.json
    python inference.py photo.jpg --condition upload --heatmap heat.png

Score = mean of the top-256 stride-2 logits of the edit map (higher = more likely edited).
`flagged` compares the score with the 2%-false-alarm threshold calibrated on the 600 real
eval photos for the chosen --condition. The image is preprocessed the same way as in that
calibration, so pick the condition that matches how you want to compare:
  jpeg_q90 (default): re-save as JPEG quality 90, then score (headline 85.3% catch)
  upload:             resize to max side 1600, re-save as JPEG q85, then score (81.6% catch)
  native:             score the file as-is (not recommended for comparisons: in the eval set,
                      edited and real photos had different file formats, so native numbers are
                      likely inflated)
Triage signal for human review only; not proof of editing or fraud.
"""
import argparse, io, json, os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v9 import LocNet, img_score

REPO_ID = "obuladinnesai/car-damage-photo-integrity-v9"

def _hub(fname):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO_ID, fname)

def _local_or_hub(fname):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
    return p if os.path.exists(p) else _hub(fname)

def pick_device(name=None):
    if name: return torch.device(name)
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_model(weights=None, config=None, device=None):
    from safetensors.torch import load_file
    weights = weights or _local_or_hub("model.safetensors")
    config = config or _local_or_hub("config.json")
    cfg = json.load(open(config))
    net = LocNet(pretrained=False)
    net.load_state_dict(load_file(weights), strict=True)
    dev = pick_device(device)
    return net.to(dev).eval(), cfg, dev

def preprocess(im, condition="jpeg_q90"):
    im = im.convert("RGB")
    if condition == "upload":
        W, H = im.size; s = min(1.0, 1600 / max(W, H))
        if s < 1: im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
    if condition in ("jpeg_q90", "upload"):
        b = io.BytesIO(); im.save(b, "JPEG", quality=90 if condition == "jpeg_q90" else 85); b.seek(0)
        im = Image.open(b).convert("RGB")
    elif condition != "native":
        raise ValueError(f"unknown condition {condition!r}")
    return im

@torch.no_grad()
def score_image(net, cfg, dev, image, condition="jpeg_q90"):
    """image: path or PIL.Image. Returns dict(score, threshold, flagged, heatmap[H,W] in 0..1)."""
    im = Image.open(image) if isinstance(image, (str, os.PathLike)) else image
    im = preprocess(im, condition)
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)[None].float().div(255)
    h, w = x.shape[-2:]
    x = F.pad(x, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32), mode="reflect").to(dev)
    lg = net(x)[..., :(h + 1) // 2, :(w + 1) // 2]          # logits at stride 2
    score = float(img_score(lg, k=cfg["scoring"]["top_k"]))
    heat = torch.sigmoid(F.interpolate(lg.float(), size=(h, w), mode="bilinear", align_corners=False))[0, 0].cpu().numpy()
    thr = cfg["thresholds"][condition]["fpr_2pct"]
    return dict(score=score, threshold=thr, flagged=bool(score >= thr), condition=condition, heatmap=heat)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--weights"); ap.add_argument("--config"); ap.add_argument("--device")
    ap.add_argument("--condition", default="jpeg_q90", choices=["jpeg_q90", "upload", "native"])
    ap.add_argument("--heatmap", help="save the edit heatmap PNG here (only for a single image)")
    a = ap.parse_args()
    net, cfg, dev = load_model(a.weights, a.config, a.device)
    for p in a.images:
        r = score_image(net, cfg, dev, p, a.condition)
        print(json.dumps({"image": os.path.basename(p), "score": round(r["score"], 4), "threshold": round(r["threshold"], 4),
                          "flagged": r["flagged"], "condition": a.condition, "max_heat": round(float(r["heatmap"].max()), 4)}))
        if a.heatmap and len(a.images) == 1:
            Image.fromarray((r["heatmap"] * 255).astype(np.uint8)).save(a.heatmap)

if __name__ == "__main__":
    main()
