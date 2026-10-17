"""Check that v9/*.py are byte-identical to the files at the pinned Hugging Face revision."""
import hashlib, os, sys
from huggingface_hub import hf_hub_download

REPO_ID = "obuladinnesai/car-damage-photo-integrity-v9"
REVISION = "4b7b6ea59cc125119794b2f5a49efb966ec3e5ff"  # keep in sync with inference.py

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v9")
digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
ok = True
for name in ("inference.py", "model_v9.py"):
    same = digest(os.path.join(root, name)) == digest(hf_hub_download(REPO_ID, name, revision=REVISION))
    print(f"{'OK  ' if same else 'DIFF'} v9/{name} vs {REPO_ID}@{REVISION[:10]}")
    ok &= same
sys.exit(0 if ok else 1)
