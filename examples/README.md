# Examples

| File | What it is |
|---|---|
| `car.jpg` | **Unedited** photo of a car after a collision with a tree, downscaled to 1600×1067 |
| `car_expected.json`, `car_overlay.jpg` | Output and heatmap overlay for `car.jpg`. Not flagged; the overlay is effectively blank |
| `car_inpainted.png` | ⚠️ **Deliberately AI-edited** copy of `car.jpg`, made for testing. The crumpled front panel left of the windscreen was removed with LaMa inpainting (3.5% of the image) |
| `car_inpainted_mask.png` | The exact edited region (white). Use it as ground truth |
| `car_inpainted_expected.json`, `car_inpainted_overlay.jpg` | Output and overlay for the edited copy. Flagged; the red area matches the mask |

`car_inpainted.png` is a synthetic test image. It does not show a real vehicle's condition and must
not be presented as a genuine photo. It was made with
[`scripts/make_inpainted_example.py`](../scripts/make_inpainted_example.py)
(simple-lama-inpainting 0.1.0, `big-lama.pt`, CPU).

## Source and licence

`car.jpg` is derived from [Car collision with tree.jpg](https://commons.wikimedia.org/wiki/File:Car_collision_with_tree.jpg)
by Rico Loeb (Canon EOS 600D, 2013-06-28), released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) on Wikimedia Commons. It was resized
from 5184×3456 and re-saved as JPEG quality 92. Original SHA-1:
`53607911ee986c7287085d1e5f0d1aab035ad1ad`. `car_inpainted.png` is a modified version of `car.jpg`.

## Results

Pinned revision `4b7b6ea`, CPU. Scores are the same on Apple MPS.

| Image | Condition | Score | Threshold | Flagged | Heatmap vs mask (IoU at 0.5) |
|---|---|---:|---:|---|---:|
| `car.jpg` | `upload` (default) | -5.19 | 4.19 | no | n/a |
| `car.jpg` | `jpeg_q90` | -2.35 | 4.19 | no | n/a |
| `car_inpainted.png` | `upload` (default) | 8.31 | 4.19 | **yes** | 0.90 (precision 0.96, recall 0.93) |
| `car_inpainted.png` | `jpeg_q90` | 7.78 | 4.19 | **yes** | |

## Known failure: re-saving the edit as JPEG

If the edited image is saved as JPEG quality 92 before scoring, the `upload` condition misses it
completely (score -5.73, blank heatmap), while `jpeg_q90` still catches it (6.92, IoU 0.84).
`upload` re-encodes at JPEG q85. Re-encoding that JPEG at different qualities and scoring with
`--condition native` gives:

| Re-encode quality | 80 | 85 | 87 | 88 | 89 | 90 | 92 | 95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Edited score | 6.93 | **-5.73** | **-3.13** | 1.80 | 5.17 | 6.92 | 7.78 | 5.96 |
| Unedited `car.jpg` score | -1.18 | -5.19 | -4.82 | -4.23 | -3.18 | -2.35 | -2.26 | -3.63 |

So detection of this edit depends on the exact compression history. Reproduce it with:

```bash
python -c "from PIL import Image; Image.open('examples/car_inpainted.png').save('edited_q92.jpg', quality=92)"
python inference.py --image edited_q92.jpg --condition upload     # missed
python inference.py --image edited_q92.jpg --condition jpeg_q90   # flagged
```

This is one image, not a benchmark. Please report similar cases in the
[model discussion](https://huggingface.co/obuladinnesai/car-damage-photo-integrity-v9/discussions/1).

Only add images you are allowed to redistribute. Do not add claim, customer or employer photos.
