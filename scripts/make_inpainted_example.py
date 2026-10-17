"""Recreate examples/car_inpainted.png: remove the crumpled front panel from examples/car.jpg with LaMa.

    pip install simple-lama-inpainting==0.1.0   # separate environment recommended; downloads big-lama.pt (~200 MB)
    python scripts/make_inpainted_example.py examples/car.jpg examples/car_inpainted.png examples/car_inpainted_mask.png

Save the result losslessly (PNG). See examples/README.md for why the file format matters.
"""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFilter
import functools, torch
# big-lama.pt was scripted on CUDA; map it to CPU so it loads on machines without NVIDIA GPUs.
torch.jit.load = functools.partial(torch.jit.load, map_location="cpu")
from simple_lama_inpainting import SimpleLama
src, out_img, out_mask = sys.argv[1:4]
im = Image.open(src).convert("RGB"); W, H = im.size
# Remove the crumpled front wing/hood panel that sticks out left of the A-pillar.
mask = Image.new("L", (W, H), 0)
ImageDraw.Draw(mask).polygon([(250, 470), (470, 455), (480, 600), (430, 730), (255, 720)], fill=255)
mask = mask.filter(ImageFilter.MaxFilter(9))
res = SimpleLama(device=torch.device("cpu"))(im, mask)
res = res.crop((0, 0, W, H))  # LaMa pads to a multiple of 8
res.save(out_img, quality=92); mask.save(out_mask)
print("mask area %.2f%%" % (100 * (np.asarray(mask) > 127).mean()), res.size)
