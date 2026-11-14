"""Figure: keyhole specimen geometry and the finite element analysis of M. Groper (images from the FE model)."""
import os as _os
ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
for _d in ("tex/figs", "figs", "results", "data"): _os.makedirs(_os.path.join(ROOT, _d), exist_ok=True)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "pdf.fonttype": 42})
D = ROOT + "/realdata/fea/"

def load(f, box=None):
    im = Image.open(D + f).convert("RGB")
    if box: im = im.crop(box)
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bb = ImageChops.difference(im, bg).convert("L").point(lambda v: 255 if v > 18 else 0).getbbox()
    return np.asarray(im.crop(bb) if bb else im)

panels = [("drawing.png", (0, 0, 472, 352), "(a) keyhole specimen (mm)"),
          ("bc.jpg", None, "(b) half model, pin loads"),
          ("mesh2.PNG", (0, 120, 1366, 1008), "(c) mesh of the half model"),
          ("mesh1.PNG", (0, 120, 1366, 1008), "(d) notch mesh, 0.1 mm elements"),
          ("stress_full.jpg", None, r"(e) $\sigma_\theta$ (MPa) at 8.895 kN"),
          ("stress_zoom.jpg", None, r"(f) notch root, max 312.6 MPa")]
fig, axs = plt.subplots(2, 3, figsize=(7.1, 3.9), gridspec_kw=dict(wspace=0.04, hspace=0.18))
for ax, (f, box, t) in zip(axs.ravel(), panels):
    ax.imshow(load(f, box)); ax.set_axis_off(); ax.set_title(t, fontsize=7)
fig.savefig(ROOT + "/tex/figs/fig_fea.pdf", bbox_inches="tight", dpi=400)
fig.savefig(ROOT + "/figs/fig_fea.png", bbox_inches="tight", dpi=300)
