"""Export every poster section as its own readable PNG for close review.

Crops the high-resolution poster render (compiled/final_clean_highres.jpg) into
named section images under poster/section_review/. Each crop is downscaled so
text stays readable when inspected one section at a time.

Usage:
    .venv\\Scripts\\python.exe poster\\crop_sections.py
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
SRC = ROOT / "compiled" / "final_clean_highres.jpg"
OUT = ROOT / "section_review"
PAPER = (251, 248, 241)
MAXDIM = 1700

# (x0, y0, x1, y1) as fractions of the full poster.
SECTIONS = {
    "01_header": (0.00, 0.000, 1.00, 0.105),
    "02_message_band": (0.00, 0.100, 1.00, 0.220),
    "03a_benchmark_design": (0.00, 0.215, 0.435, 0.450),
    "03b_task_regime_map": (0.420, 0.215, 1.00, 0.450),
    "04_full_benchmark": (0.00, 0.428, 1.00, 0.612),
    "05a_ld50_leakage": (0.00, 0.595, 0.355, 0.805),
    "05b_geometry_check": (0.330, 0.595, 0.675, 0.805),
    "05c_tox21_transfer": (0.655, 0.595, 1.00, 0.805),
    "06_conclusions": (0.00, 0.795, 1.00, 0.910),
    "07_footer": (0.00, 0.905, 1.00, 1.000),
}


def main():
    if not SRC.exists():
        raise SystemExit(f"Render not found: {SRC}. Run render_poster.js first.")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    im = Image.open(SRC).convert("RGBA")
    W, H = im.size
    for name, (x0, y0, x1, y1) in SECTIONS.items():
        crop = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        flat = Image.new("RGBA", crop.size, PAPER + (255,))
        flat.alpha_composite(crop)
        flat = flat.convert("RGB")
        if max(flat.size) > MAXDIM:
            scale = MAXDIM / max(flat.size)
            flat = flat.resize((int(flat.size[0] * scale), int(flat.size[1] * scale)))
        flat.save(OUT / f"{name}.png")
        print(name, flat.size)


if __name__ == "__main__":
    main()
