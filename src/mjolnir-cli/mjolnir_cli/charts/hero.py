"""The README hero image (``assets/hero.png``): the committed background
(``assets/hero-background.png``) with the slogan and the gradient pills drawn
under its MJOLNIR logotype.
"""

from __future__ import annotations

from pathlib import Path

from mjolnir_cli.charts._common import FONT_DIR, display_family
from mjolnir_cli.config import RepoLayout


HERO = "hero.png"
HERO_BACKGROUND = "hero-background.png"
HERO_SLOGAN = ("Jetson AGX Thor vLLM serving CLI", "with dedicated paged fp8 decode-attention kernel")
# (label, gradient left, gradient right): NVIDIA green, Mjolnir orange, vLLM blue
HERO_PILLS = (
    ("Jetson AGX Thor", "#8cc63f", "#3f8a12"),
    ("CLI", "#f2925a", "#d4501f"),
    ("vLLM {vllm}", "#5aa6ff", "#2a5fd6"),
)

_BUNDLED_WEIGHTS = {"normal": "Regular", "medium": "Medium", "semibold": "SemiBold", "bold": "Bold"}


def _font_file(family: str, weight: str) -> str:
    bundled = FONT_DIR / f"InterDisplay-{_BUNDLED_WEIGHTS.get(weight, 'Regular')}.ttf"
    if family == "Inter Display" and bundled.exists():
        return str(bundled)
    from matplotlib import font_manager

    return font_manager.findfont(font_manager.FontProperties(family=family, weight=weight))


def logotype_box(img, x_from: float = 0.62) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the white logotype on the dark right
    side of the background: near-white pixels right of ``x_from`` of the
    width (the lightning lives on the left)."""
    import numpy as np

    a = np.asarray(img.convert("RGB")).astype(int)
    x0 = int(a.shape[1] * x_from)
    white = a[:, x0:].min(axis=2) > 225
    ys, xs = np.nonzero(white)
    if len(xs) < 50:  # no logotype found: the right third's centre, upper half
        w, h = img.size
        return int(w * 0.70), int(h * 0.15), int(w * 0.86), int(h * 0.62)
    return int(xs.min() + x0), int(ys.min()), int(xs.max() + x0), int(ys.max())


def render_hero(layout: RepoLayout) -> Path | None:
    """``assets/hero.png`` — the README header: ``assets/hero-background.png``
    with the slogan (two lines, centred under the MJOLNIR logotype) and a row
    of gradient pills (Jetson AGX Thor · CLI · vLLM <version>) under it."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    from mjolnir_cli.config import VLLM_VERSION

    if layout.repo_root is None:
        return None
    src = layout.repo_root / "assets" / HERO_BACKGROUND
    if not src.exists():
        return None
    bg = Image.open(src).convert("RGBA")
    W, H = bg.size
    left, _top, right, bottom = logotype_box(bg)
    cx = (left + right) / 2
    S = 2  # draw at 2x, downsample: smooth pill edges and text
    fam = display_family()
    slogan_font = ImageFont.truetype(_font_file(fam, "medium"), int(round(H * 0.040)) * S)
    pill_font = ImageFont.truetype(_font_file(fam, "semibold"), int(round(H * 0.030)) * S)

    over = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d, ds = ImageDraw.Draw(over), ImageDraw.Draw(shadow)

    # slogan: two centred lines under the logotype
    y = (bottom + H * 0.055) * S
    line_h = slogan_font.size * 1.38
    for line in HERO_SLOGAN:
        w = d.textlength(line, font=slogan_font)
        x = cx * S - w / 2
        ds.text((x, y + 2 * S), line, font=slogan_font, fill=(0, 0, 0, 170))
        d.text((x, y), line, font=slogan_font, fill=(232, 238, 248, 255))
        y += line_h

    # pills: one centred row
    labels = [(t.format(vllm=VLLM_VERSION), c0, c1) for t, c0, c1 in HERO_PILLS]
    pad_x, pill_h, gap = 18 * S, int(round(H * 0.060)) * S, 14 * S
    widths = [d.textlength(t, font=pill_font) + 2 * pad_x for t, _, _ in labels]
    x = cx * S - (sum(widths) + gap * (len(widths) - 1)) / 2
    y = int(y + H * 0.035 * S)
    for (text, c0, c1), w in zip(labels, widths):
        box = (int(x), y, int(x + w), y + pill_h)
        ds.rounded_rectangle((box[0], box[1] + 3 * S, box[2], box[3] + 3 * S), radius=pill_h // 2, fill=(0, 0, 0, 120))
        # horizontal gradient, clipped to the pill by a rounded mask
        a0, a1 = Image.new("RGBA", (1, 1), c0).getpixel((0, 0)), Image.new("RGBA", (1, 1), c1).getpixel((0, 0))
        grad = Image.new("RGBA", (box[2] - box[0], pill_h))
        gw = max(1, grad.size[0] - 1)
        grad.putdata(
            [
                tuple(int(a0[k] + (a1[k] - a0[k]) * (i / gw)) for k in range(4))
                for _ in range(pill_h)
                for i in range(grad.size[0])
            ]
        )
        mask = Image.new("L", grad.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, grad.size[0] - 1, pill_h - 1), radius=pill_h // 2, fill=255)
        over.paste(grad, box[:2], mask)
        # a thin light rim on top
        d.rounded_rectangle(box, radius=pill_h // 2, outline=(255, 255, 255, 70), width=S)
        tw = d.textlength(text, font=pill_font)
        ty = y + (pill_h - pill_font.size) / 2 - pill_font.size * 0.08
        d.text((x + (w - tw) / 2, ty + S), text, font=pill_font, fill=(0, 0, 0, 90))
        d.text((x + (w - tw) / 2, ty), text, font=pill_font, fill=(255, 255, 255, 255))
        x += w + gap

    shadow = shadow.filter(ImageFilter.GaussianBlur(4 * S))
    layer = Image.alpha_composite(shadow, over).resize((W, H), Image.LANCZOS)
    out = layout.repo_root / "assets" / HERO
    Image.alpha_composite(bg, layer).convert("RGB").save(out, optimize=True)
    return out
