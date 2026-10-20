"""
rs_core.py - all image-processing logic for the Remote Sensing demo.

Only NumPy is needed here (no Streamlit, no Matplotlib), so every function
can be tested on its own with:  python self_test.py

A "scene" is a plain Python dictionary:
    scene["dn"]    -> {"blue": 2-D uint16 array, "green": ..., "red": ..., "nir": ...}
    scene["refl"]  -> same keys, float32 surface reflectance (0..1), NaN = no data
    scene["meta"]  -> dictionary of text/number facts about the data (source, date, ...)
    scene["samples"] -> optional {name: (row, col)} example pixels (synthetic scene only)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Band information (Sentinel-2 MSI, the sensor used for the real data)
# --------------------------------------------------------------------------
BANDS = ("blue", "green", "red", "nir")

BAND_TABLE = {
    "blue":  {"s2": "B02", "nm": 490, "label": "Blue"},
    "green": {"s2": "B03", "nm": 560, "label": "Green"},
    "red":   {"s2": "B04", "nm": 665, "label": "Red"},
    "nir":   {"s2": "B08", "nm": 842, "label": "Near-infrared (NIR)"},
}

# Sentinel-2 Level-2A (processing baseline >= 04.00): reflectance = DN * 0.0001 - 0.1
DEFAULT_SCALE = 0.0001
DEFAULT_OFFSET = -0.1


# --------------------------------------------------------------------------
# Loading / saving scenes
# --------------------------------------------------------------------------
def build_scene(dn: dict, meta: dict, samples: dict | None = None) -> dict:
    """Convert raw digital numbers (DN) into reflectance and bundle everything."""
    scale = float(meta.get("scale", DEFAULT_SCALE))
    offset = float(meta.get("offset", DEFAULT_OFFSET))
    refl = {}
    for b in BANDS:
        arr = dn[b].astype(np.float32) * np.float32(scale) + np.float32(offset)
        arr = np.clip(arr, 0.0, None)          # tiny negative values -> 0
        arr[dn[b] == 0] = np.nan               # DN 0 means "no data"
        refl[b] = arr.astype(np.float32)
    return {"dn": dn, "refl": refl, "meta": meta, "samples": samples}


def save_scene(path, dn: dict, meta: dict) -> None:
    """Save four DN bands + metadata into one compressed .npz file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{b: dn[b] for b in BANDS}, meta=np.array(json.dumps(meta)))


def load_scene(path) -> dict:
    """Load a scene saved by save_scene()."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z["meta"].item())
        dn = {b: z[b] for b in BANDS}
    return build_scene(dn, meta)


# --------------------------------------------------------------------------
# Synthetic scene (NOT real satellite data - for controlled demonstrations)
# --------------------------------------------------------------------------
WATER, SOIL, BUILT, HEALTHY, STRESSED = 0, 1, 2, 3, 4

# Typical surface-reflectance values (blue, green, red, nir) chosen by the author
# from the general shape of published spectral signatures. They are illustrative.
SIGNATURES = {
    WATER:    (0.045, 0.055, 0.035, 0.020),
    SOIL:     (0.100, 0.140, 0.190, 0.260),
    BUILT:    (0.140, 0.160, 0.180, 0.210),
    HEALTHY:  (0.030, 0.070, 0.040, 0.420),
    STRESSED: (0.050, 0.090, 0.080, 0.220),
}


def _smooth(a: np.ndarray, passes: int = 4) -> np.ndarray:
    """Cheap blur: repeatedly average a pixel with its 4 neighbours."""
    for _ in range(passes):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a


def make_synthetic_scene(size: int = 300, seed: int = 42) -> dict:
    """Build a 4-band fake landscape: lake, river, bare soil, town, 6 crop fields."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cls = np.full((size, size), SOIL, dtype=np.uint8)

    # six crop fields (3 rows x 2 columns) in the right part of the image
    field_pattern = [[HEALTHY, STRESSED], [STRESSED, HEALTHY], [HEALTHY, SOIL]]
    x0, x1 = int(0.55 * size), int(0.95 * size)
    y0, y1 = int(0.08 * size), int(0.92 * size)
    fh, fw = (y1 - y0) // 3, (x1 - x0) // 2
    for i in range(3):
        for j in range(2):
            ys, xs = y0 + i * fh, x0 + j * fw
            cls[ys + 3: ys + fh - 3, xs + 3: xs + fw - 3] = field_pattern[i][j]

    # built-up block (town), lower left
    cls[int(0.62 * size): int(0.92 * size), int(0.05 * size): int(0.28 * size)] = BUILT
    # river (curvy vertical band) and a round lake (drawn last so they stay on top)
    river_c = 0.38 * size + 0.04 * size * np.sin(yy / size * 2 * np.pi)
    cls[np.abs(xx - river_c) < 0.025 * size] = WATER
    cls[(yy - 0.20 * size) ** 2 + (xx - 0.15 * size) ** 2 < (0.09 * size) ** 2] = WATER

    # reflectance from class signatures, then add brightness texture + sensor noise
    texture = 1.0 + 0.10 * _smooth(rng.normal(size=(size, size)), 6) * 6
    refl = {}
    for k, b in enumerate(BANDS):
        base = np.zeros((size, size), dtype=np.float32)
        for c, sig in SIGNATURES.items():
            base[cls == c] = sig[k]
        noise = 1.0 + 0.03 * rng.normal(size=(size, size))
        refl[b] = np.clip(base * texture * noise, 0.001, 1.0).astype(np.float32)

    dn = {b: np.round((refl[b] - DEFAULT_OFFSET) / DEFAULT_SCALE).astype(np.uint16) for b in BANDS}

    c = lambda r, col: (int(r * size), int(col * size))
    samples = {
        "Water (lake)": c(0.20, 0.15),
        "Bare soil": c(0.45, 0.22),
        "Built-up (town)": c(0.77, 0.165),
        "Healthy vegetation (field)": (y0 + fh // 2, x0 + fw // 2),
        "Stressed / sparse vegetation (field)": (y0 + fh // 2, x0 + fw + fw // 2),
    }
    meta = {
        "is_real": False,
        "title": "SYNTHETIC demonstration scene (computer-generated, NOT satellite data)",
        "source": "Generated by make_synthetic_scene() in rs_core.py (random seed %d)" % seed,
        "license": "Created for this project - free to use",
        "sensor": "none (simulated 4-band image using Sentinel-2-like band positions)",
        "datetime": "n/a", "cloud_cover": "n/a", "pixel_size_m": 10,
        "scale": DEFAULT_SCALE, "offset": DEFAULT_OFFSET,
        "note": "Reflectance values per land-cover class are illustrative, plus random noise.",
    }
    return build_scene(dn, meta, samples)


# --------------------------------------------------------------------------
# Contrast enhancement (each function maps one band to display values 0..1)
# --------------------------------------------------------------------------
def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _rescale(band: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Linear map lo->0, hi->1, clip, and turn NaN (no data) into 0 (black)."""
    if not hi > lo:
        return np.zeros(band.shape, dtype=np.float32)
    out = (band - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


def enhance_none(band):
    """No enhancement: show reflectance directly on a fixed 0..1 scale (looks dark)."""
    return np.clip(np.nan_to_num(band, nan=0.0), 0.0, 1.0).astype(np.float32)


def enhance_minmax(band):
    """Linear stretch from the band's minimum to its maximum."""
    v = _finite(band)
    return _rescale(band, float(v.min()), float(v.max()))


def enhance_percentile(band, low=2.0, high=98.0):
    """Linear stretch that ignores the darkest `low`% and brightest `100-high`% of pixels."""
    lo, hi = np.percentile(_finite(band), [low, high])
    return _rescale(band, float(lo), float(hi))


def enhance_gamma(band, gamma=0.6):
    """Scale 0..(99th percentile) to 0..1, then apply out = in ** gamma.
    gamma < 1 brightens dark areas, gamma > 1 darkens them."""
    p99 = float(np.percentile(_finite(band), 99))
    return (_rescale(band, 0.0, p99) ** gamma).astype(np.float32)


def enhance_equalize(band, nbins=256):
    """Histogram equalisation: use the cumulative histogram (CDF) as the mapping."""
    v = _finite(band)
    lo, hi = float(v.min()), float(v.max())
    if not hi > lo:
        return np.zeros(band.shape, dtype=np.float32)
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    cdf = hist.cumsum().astype(np.float64)
    span = cdf[-1] - cdf[0]
    cdf = (cdf - cdf[0]) / span if span > 0 else np.zeros_like(cdf)
    centers = (edges[:-1] + edges[1:]) / 2.0
    out = np.interp(band, centers, cdf)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


ENHANCEMENTS = [
    "None (raw reflectance 0-1)",
    "Linear stretch (min-max)",
    "Percentile stretch",
    "Gamma correction",
    "Histogram equalisation",
]


def apply_enhancement(band, method, low=2.0, high=98.0, gamma=0.6):
    if method == ENHANCEMENTS[0]:
        return enhance_none(band)
    if method == ENHANCEMENTS[1]:
        return enhance_minmax(band)
    if method == ENHANCEMENTS[2]:
        return enhance_percentile(band, low, high)
    if method == ENHANCEMENTS[3]:
        return enhance_gamma(band, gamma)
    if method == ENHANCEMENTS[4]:
        return enhance_equalize(band)
    raise ValueError("Unknown enhancement: %s" % method)


# --------------------------------------------------------------------------
# Colour composites
# --------------------------------------------------------------------------
COMPOSITES = {
    "True colour (R, G, B)": ("red", "green", "blue"),
    "False colour (NIR, R, G)": ("nir", "red", "green"),
}


def raw_channels(scene, kind):
    """The three reflectance bands that feed the red, green, blue display channels."""
    return [scene["refl"][b] for b in COMPOSITES[kind]]


def make_composite(channels, method, low=2.0, high=98.0, gamma=0.6):
    """Enhance each channel separately and stack them into an (H, W, 3) image."""
    enhanced = [apply_enhancement(ch, method, low, high, gamma) for ch in channels]
    return np.dstack(enhanced), enhanced


# --------------------------------------------------------------------------
# NDVI
# --------------------------------------------------------------------------
def compute_ndvi(nir, red):
    """NDVI = (NIR - Red) / (NIR + Red), computed per pixel. NaN where undefined."""
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    num, den = nir - red, nir + red
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=np.isfinite(den) & (den != 0))
    return out


NDVI_CLASS_NAMES = [
    "Water / very dark (NDVI < 0)",
    "Bare soil / built-up (0 to 0.2)",
    "Sparse or stressed vegetation (0.2 to 0.5)",
    "Dense healthy vegetation (>= 0.5)",
]
NDVI_BREAKS = (0.0, 0.2, 0.5)


def classify_ndvi(ndvi):
    """Return (class map 0..3, with 255 for no-data; percentage table)."""
    cls = np.digitize(ndvi, NDVI_BREAKS).astype(np.uint8)
    valid = np.isfinite(ndvi)
    cls[~valid] = 255
    n = max(int(valid.sum()), 1)
    pct = [100.0 * float((cls == k).sum()) / n for k in range(4)]
    return cls, pct


# --------------------------------------------------------------------------
# Resolution demonstrations
# --------------------------------------------------------------------------
def degrade_spatial(band, factor):
    """Average factor x factor blocks, then repeat each value so the image keeps its
    size but looks blocky - a simulation of a sensor with `factor` times larger pixels."""
    if factor <= 1:
        return band.copy()
    h, w = band.shape
    h2, w2 = h - h % factor, w - w % factor
    blocks = band[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(blocks, axis=(1, 3))
    return np.repeat(np.repeat(mean, factor, axis=0), factor, axis=1).astype(np.float32)


def quantize(unit_band, bits):
    """Reduce a 0..1 image to 2**bits grey levels (radiometric resolution)."""
    levels = 2 ** int(bits)
    q = np.floor(np.clip(unit_band, 0, 1) * levels)
    return (np.clip(q, 0, levels - 1) / (levels - 1)).astype(np.float32)


# --------------------------------------------------------------------------
# Small helpers used by the app
# --------------------------------------------------------------------------
def pixel_window(arr, row, col, n=8):
    """n x n block of values with top-left corner (row, col), kept inside the image."""
    row = int(min(max(row, 0), arr.shape[0] - n))
    col = int(min(max(col, 0), arr.shape[1] - n))
    return arr[row: row + n, col: col + n], row, col


def pixel_values(scene, row, col):
    """Reflectance of the 4 bands at one pixel, in band order."""
    return [float(scene["refl"][b][row, col]) for b in BANDS]


def quick_picks(scene, ndvi):
    """Example pixels to click through: named samples (synthetic) or NDVI-percentile
    pixels (real data). Returns {label: (row, col)}."""
    if scene.get("samples"):
        return dict(scene["samples"])
    picks = {}
    for label, q in [("Dense vegetation? (about 99th percentile NDVI)", 99),
                     ("Typical pixel (median NDVI)", 50),
                     ("Water or very dark? (about 1st percentile NDVI)", 1)]:
        target = np.nanpercentile(ndvi, q)
        idx = np.nanargmin(np.abs(ndvi - target))
        picks[label] = tuple(int(i) for i in np.unravel_index(idx, ndvi.shape))
    return picks
