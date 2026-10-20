"""
plots.py - every Matplotlib figure used by the app.
Each function returns a Figure object; app.py only displays it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")                      # draw to memory, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import rs_core as rs

CLASS_COLOURS = ["#2b83ba", "#d9c38c", "#a6d96a", "#1a9641"]


def _mark(ax, row, col):
    """Draw a ring on the chosen pixel (white halo + black ring so it shows on any colour)."""
    ax.plot(col, row, "o", mfc="none", mec="white", ms=18, mew=4)
    ax.plot(col, row, "o", mfc="none", mec="black", ms=18, mew=1.8)


def fig_em_spectrum():
    """Electromagnetic spectrum (log scale) with the 4 bands used in this project."""
    fig, ax = plt.subplots(figsize=(10, 2.6))
    regions = [  # name, start, end in micrometres, colour
        ("UV", 0.1, 0.4, "#b39ddb"), ("Visible", 0.4, 0.7, "#fff176"),
        ("NIR", 0.7, 1.3, "#ef9a9a"), ("SWIR", 1.3, 3.0, "#f48fb1"),
        ("Thermal IR", 3.0, 15.0, "#ffab91"), ("Microwave / radar", 1e3, 1e6, "#90caf9"),
    ]
    for name, a, b, c in regions:
        ax.axvspan(a, b, color=c, alpha=0.85)
        ax.text(np.sqrt(a * b), 0.88, name, ha="center", va="center", fontsize=8)
    colours = {"blue": "#1f4ed8", "green": "#188038", "red": "#d32f2f", "nir": "#7b1fa2"}
    for b in rs.BANDS:
        x = rs.BAND_TABLE[b]["nm"] / 1000.0
        ax.axvline(x, color=colours[b], lw=2)
        ax.text(x, 0.35 if b != "green" else 0.2, rs.BAND_TABLE[b]["s2"] + "\n" + str(rs.BAND_TABLE[b]["nm"]) + " nm",
                color=colours[b], fontsize=7, ha="center", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Wavelength (micrometres, log scale)   -   1 um = 1000 nm")
    ax.set_title("Electromagnetic spectrum and the 4 bands used here (lines)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_four_bands(scene):
    """The 4 bands as grey images (each with its own 2-98% display stretch)."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, b in zip(axes, rs.BANDS):
        ax.imshow(rs.enhance_percentile(scene["refl"][b]), cmap="gray", vmin=0, vmax=1)
        info = rs.BAND_TABLE[b]
        ax.set_title("%s (%s)\n%d nm" % (info["label"], info["s2"], info["nm"]), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_band_with_window(scene, band, row, col, n=8):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(rs.enhance_percentile(scene["refl"][band]), cmap="gray", vmin=0, vmax=1)
    ax.add_patch(Rectangle((col - 0.5, row - 0.5), n, n, fill=False, ec="red", lw=2))
    ax.set_title("%s band - red box = %dx%d pixels shown in the tables" % (band.upper(), n, n), fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def fig_resolution(scene, factor, bits):
    """Original vs degraded true-colour image (spatial + radiometric resolution)."""
    chans_o, chans_d = [], []
    for b in ("red", "green", "blue"):
        band = scene["refl"][b]
        chans_o.append(rs.enhance_percentile(band))
        d = rs.enhance_percentile(rs.degrade_spatial(band, factor))
        chans_d.append(rs.quantize(d, bits))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(np.dstack(chans_o))
    axes[0].set_title("Original: %d m pixels, 256 grey levels" % scene["meta"]["pixel_size_m"], fontsize=10)
    axes[1].imshow(np.dstack(chans_d), interpolation="nearest")
    axes[1].set_title("Degraded: %d m pixels, %d grey levels" %
                      (scene["meta"]["pixel_size_m"] * factor, 2 ** bits), fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def fig_composites(raw_img, enh_img, title_kind, method):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(raw_img)
    axes[0].set_title("BEFORE: raw reflectance, no enhancement", fontsize=10)
    axes[1].imshow(enh_img)
    axes[1].set_title("AFTER: %s" % method, fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title_kind, fontsize=11)
    fig.tight_layout()
    return fig


def fig_histograms(raw_chs, enh_chs, ch_names):
    """Histograms of the 3 display channels before and after enhancement."""
    colours = ["#d32f2f", "#188038", "#1f4ed8"]
    valid_vals = [c[np.isfinite(c)] for c in raw_chs]
    xmax = float(max(np.percentile(v, 99.9) for v in valid_vals))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for raw, enh, col, name in zip(raw_chs, enh_chs, colours, ch_names):
        mask = np.isfinite(raw)
        axes[0].hist(raw[mask], bins=120, range=(0, xmax), histtype="step", color=col, lw=1.6, label=name)
        axes[1].hist(enh[mask], bins=120, range=(0, 1), histtype="step", color=col, lw=1.6, label=name)
    axes[0].set_title("Histogram BEFORE (x = reflectance)", fontsize=10)
    axes[1].set_title("Histogram AFTER (x = display value 0-1)", fontsize=10)
    for ax in axes:
        ax.set_ylabel("number of pixels")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_ndvi_vs_falsecolour(false_img, ndvi, row, col, stretch_colour_scale):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    axes[0].imshow(false_img)
    axes[0].set_title("False colour (NIR, R, G) - vegetation looks red", fontsize=10)
    if stretch_colour_scale:
        vmin, vmax = np.nanpercentile(ndvi, [2, 98])
    else:
        vmin, vmax = -1.0, 1.0
    im = axes[1].imshow(ndvi, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[1].set_title("NDVI map  (colour scale %.2f to %.2f)" % (vmin, vmax), fontsize=10)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    cb.set_label("NDVI")
    for ax in axes:
        ax.tick_params(labelsize=7)
        _mark(ax, row, col)
    fig.tight_layout()
    return fig


def fig_spectra(scene, picks, selected_label):
    """Spectral signatures (reflectance vs wavelength) of the quick-pick pixels."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    wl = [rs.BAND_TABLE[b]["nm"] for b in rs.BANDS]
    for label, (r, c) in picks.items():
        vals = rs.pixel_values(scene, r, c)
        sel = label == selected_label
        ax.plot(wl, vals, "-o", lw=3.2 if sel else 1.2, alpha=1.0 if sel else 0.55,
                label=label + ("  <- selected" if sel else ""))
    ax.axvspan(640, 690, color="#d32f2f", alpha=0.12)
    ax.axvspan(780, 900, color="#7b1fa2", alpha=0.12)
    ax.text(665, ax.get_ylim()[0], "Red band", ha="center", va="bottom", fontsize=8, color="#d32f2f")
    ax.text(842, ax.get_ylim()[0], "NIR band", ha="center", va="bottom", fontsize=8, color="#7b1fa2")
    ax.set_xticks(wl)
    ax.set_xticklabels(["Blue\n490", "Green\n560", "Red\n665", "NIR\n842"])
    ax.set_ylabel("Surface reflectance")
    ax.set_xlabel("Band / wavelength (nm)")
    ax.set_title("Spectral signatures of the example pixels", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def fig_classes(cls, pct):
    colours = ListedColormap(CLASS_COLOURS)
    shown = np.ma.masked_equal(cls, 255)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.imshow(shown, cmap=colours, vmin=0, vmax=3, interpolation="nearest")
    ax.legend(handles=[Patch(color=CLASS_COLOURS[i], label="%s  %.1f%%" % (rs.NDVI_CLASS_NAMES[i], pct[i]))
                       for i in range(4)],
              loc="upper center", bbox_to_anchor=(0.5, -0.04), fontsize=7, frameon=False)
    ax.set_title("NDVI classes (rule-of-thumb thresholds)", fontsize=10)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def fig_ndvi_hist(ndvi):
    v = ndvi[np.isfinite(ndvi)]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(v, bins=100, range=(-1, 1), color="#558b2f")
    for x in rs.NDVI_BREAKS:
        ax.axvline(x, color="k", ls="--", lw=1)
    ax.set_xlabel("NDVI")
    ax.set_ylabel("number of pixels")
    ax.set_title("NDVI histogram (dashed lines = class thresholds)", fontsize=10)
    fig.tight_layout()
    return fig
