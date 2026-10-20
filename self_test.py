"""
self_test.py - quick automatic check of the core computations and figures.
Run:  python self_test.py      (takes a few seconds; prints PASS lines, stops at the first failure)
"""
import os
import tempfile

import numpy as np

import plots
import rs_core as rs


def check(name, condition):
    assert condition, "FAILED: " + name
    print("PASS  " + name)


# ---- NDVI formula on hand-made numbers --------------------------------------
nir = np.array([[0.40, 0.20, 0.02, 0.0, np.nan]], dtype=np.float32)
red = np.array([[0.10, 0.20, 0.04, 0.0, 0.10]], dtype=np.float32)
nd = rs.compute_ndvi(nir, red)
check("NDVI (0.40-0.10)/(0.40+0.10) = 0.6", abs(nd[0, 0] - 0.6) < 1e-6)
check("NDVI is 0 when NIR == Red", nd[0, 1] == 0)
check("NDVI is negative when NIR < Red (water-like)", abs(nd[0, 2] - (-1 / 3)) < 1e-6)
check("NDVI is NaN when NIR + Red = 0 (no division by zero)", np.isnan(nd[0, 3]))
check("NDVI is NaN when data is missing", np.isnan(nd[0, 4]))

# ---- Contrast enhancement ----------------------------------------------------
rng = np.random.default_rng(0)
band = (0.05 + 0.1 * rng.random((100, 100))).astype(np.float32)
band[0, 0] = np.nan
for m in rs.ENHANCEMENTS:
    out = rs.apply_enhancement(band, m)
    check("%s -> shape kept, values inside 0..1" % m, out.shape == band.shape and out.min() >= 0 and out.max() <= 1)
check("Min-max stretch reaches 0 and 1",
      rs.enhance_minmax(band)[1:, 1:].min() < 1e-6 and abs(rs.enhance_minmax(band).max() - 1) < 1e-6)
check("Raw (no enhancement) is dark: max below 0.2", rs.enhance_none(band).max() < 0.2)
eq = rs.enhance_equalize(band)
check("Equalisation flattens the histogram (std near uniform 0.289)", abs(eq[1:, 1:].std() - 0.289) < 0.03)
check("Gamma 0.5 brightens, gamma 2 darkens",
      rs.enhance_gamma(band, 0.5)[5:, 5:].mean() > rs.enhance_gamma(band, 2.0)[5:, 5:].mean())
check("Flat image does not crash", rs.enhance_percentile(np.full((5, 5), 0.3, np.float32)).max() == 0)

# ---- Resolution helpers -----------------------------------------------------
img = np.arange(64, dtype=np.float32).reshape(8, 8)
d = rs.degrade_spatial(img, 4)
check("Spatial degrade keeps size and makes 4x4 blocks", d.shape == (8, 8) and len(np.unique(d)) == 4)
check("Block value is the mean of the block", abs(d[0, 0] - img[:4, :4].mean()) < 1e-6)
check("Quantize to 2 bits gives 4 levels", len(np.unique(rs.quantize(np.linspace(0, 1, 1000), 2))) == 4)

# ---- Synthetic scene: expected NDVI order -----------------------------------
sc = rs.make_synthetic_scene()
ndvi = rs.compute_ndvi(sc["refl"]["nir"], sc["refl"]["red"])
check("Synthetic scene has 4 bands of 300x300", all(sc["refl"][b].shape == (300, 300) for b in rs.BANDS))
check("DN is uint16", all(sc["dn"][b].dtype == np.uint16 for b in rs.BANDS))
vals = {k: float(ndvi[r, c]) for k, (r, c) in sc["samples"].items()}
for k, v in vals.items():
    print("      synthetic sample  %-40s NDVI = %.3f" % (k, v))
check("Water NDVI < 0", vals["Water (lake)"] < 0)
check("Bare soil NDVI in 0..0.3", 0 <= vals["Bare soil"] < 0.3)
check("Built-up NDVI in 0..0.2", 0 <= vals["Built-up (town)"] < 0.2)
check("Stressed vegetation NDVI in 0.3..0.65", 0.3 < vals["Stressed / sparse vegetation (field)"] < 0.65)
check("Healthy vegetation NDVI > 0.7", vals["Healthy vegetation (field)"] > 0.7)
cls, pct = rs.classify_ndvi(ndvi)
check("Class percentages sum to 100", abs(sum(pct) - 100) < 1e-6)

# ---- Save / load round trip -------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "s.npz")
    rs.save_scene(path, sc["dn"], sc["meta"])
    back = rs.load_scene(path)
    check("Save/load round trip keeps DN exactly", all(np.array_equal(back["dn"][b], sc["dn"][b]) for b in rs.BANDS))
    check("Reflectance recomputed from DN matches within 0.0001",
          all(np.nanmax(np.abs(back["refl"][b] - sc["refl"][b])) < 1.1e-4 for b in rs.BANDS))
nodata_dn = {b: sc["dn"][b].copy() for b in rs.BANDS}
nodata_dn["red"][0, 0] = 0
check("DN 0 becomes NaN (no data)", np.isnan(rs.build_scene(nodata_dn, sc["meta"])["refl"]["red"][0, 0]))

# ---- Real data (only if you have downloaded it) -----------------------------
if os.path.exists(os.path.join("data", "scene.npz")):
    real = rs.load_scene(os.path.join("data", "scene.npz"))
    nd_real = rs.compute_ndvi(real["refl"]["nir"], real["refl"]["red"])
    check("Real data: NDVI within -1..1", np.nanmin(nd_real) >= -1 and np.nanmax(nd_real) <= 1)
    print("      real scene %s, NDVI min/mean/max = %.2f / %.2f / %.2f"
          % (real["meta"].get("item_id"), np.nanmin(nd_real), np.nanmean(nd_real), np.nanmax(nd_real)))
    scenes = [sc, real]
else:
    print("INFO  data/scene.npz not found - real-data checks skipped (run download_data.py first)")
    scenes = [sc]

# ---- Every figure can be drawn without errors --------------------------------
import matplotlib.pyplot as plt
for s in scenes:
    n = rs.compute_ndvi(s["refl"]["nir"], s["refl"]["red"])
    picks = rs.quick_picks(s, n)
    r, c = list(picks.values())[0]
    chans = rs.raw_channels(s, "False colour (NIR, R, G)")
    raw_img, _ = rs.make_composite(chans, rs.ENHANCEMENTS[0])
    enh_img, enh = rs.make_composite(chans, rs.ENHANCEMENTS[4])
    cl, pc = rs.classify_ndvi(n)
    figs = [plots.fig_em_spectrum(), plots.fig_four_bands(s), plots.fig_band_with_window(s, "red", r, c),
            plots.fig_resolution(s, 8, 4), plots.fig_composites(raw_img, enh_img, "k", "m"),
            plots.fig_histograms(chans, enh, ["a", "b", "c"]),
            plots.fig_ndvi_vs_falsecolour(enh_img, n, r, c, True), plots.fig_spectra(s, picks, list(picks)[0]),
            plots.fig_classes(cl, pc), plots.fig_ndvi_hist(n)]
    for f in figs:
        plt.close(f)
    check("All 10 figures draw for the %s scene" % ("real" if s["meta"]["is_real"] else "synthetic"), True)

print("\nALL TESTS PASSED")
