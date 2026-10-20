"""
download_data.py - fetch a SMALL crop of REAL Sentinel-2 L2A imagery (blue, green, red, NIR).

Run once (internet needed), then the app works offline from data/scene.npz:

    python download_data.py
    python download_data.py --lat 30.00 --lon 31.20 --half-km 4      (these are the defaults)

Data source : Sentinel-2 Level-2A, ESA / European Commission Copernicus programme.
Hosting     : "Sentinel-2 Cloud-Optimized GeoTIFFs" on the AWS Open Data programme,
              found through the public Earth Search STAC API (no account or key needed).
Licence     : Copernicus Sentinel data are free and open. Credit: "Contains modified
              Copernicus Sentinel data [year of acquisition]".

Only the pixels inside a small window are read (HTTP range requests), so the download
is a few megabytes, not the ~1 GB of a full Sentinel-2 tile.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np
import requests

try:
    import certifi
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())   # helps GDAL trust HTTPS on Windows
except ImportError:
    pass

import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

import rs_core as rs

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
ASSET_FOR_BAND = {"blue": "blue", "green": "green", "red": "red", "nir": "nir"}   # 10 m bands
CLOUD_CLASSES = (0, 1, 3, 8, 9, 10)   # SCL: no data, saturated, cloud shadow, cloud med/high, thin cirrus


def search_items(lat, lon, half_km, start, end, max_cloud, limit=12):
    """Ask the STAC API for recent, low-cloud Sentinel-2 L2A scenes covering the point."""
    dlat = half_km / 111.32
    dlon = half_km / (111.32 * np.cos(np.radians(lat)))
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [lon - dlon, lat - dlat, lon + dlon, lat + dlat],
        "datetime": "%sT00:00:00Z/%sT23:59:59Z" % (start, end),
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": limit,
    }
    r = requests.post(STAC_URL, json=body, timeout=60)
    r.raise_for_status()
    return r.json().get("features", [])


def band_scale_offset(item, asset_name):
    """Read scale/offset from the STAC metadata; fall back to the documented values."""
    rb = item["assets"][asset_name].get("raster:bands", [{}])[0]
    if "scale" in rb and "offset" in rb:
        return float(rb["scale"]), float(rb["offset"])
    when = item["properties"]["datetime"][:10]
    return (0.0001, -0.1) if when >= "2022-01-25" else (0.0001, 0.0)


def read_crop(item, lat, lon, size):
    """Read a size x size window (10 m pixels) centred on lat/lon for the 4 bands.
    Returns (dn dict, crs string) or None if the window is not fully inside this tile."""
    dn, crs_name = {}, None
    centre_xy = None
    for band, asset in ASSET_FOR_BAND.items():
        href = item["assets"][asset]["href"]
        with rasterio.open(href) as src:
            if centre_xy is None:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                centre_xy = (xs[0], ys[0])
                crs_name = src.crs.to_string()
            row, col = src.index(*centre_xy)
            win = Window(col - size // 2, row - size // 2, size, size)
            if win.col_off < 0 or win.row_off < 0 or win.col_off + size > src.width or win.row_off + size > src.height:
                return None
            dn[band] = src.read(1, window=win)
    return dn, crs_name, centre_xy


def cloud_fraction(item, centre_xy, size):
    """Fraction of crop pixels flagged as cloud / shadow / no data in the SCL layer."""
    if "scl" not in item["assets"]:
        return 0.0
    with rasterio.open(item["assets"]["scl"]["href"]) as src:     # 20 m layer
        row, col = src.index(*centre_xy)
        half = size // 4
        win = Window(col - half, row - half, size // 2, size // 2)
        scl = src.read(1, window=win, out_shape=(size, size), resampling=Resampling.nearest)
    return float(np.isin(scl, CLOUD_CLASSES).mean())


def save_preview(scene, path):
    """Write a quick PNG (true colour, false colour, NDVI) so you can check the crop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    true_img, _ = rs.make_composite(rs.raw_channels(scene, "True colour (R, G, B)"), rs.ENHANCEMENTS[2])
    false_img, _ = rs.make_composite(rs.raw_channels(scene, "False colour (NIR, R, G)"), rs.ENHANCEMENTS[2])
    ndvi = rs.compute_ndvi(scene["refl"]["nir"], scene["refl"]["red"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(true_img); ax[0].set_title("True colour")
    ax[1].imshow(false_img); ax[1].set_title("False colour (NIR, R, G)")
    im = ax[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1); ax[2].set_title("NDVI")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Download a small real Sentinel-2 crop.")
    p.add_argument("--lat", type=float, default=30.00, help="centre latitude (default 30.00)")
    p.add_argument("--lon", type=float, default=31.20, help="centre longitude (default 31.20)")
    p.add_argument("--half-km", type=float, default=4.0, help="half the crop width in km (default 4 -> 8 km x 8 km)")
    p.add_argument("--start", default=str(today - dt.timedelta(days=365)), help="start date YYYY-MM-DD")
    p.add_argument("--end", default=str(today), help="end date YYYY-MM-DD")
    p.add_argument("--max-cloud", type=float, default=10.0, help="max scene cloud cover %% (default 10)")
    p.add_argument("--out", default=os.path.join("data", "scene.npz"))
    a = p.parse_args()

    size = int(round(a.half_km * 2000 / 10.0 / 4.0)) * 4        # pixels per side, multiple of 4
    print("Searching Sentinel-2 L2A scenes near (%.4f, %.4f), %s to %s, cloud < %.0f%% ..."
          % (a.lat, a.lon, a.start, a.end, a.max_cloud))
    try:
        items = search_items(a.lat, a.lon, a.half_km, a.start, a.end, a.max_cloud)
    except requests.RequestException as e:
        sys.exit("Could not reach the STAC server: %s\nCheck your internet connection / firewall and try again." % e)
    if not items:
        sys.exit("No scenes found. Try --max-cloud 30 or a longer date range (--start 2025-01-01).")
    print("Found %d candidate scenes. Checking them (newest first)..." % len(items))

    for item in items:
        sid = item["id"]
        try:
            res = read_crop(item, a.lat, a.lon, size)
            if res is None:
                print("  %s: crop not fully inside this tile, skipping" % sid)
                continue
            dn, crs_name, centre_xy = res
            cf = cloud_fraction(item, centre_xy, size)
        except Exception as e:                      # network hiccup, missing asset, ...
            print("  %s: could not read (%s), skipping" % (sid, e))
            continue
        nodata = float(np.mean([(dn[b] == 0).mean() for b in rs.BANDS]))
        if cf > 0.02 or nodata > 0.01:
            print("  %s: cloud/no-data in crop = %.1f%% / %.1f%%, skipping" % (sid, 100 * cf, 100 * nodata))
            continue

        scales = {band_scale_offset(item, ASSET_FOR_BAND[b]) for b in rs.BANDS}
        if len(scales) != 1:
            print("  %s: bands have different scale/offset, skipping" % sid)
            continue
        scale, offset = scales.pop()
        props = item["properties"]
        meta = {
            "is_real": True,
            "title": "REAL Sentinel-2 L2A satellite data (genuine blue, green, red and NIR bands)",
            "source": "ESA Copernicus Sentinel-2 L2A via AWS Open Data / Earth Search STAC (%s)" % STAC_URL,
            "item_id": sid,
            "license": "Copernicus Sentinel data: free and open. Credit: Contains modified Copernicus Sentinel data %s."
                       % props["datetime"][:4],
            "sensor": "Sentinel-2 MSI, bands B02 (490 nm), B03 (560 nm), B04 (665 nm), B08 (842 nm), all 10 m",
            "datetime": props["datetime"], "cloud_cover": props.get("eo:cloud_cover", "n/a"),
            "pixel_size_m": 10, "crs": crs_name,
            "centre_lat": a.lat, "centre_lon": a.lon, "half_km": a.half_km,
            "scale": scale, "offset": offset,
            "note": "Reflectance = DN * scale + offset. DN 0 = no data.",
        }
        rs.save_scene(a.out, dn, meta)
        scene = rs.load_scene(a.out)
        ndvi = rs.compute_ndvi(scene["refl"]["nir"], scene["refl"]["red"])
        preview = os.path.join(os.path.dirname(a.out) or ".", "preview.png")
        save_preview(scene, preview)
        print("\nSUCCESS: saved %s  (%d x %d pixels, scene %s, acquired %s)" % (a.out, size, size, sid, props["datetime"][:10]))
        print("  scale=%g offset=%g | mean reflectance: %s"
              % (scale, offset, {b: round(float(np.nanmean(scene['refl'][b])), 3) for b in rs.BANDS}))
        print("  NDVI min/mean/max = %.2f / %.2f / %.2f" % (np.nanmin(ndvi), np.nanmean(ndvi), np.nanmax(ndvi)))
        print("  Quick-look image written to %s - open it to check the area looks interesting." % preview)
        return
    sys.exit("No cloud-free crop found among the candidates. Try --max-cloud 30, a longer date range, or other coordinates.")


if __name__ == "__main__":
    main()
