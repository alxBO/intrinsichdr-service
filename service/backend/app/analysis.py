"""Image analysis utilities for SDR input and HDR output."""

import numpy as np
import cv2


def analyze_sdr(img_bytes: bytes, file_size: int, filename: str) -> dict:
    """Analyze an SDR image from bytes. Returns metadata and histogram data."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot decode image")

    h, w, c = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Per-channel histograms (256 bins)
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten().tolist()
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten().tolist()
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten().tolist()

    total_pixels = h * w
    # Clipping: pixels at 0 or 255 in any channel
    clipped_low = int(np.any(img_rgb == 0, axis=-1).sum())
    clipped_high = int(np.any(img_rgb == 255, axis=-1).sum())
    clipping_percent = 100.0 * (clipped_low + clipped_high) / total_pixels

    # Luminance stats
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_brightness = float(np.mean(gray))
    median_brightness = float(np.median(gray))

    # Linear luminance (same space as HDR output for direct comparison)
    # Gamma decode (sRGB ~ 2.2) then Rec.709 luminance
    img_linear = np.power(img_rgb.astype(np.float32) / 255.0, 2.2)
    lum_linear = _luminance(img_linear)
    mean_luminance_linear = float(np.mean(lum_linear))
    peak_luminance_linear = float(np.max(lum_linear))

    # Contrast ratio in linear domain (P99.9 / P0.1)
    positive_lum = lum_linear[lum_linear > 0]
    if positive_lum.size > 0:
        p_low = float(np.percentile(positive_lum, 0.1))
        p_high = float(np.percentile(positive_lum, 99.9))
        contrast_ratio = p_high / p_low if p_low > 0 else 0.0
    else:
        contrast_ratio = 0.0

    # Dynamic range in EV (from non-zero luminance values)
    positive_gray = gray[gray > 0]
    if positive_gray.size > 0:
        dynamic_range_ev = float(np.log2(positive_gray.max() / positive_gray.min()))
    else:
        dynamic_range_ev = 0.0

    ext = filename.rsplit('.', 1)[-1].upper() if '.' in filename else 'UNKNOWN'

    return {
        "width": w,
        "height": h,
        "file_size_bytes": file_size,
        "format": ext,
        "histogram": {
            "r": [int(x) for x in hist_r],
            "g": [int(x) for x in hist_g],
            "b": [int(x) for x in hist_b],
        },
        "dynamic_range_ev": round(dynamic_range_ev, 1),
        "mean_brightness": round(mean_brightness, 1),
        "median_brightness": round(median_brightness, 1),
        "clipping_percent": round(clipping_percent, 2),
        "mean_luminance_linear": round(mean_luminance_linear, 4),
        "peak_luminance_linear": round(peak_luminance_linear, 4),
        "contrast_ratio": round(contrast_ratio, 1),
    }


def _luminance(img: np.ndarray) -> np.ndarray:
    """Compute luminance from linear RGB."""
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def analyze_hdr(hdr_img: np.ndarray) -> dict:
    """Analyze an HDR image (float32 RGB). Returns dynamic range, luminance stats, etc."""
    lum = _luminance(hdr_img)
    positive = lum[lum > 0]

    if positive.size > 0:
        # Use 0.1th and 99.9th percentiles for robust dynamic range estimation
        lum_low = float(np.percentile(positive, 0.1))
        lum_high = float(np.percentile(positive, 99.9))
        dynamic_range_ev = float(np.log2(lum_high / lum_low)) if lum_low > 0 else 0.0
        contrast_ratio = lum_high / lum_low if lum_low > 0 else 0.0
    else:
        dynamic_range_ev = 0.0
        contrast_ratio = 0.0

    peak_luminance = float(hdr_img.max())
    mean_luminance = float(np.mean(lum))

    percentiles = {}
    for p in [50, 90, 99, 99.9]:
        percentiles[str(p)] = round(float(np.percentile(lum, p)), 4)

    # Log-domain histogram for display (100 bins)
    if positive.size > 0:
        log_lum = np.log10(np.clip(positive, 1e-6, None))
        log_min, log_max = float(log_lum.min()), float(log_lum.max())
        hist_counts, hist_edges = np.histogram(log_lum, bins=100)
        hdr_histogram = {
            "counts": hist_counts.tolist(),
            "bin_edges": [round(float(e), 4) for e in hist_edges.tolist()],
            "log_min": round(log_min, 4),
            "log_max": round(log_max, 4),
        }
    else:
        hdr_histogram = {"counts": [], "bin_edges": [], "log_min": 0, "log_max": 0}

    return {
        "dynamic_range_ev": round(dynamic_range_ev, 1),
        "peak_luminance": round(peak_luminance, 4),
        "mean_luminance": round(mean_luminance, 4),
        "luminance_percentiles": percentiles,
        "contrast_ratio": round(contrast_ratio, 1),
        "hdr_histogram": hdr_histogram,
    }
