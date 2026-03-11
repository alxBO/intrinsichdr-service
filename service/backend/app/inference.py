"""IntrinsicHDR inference pipeline - supports CUDA, MPS, and CPU."""

import logging
import os
import sys
import threading
from typing import Callable, Optional

import cv2
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, float, str], None]]

# Path to the IntrinsicHDR vendor directory
VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "IntrinsicHDR")


def _get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_vendor_path():
    """Add vendor IntrinsicHDR directory to sys.path if not already present."""
    vendor_abs = os.path.abspath(VENDOR_DIR)
    if vendor_abs not in sys.path:
        sys.path.insert(0, vendor_abs)
    return vendor_abs


class IntrinsicHDRPipeline:
    """Wraps the vendor IntrinsicHDR pipeline for our service."""

    def __init__(self):
        self.device = _get_device()
        logger.info("IntrinsicHDR pipeline using device: %s", self.device)

        vendor_path = _ensure_vendor_path()
        logger.info("Vendor path: %s", vendor_path)

        # Import vendor modules — must happen after _ensure_vendor_path()
        import importlib
        vendor_inference = importlib.import_module("inference")
        self._intrinsic_hdr = vendor_inference.intrinsic_hdr
        self._load_reconstruction_models = vendor_inference.load_reconstruction_models
        from intrinsic_decomposition.common.model_util import load_models

        # Load decomposition models (downloads from GitHub releases on first run)
        logger.info("Loading intrinsic decomposition models...")
        self._decomp_models = load_models(
            ord_path='vivid_bird_318_300.pt',
            mrg_path='fluent_eon_138_200.pt',
            device=self.device,
        )
        logger.info("Decomposition models loaded.")

        # Load reconstruction models (shading, albedo, refinement)
        logger.info("Loading reconstruction models...")
        self._recon_models = self._load_reconstruction_models(self.device)
        logger.info("Reconstruction models loaded.")

        self.lock = threading.Lock()
        logger.info("IntrinsicHDR pipeline ready.")

    @torch.no_grad()
    def run(self, img_bytes: bytes, progress_cb: ProgressCallback = None,
            max_res: int = 4096, img_scale: float = 1.0, proc_scale: float = 1.0) -> np.ndarray:
        """Run the full IntrinsicHDR pipeline.

        Args:
            img_bytes: Raw image bytes (JPEG/PNG/etc).
            progress_cb: Callback(stage, progress, message).
            max_res: Maximum processing resolution (longest side).
            img_scale: Input brightness scale factor.
            proc_scale: Processing scale factor.

        Returns:
            float32 numpy array (H, W, 3) - linear HDR image.
        """
        with self.lock:
            # --- Decode image ---
            if progress_cb:
                progress_cb("preprocessing", 0.05, "Decoding image...")

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Cannot decode image")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            del img_bgr

            # Normalize to [0, 1] float32 (vendor expects this)
            ldr_c = img_rgb.astype(np.float32) / 255.0

            if progress_cb:
                progress_cb("preprocessing", 0.10, "Image loaded.")

            # --- Run vendor pipeline ---
            if progress_cb:
                progress_cb("inference", 0.15, "Running IntrinsicHDR pipeline...")

            results = self._intrinsic_hdr(
                decomp_models=self._decomp_models,
                reconstruction_networks=self._recon_models,
                ldr_c=ldr_c,
                max_res=max_res,
                proc_scale=proc_scale,
            )

            if progress_cb:
                progress_cb("inference", 0.85, "IntrinsicHDR complete.")

            # Apply img_scale (brightness adjustment)
            hdr_out = results['rgb_hdr']
            if img_scale != 1.0:
                hdr_out = hdr_out * img_scale

            # Ensure non-negative
            hdr_out = np.maximum(hdr_out, 0.0).astype(np.float32)

            if progress_cb:
                progress_cb("postprocessing", 0.92, "Post-processing complete.")

            return hdr_out

    @staticmethod
    def _clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def close(self):
        del self._decomp_models, self._recon_models
        self._clear_device_cache()
