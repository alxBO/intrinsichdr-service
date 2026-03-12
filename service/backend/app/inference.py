"""IntrinsicHDR inference pipeline - supports CUDA, MPS, and CPU.

Uses the vendor's pipeline (ECCV 2024) with two changes:
1. Neural linearization via SingleHDR (DequantizationNet + LinearizationNet)
   replacing the vendor's TensorFlow-based dequantize_and_linearize.py.
2. blend_imgs fix: vendor uses mask >= 0 (always True) for lstsq fitting,
   which scales ALL pixels. We use mask > 0 to fit only highlight overlap.
"""

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

# Path to SingleHDR weights and PCA basis
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "weights")


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
        self._hdr_reconstruction = vendor_inference.hdr_reconstruction
        self._load_reconstruction_models = vendor_inference.load_reconstruction_models
        from intrinsic_decomposition.common.model_util import load_models
        from intrinsic_decomposition.common.general import round_32
        from src.decomposition_utils import decompose_torch

        self._load_models = load_models
        self._round_32 = round_32
        self._decompose_torch = decompose_torch

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

        # Load SingleHDR linearization networks (PyTorch)
        logger.info("Loading SingleHDR linearization networks...")
        backend_dir = os.path.join(os.path.dirname(__file__), "..")
        if os.path.abspath(backend_dir) not in sys.path:
            sys.path.insert(0, os.path.abspath(backend_dir))
        from linearization_nets import DequantizationNet, LinearizationNet

        weights_dir = os.path.abspath(WEIGHTS_DIR)
        invemor_path = os.path.join(weights_dir, "invemor.txt")

        self._deq_net = DequantizationNet()
        self._lin_net = LinearizationNet(invemor_path)

        self._deq_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "dequantization.pt"),
            map_location="cpu", weights_only=True))
        self._lin_net.load_state_dict(torch.load(
            os.path.join(weights_dir, "linearization.pt"),
            map_location="cpu", weights_only=True), strict=False)

        self._deq_net.to(self.device).eval()
        self._lin_net.to(self.device).eval()
        logger.info("SingleHDR linearization networks loaded.")

        self.lock = threading.Lock()
        logger.info("IntrinsicHDR pipeline ready.")

    @torch.no_grad()
    def _neural_linearize(self, srgb_np: np.ndarray) -> np.ndarray:
        """Dequantize + linearize an sRGB image using SingleHDR networks.

        DequantizationNet and LinearizationNet are channel-order agnostic
        (no VGG mean subtraction, no channel-specific ops), so RGB input is fine.
        """
        from linearization_nets import apply_rf_torch

        h, w = srgb_np.shape[:2]
        img = srgb_np.copy()

        was_resized = False
        if h % 64 != 0 or w % 64 != 0:
            rh = int(np.ceil(float(h) / 64.0)) * 64
            rw = int(np.ceil(float(w) / 64.0)) * 64
            img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_CUBIC)
            was_resized = True

        padding = 32
        img_padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'symmetric')

        x = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).to(self.device)

        C_pred = self._deq_net(x).clamp(0, 1)
        del x

        pred_invcrf = self._lin_net(C_pred)
        B_pred = apply_rf_torch(C_pred, pred_invcrf)
        del C_pred, pred_invcrf

        linear = B_pred[0].permute(1, 2, 0).cpu().numpy()
        del B_pred

        linear = linear[padding:-padding, padding:-padding]

        if was_resized:
            linear = cv2.resize(linear, (w, h), interpolation=cv2.INTER_CUBIC)

        return np.clip(linear, 0, None).astype(np.float32)

    def _blend_direct(self, ldr, hdr, mask):
        """Blend LDR and HDR using mask — no lstsq scaling.

        Where mask == 0: use original LDR (preserves blacks/midtones exactly).
        Where mask > 0: soft blend towards reconstructed HDR highlights.
        No global scaling, so dark values are never lifted.
        """
        mask_pct = 100.0 * (mask > 0.01).mean()
        logger.info("blend: mask coverage=%.1f%%, max=%.4f", mask_pct, mask.max())

        blended = mask[:, :, np.newaxis] * hdr + (1 - mask[:, :, np.newaxis]) * ldr

        return blended

    @torch.no_grad()
    def run(self, img_bytes: bytes, progress_cb: ProgressCallback = None,
            max_res: int = 4096, img_scale: float = 1.0, proc_scale: float = 1.0) -> np.ndarray:
        """Run vendor pipeline with neural linearization and fixed blend."""
        with self.lock:
            if progress_cb:
                progress_cb("preprocessing", 0.05, "Decoding image...")

            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Cannot decode image")

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            del img_bgr

            ldr_srgb = img_rgb.astype(np.float32) / 255.0
            ldr_srgb = np.clip(ldr_srgb, 0, 1)
            h_in, w_in = ldr_srgb.shape[:2]

            # --- Neural linearization ---
            if progress_cb:
                progress_cb("linearization", 0.08, "Neural linearization (SingleHDR)...")

            ldr_c = self._neural_linearize(ldr_srgb)

            pct_above_08 = 100.0 * (ldr_c.max(axis=-1) > 0.8).mean()
            logger.info(
                "Linearized: range=[%.4f, %.4f], mean=%.4f, >0.8: %.1f%%",
                ldr_c.min(), ldr_c.max(), ldr_c.mean(), pct_above_08,
            )

            if progress_cb:
                progress_cb("linearization", 0.15, "Linearized (%dx%d)." % (w_in, h_in))

            # --- Vendor pipeline: decompose + reconstruct ---
            if progress_cb:
                progress_cb("decomposition", 0.20, "Intrinsic decomposition...")

            ldr_c = np.clip(ldr_c, 0, 1)

            if max(h_in, w_in) > max_res:
                s = max_res / max(h_in, w_in)
                h_proc, w_proc = h_in * s, w_in * s
            else:
                h_proc, w_proc = h_in, w_in

            new_h = self._round_32(h_proc)
            new_w = self._round_32(w_proc)
            ldr_resized = cv2.resize(ldr_c, (new_w, new_h))
            ldr_t = torch.tensor(ldr_resized * proc_scale).permute(2, 0, 1).unsqueeze(0)

            pred_inv_shading, pred_albedo = self._decompose_torch(
                self._decomp_models,
                torch.clamp(ldr_t, 0, 1),
            )

            if pred_inv_shading is None:
                raise RuntimeError("Intrinsic decomposition failed (lstsq did not converge)")

            if progress_cb:
                progress_cb("reconstruction", 0.45, "HDR reconstruction...")

            # Vendor's hdr_reconstruction — linear mask, as model was trained
            rec_results = self._hdr_reconstruction(
                self._recon_models,
                pred_albedo, pred_inv_shading,
                ldr_t, proc_scale,
            )

            hdr_raw = rec_results[0]       # Refined HDR (linear space)
            blend_mask = rec_results[5]     # Linear highlight mask

            logger.info(
                "Raw HDR (pre-blend): range=[%.4f, %.4f], mean=%.4f",
                hdr_raw.min(), hdr_raw.max(), hdr_raw.mean(),
            )
            logger.info(
                "Linear mask: coverage=%.1f%%, max=%.4f",
                100.0 * (blend_mask > 0.01).mean(), blend_mask.max(),
            )

            if progress_cb:
                progress_cb("postprocessing", 0.75, "Blending...")

            # Resize to original resolution
            hdr_r = cv2.resize(hdr_raw, (w_in, h_in), interpolation=cv2.INTER_CUBIC)

            # sRGB blend mask: tight threshold on near-clipped pixels only.
            # Only blend in HDR where pixels are truly clipped (sRGB >= 0.95),
            # with a narrow transition 0.95→1.0. This avoids reducing brightness
            # of moderately bright pixels where the model output < original.
            srgb_max = ldr_srgb.max(axis=-1)  # max channel per pixel
            bl_mask = np.clip((srgb_max - 0.95) / 0.05, 0, 1)

            logger.info(
                "sRGB blend mask: coverage=%.1f%% (linear mask was %.1f%%)",
                100.0 * (bl_mask > 0.01).mean(),
                100.0 * (blend_mask > 0.01).mean(),
            )

            # Gamma 2.2 decode — matches frontend's SDR→linear conversion
            # (proper sRGB has a linear segment below 0.04045 that lifts darks ~12x
            # vs gamma 2.2, destroying contrast in the comparison metrics)
            srgb_linear = np.power(ldr_srgb, 2.2).astype(np.float32)

            # Direct blend: sRGB-decoded darks (true blacks) + HDR highlights
            hdr_out = self._blend_direct(srgb_linear, hdr_r, bl_mask)

            logger.info(
                "After blend: range=[%.4f, %.4f], mean=%.4f",
                hdr_out.min(), hdr_out.max(), hdr_out.mean(),
            )

            if img_scale != 1.0:
                hdr_out = hdr_out * img_scale

            hdr_out = np.maximum(hdr_out, 0.0).astype(np.float32)

            if progress_cb:
                progress_cb("postprocessing", 0.92, "Complete.")

            logger.info(
                "HDR output: shape=%s, range=[%.4f, %.4f], mean=%.4f",
                hdr_out.shape, hdr_out.min(), hdr_out.max(), hdr_out.mean(),
            )

            return hdr_out

    @staticmethod
    def _clear_device_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def close(self):
        del self._decomp_models, self._recon_models, self._deq_net, self._lin_net
        self._clear_device_cache()
