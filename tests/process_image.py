#!/usr/bin/env python3
from __future__ import annotations

"""
Small TorchXRayVision helper for manual sanity checks.

This replaces the older duplicated scripts:
- tests/process_image_224.py
- tests/process_image_512.py

It keeps the same basic behavior: load an X-ray (DICOM or common image formats),
optionally center-crop+resize, run a pretrained TorchXRayVision model, and print
per-pathology predictions (and optionally pooled features).
"""

import argparse
import re
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms
import torchxrayvision as xrv


def _infer_input_size(weights: str) -> int | None:
    """Infer model input size from weight name (e.g. 'res224' / 'res512')."""
    m = re.search(r"res(\d{3,4})", weights)
    return int(m.group(1)) if m else None


def _load_xrv_image(path: str | Path) -> np.ndarray:
    """Load and normalize an X-ray image for TorchXRayVision.

    Returns a numpy array shaped (1, H, W) in TorchXRayVision's normalized space.
    """
    p = Path(path)
    s = str(p)
    try:
        return xrv.utils.load_image(s)
    except Exception:
        # Some DICOM exports omit the 128-byte preamble ("DICM" marker); fall back.
        if p.suffix.lower() in (".dcm", ".dicom"):
            return xrv.utils.read_xray_dcm(s)[None, ...]
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    # Jupyter notebooks sometimes pass "-f <path>"; accept and ignore it.
    parser.add_argument("-f", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("img_path", type=str, help="Path to an image or DICOM file")
    parser.add_argument(
        "--weights",
        type=str,
        default="densenet121-res224-all",
        help="TorchXRayVision pretrained weights name",
    )
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA if available")
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Apply TorchXRayVision center-crop + resize (recommended for consistency)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Resize target (e.g. 224 or 512). If omitted, inferred from --weights.",
    )
    parser.add_argument("--feats", action="store_true", help="Also print pooled features")

    cfg = parser.parse_args()

    img = _load_xrv_image(cfg.img_path)

    if cfg.resize:
        size = int(cfg.size) if int(cfg.size) > 0 else (_infer_input_size(cfg.weights) or 224)
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)]
        )
    else:
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop()])

    img = transform(img)

    model = xrv.models.get_model(cfg.weights)
    device = torch.device("cuda" if (cfg.cuda and torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    out: dict[str, object] = {}
    with torch.no_grad():
        x = torch.from_numpy(img).unsqueeze(0).to(device)

        if cfg.feats:
            # TorchXRayVision models typically expose `.features()`; this is best-effort.
            feats = model.features(x)  # type: ignore[attr-defined]
            feats = F.relu(feats, inplace=False)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            out["feats"] = feats.detach().cpu().numpy().reshape(-1).tolist()

        preds = model(x).detach().cpu().squeeze().numpy()

    # Prefer model-provided pathology names when available and dimensionally compatible.
    names = list(getattr(model, "pathologies", []) or [])
    if not names or len(names) != int(preds.shape[0]):
        names = list(xrv.datasets.default_pathologies)[: int(preds.shape[0])]
    out["preds"] = dict(zip(names, [float(p) for p in preds.tolist()]))

    pprint(out)


if __name__ == "__main__":
    main()

