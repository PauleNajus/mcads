from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
from typing import Any, Optional, Tuple
import torchxrayvision as xrv
from .model_loader import load_model, get_model_lock
from .xrv_io import load_xrv_image as _load_xrv_image
from .transforms import get_xrv_transform, get_center_crop_bounds

logger = logging.getLogger(__name__)


def _overlay_colormap_on_gray(
    gray01: np.ndarray,
    heatmap01: np.ndarray,
    *,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    gamma: float = 1.0,
) -> np.ndarray:
    """Overlay a heatmap on a grayscale image (OpenCV-friendly).

    This intentionally uses *per-pixel* alpha (derived from heatmap intensity)
    instead of a constant alpha, so low-activation regions remain mostly
    unchanged (closer to Chester's visualization style).

    Args:
        gray01: 2D grayscale image scaled to [0, 1]
        heatmap01: 2D heatmap scaled to [0, 1] (resized to match gray if needed)
        alpha: max overlay strength (0..1)
        colormap: OpenCV colormap id
        gamma: optional gamma on heatmap before alpha (>=1 focuses on peaks)

    Returns:
        BGR uint8 image (OpenCV channel order).
    """
    if gray01.ndim != 2:
        raise ValueError("gray01 must be a 2D array")

    heat = heatmap01
    if heat.shape[:2] != gray01.shape[:2]:
        heat = cv2.resize(heat, (gray01.shape[1], gray01.shape[0]))

    gray_u8 = (np.clip(gray01, 0.0, 1.0) * 255.0).astype(np.uint8)
    img_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)  # type: ignore

    heat_f = np.clip(heat.astype(np.float32), 0.0, 1.0)
    if gamma != 1.0:
        heat_f = np.power(heat_f, float(gamma))

    heat_u8 = (heat_f * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_u8, colormap)  # type: ignore

    a = (heat_f * float(alpha)).clip(0.0, 1.0).astype(np.float32)[..., None]
    out = img_bgr.astype(np.float32) * (1.0 - a) + heat_bgr.astype(np.float32) * a
    return out.astype(np.uint8)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap `NoInplaceReLU` wrappers (used for interpretability)."""
    return model.model if isinstance(model, NoInplaceReLU) else model


def _forward_xrv_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Best-effort raw-logit forward for TorchXRayVision models.

    TorchXRayVision core models may apply sigmoid + operating-point normalization
    in `forward()`. For gradient-based explanations, logits are usually more
    stable (avoid saturation and piecewise ops).
    """
    base = _unwrap_model(model)

    # TorchXRayVision DenseNet has `.features` + `.classifier`.
    if hasattr(base, "features") and hasattr(base, "classifier"):
        # Cast to Any to appease mypy's "Tensor not callable" check on dynamic attributes
        base_any: Any = base
        feats = base_any.features(x)
        feats = F.relu(feats, inplace=False)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
        return base_any.classifier(pooled)

    # TorchXRayVision ResNet wrapper stores torchvision model at `.model`.
    inner = getattr(base, "model", None)
    if inner is not None and callable(inner):
        return inner(x)

    # Fallback: best effort (may include sigmoid/op_norm depending on model).
    return base(x)


def _model_device(model: torch.nn.Module) -> torch.device:
    """Best-effort device detection for a (possibly wrapped) torch module."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _pathology_names_for_output(model: torch.nn.Module, n_outputs: int) -> list[str]:
    """Return a pathology name list that matches the model output dimension.

    TorchXRayVision models *usually* expose `model.pathologies`. When it doesn't
    match the output dimension, we fall back to `xrv.datasets.default_pathologies`.
    """
    # Keep label ordering consistent with inference.
    #
    # In `xrayapp/inference_logic.py`, ResNet outputs are *always* mapped using
    # `xrv.datasets.default_pathologies` (even if `model.pathologies` exists),
    # because some TorchXRayVision ResNet weight bundles have been observed to
    # expose a `pathologies` list that does not match the forward output order.
    #
    # If we used the wrong ordering here, GRAD-CAM / PLI could explain the
    # wrong class index (accuracy-critical bug).
    base = _unwrap_model(model)
    if base.__class__.__name__.lower().startswith("resnet"):
        return list(xrv.datasets.default_pathologies)[:n_outputs]

    model_names = list(getattr(model, "pathologies", []) or [])
    if model_names and len(model_names) == n_outputs:
        return model_names

    default_names = list(xrv.datasets.default_pathologies)
    if len(default_names) == n_outputs:
        return default_names

    # Final fallback: keep things consistent length-wise.
    if model_names:
        return model_names[:n_outputs]
    return default_names[:n_outputs]


def disable_inplace_relu(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively disables in-place ReLU operations in a model.
    Returns the modified model.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
    return model


class NoInplaceReLU(torch.nn.Module):
    """
    A wrapper module that ensures no in-place ReLU operations are performed.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Disable in-place operations in the model
        disable_inplace_relu(self.model)
        
        # Copy model attributes
        if hasattr(model, 'pathologies'):
            self.pathologies = model.pathologies
            
    def forward(self, x):
        return self.model(x)


def _prepare_interpretability(image_path: str, model_type: str) -> Tuple[torch.nn.Module, torch.Tensor, np.ndarray]:
    """Shared setup: load model, wrap it, load image, transform it.
    
    Returns:
        wrapped_model: The model wrapped in NoInplaceReLU and on the correct device.
        img_tensor: The input tensor (1, C, H, W) on the correct device.
        original_vis: The original image as a 2D numpy array [0, 1] for visualization.
    """
    # Load model via shared cache
    logger.info("Loading model for interpretability: %s", model_type)
    model, resize_dim = load_model(model_type)
    logger.info("Model loaded successfully, resize_dim: %s", resize_dim)
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Load + normalize image (supports DICOM via pydicom).
    img = _load_xrv_image(image_path)
    # Preserve original image for visualization (2D).
    original_img = img[0].copy()
    
    # IMPORTANT: Keep preprocessing consistent with inference (`xrayapp/utils.py`).
    transform = get_xrv_transform(resize_dim)
    
    # Apply transforms
    img = transform(img)
    
    # To tensor on the same device as the cached model (prevents CPU/GPU mismatch).
    device = _model_device(wrapped_model)
    img_tensor = torch.from_numpy(img).float()
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    elif len(img_tensor.shape) < 3:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    img_tensor = img_tensor.to(device)

    # Prepare original image for visualization (2D, [0, 1])
    if len(original_img.shape) == 3:
        if original_img.shape[0] == 1:
            original_vis = original_img.squeeze(0)
        elif original_img.shape[0] == 3:
            original_vis = original_img.mean(axis=0)
        else:
            # Unexpected channel count, fallback to mean if it looks like (C, H, W)
            original_vis = original_img.mean(axis=0)
    else:
        original_vis = original_img
    
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min() + 1e-8)
    
    return wrapped_model, img_tensor, original_vis


def _postprocess_and_overlay(
    heatmap: np.ndarray, 
    original_vis: np.ndarray, 
    colormap: int = cv2.COLORMAP_JET, 
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Map heatmap back to original image coords and create overlay.
    
    Args:
        heatmap: The heatmap in model-cropped space.
        original_vis: The original visualization image (2D, [0, 1]).
        colormap: OpenCV colormap to use.
        alpha: Overlay opacity.
        
    Returns:
        heatmap_full: The heatmap mapped to the full original image size (float32).
        overlaid_img: The RGB overlay image (uint8).
    """
    # Map the heatmap back into the center-crop window inside the original image
    H, W = int(original_vis.shape[0]), int(original_vis.shape[1])
    y0, y1, x0, x1 = get_center_crop_bounds(H, W)
    crop_h, crop_w = int(y1 - y0), int(x1 - x0)

    # Resize heatmap to match the *cropped* region.
    heatmap_crop = cv2.resize(heatmap, (crop_w, crop_h))
    
    # Re-normalize after interpolation
    hm_max = float(np.max(heatmap_crop))
    if hm_max > 0:
        heatmap_crop = heatmap_crop / hm_max  # type: ignore

    # Create full-size heatmap (same shape as original image).
    heatmap_full = np.zeros((H, W), dtype=np.float32)
    heatmap_full[y0:y1, x0:x1] = heatmap_crop.astype(np.float32)

    # Overlay on full image
    overlay_bgr = _overlay_colormap_on_gray(original_vis, heatmap_full, alpha=alpha, colormap=colormap, gamma=1.0)
    overlaid_img = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return heatmap_full, overlaid_img


class GradCAM:
    """
    Grad-CAM implementation for DenseNet-121 and other convolutional networks
    Based on: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Cache the underlying model for architecture inspection.
        self.base_model = _unwrap_model(model)

        # Pick a sensible default target layer (last conv-stage feature map).
        if target_layer is None:
            # TorchXRayVision DenseNet: `features.norm5` is the final conv-stage output.
            if hasattr(self.base_model, "features") and hasattr(self.base_model.features, "norm5"):
                self.target_layer = self.base_model.features.norm5
            # Fallback for DenseNet-like models.
            elif hasattr(self.base_model, "features") and hasattr(self.base_model.features, "denseblock4"):
                self.target_layer = self.base_model.features.denseblock4
            # TorchXRayVision ResNet wrapper: torchvision model is at `.model`.
            elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layer4"):
                self.target_layer = self.base_model.model.layer4[-1]
            # Raw torchvision ResNet.
            elif hasattr(self.base_model, "layer4"):
                self.target_layer = self.base_model.layer4[-1]
            else:
                logger.debug("Unable to auto-detect Grad-CAM target layer; model structure follows.")
                for name, _module in self.base_model.named_modules():
                    logger.debug("Module: %s", name)
                raise ValueError("Could not determine the target layer. Please specify explicitly.")
        else:
            self.target_layer = target_layer
        
        # Hooks populate these; guard before use
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        # Initialize to None
        self.activations = None
        self.gradients = None
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        # Register hooks to capture activations and gradients
        self.hook_forward = self.target_layer.register_forward_hook(forward_hook)
        self.hook_backward = self.target_layer.register_full_backward_hook(backward_hook)
    
    def _release_hooks(self):
        # Make hook cleanup idempotent (important in long-running web workers).
        hook_forward = getattr(self, "hook_forward", None)
        if hook_forward is not None:
            hook_forward.remove()
            self.hook_forward = None
        hook_backward = getattr(self, "hook_backward", None)
        if hook_backward is not None:
            hook_backward.remove()
            self.hook_backward = None

    def close(self) -> None:
        """Explicitly remove hooks to avoid server-side memory leaks."""
        self._release_hooks()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        # Return None (or False) to propagate exceptions.


    def _safe_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with a narrow CPU fallback for known PyTorch CPU issues."""
        try:
            return self.model(input_tensor)
        except RuntimeError as e:
            if "could not create a primitive" in str(e):
                logger.warning("PyTorch primitive error in Grad-CAM; using CPU fallback.")
                self.model = self.model.cpu()
                return self.model(input_tensor.cpu())
            raise

    def _safe_forward_logits(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns logits (no sigmoid/op_norm if possible)."""
        try:
            return _forward_xrv_logits(self.model, input_tensor)
        except RuntimeError as e:
            if "could not create a primitive" in str(e):
                logger.warning("PyTorch primitive error in Grad-CAM logits forward; using CPU fallback.")
                self.model = self.model.cpu()
                self.base_model = _unwrap_model(self.model)
                return _forward_xrv_logits(self.model, input_tensor.cpu())
            raise

    def _resolve_target_index(
        self,
        output: torch.Tensor,
        target_class: str | int | None,
    ) -> int:
        """Resolve a target class (name or index) to an output index."""
        n_outputs = int(output.shape[-1])

        if target_class is None:
            return int(torch.argmax(output).item())

        if isinstance(target_class, str):
            names = _pathology_names_for_output(self.model, n_outputs)
            # UI/display names may differ from model names (e.g. "Pleural Thickening"
            # vs TorchXRayVision's "Pleural_Thickening"). Try a few safe normalizations.
            candidates = [
                target_class,
                target_class.replace(" ", "_"),
                target_class.replace("_", " "),
            ]
            for cand in candidates:
                if cand in names:
                    return int(names.index(cand))
            # Case-insensitive fallback.
            lower_map = {n.lower(): i for i, n in enumerate(names)}
            for cand in candidates:
                idx = lower_map.get(cand.lower())
                if idx is not None:
                    return int(idx)
            raise ValueError(f"Pathology {target_class} not found in model outputs.")

        idx = int(target_class)
        if not (0 <= idx < n_outputs):
            raise ValueError(f"Target class index {idx} is out of range for {n_outputs} outputs.")
        return idx
    
    def get_heatmap(self, input_tensor, target_class=None):
        """
        Generate class activation heatmap
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            target_class: Index of the target class (if None, uses the class with maximum score)
        
        Returns:
            heatmap: Numpy array representing the heatmap
            output: Model output logits
        """
        # Forward pass (prefer logits to avoid sigmoid/op-norm saturation).
        self.model.zero_grad()
        logits = self._safe_forward_logits(input_tensor)
        target_idx = self._resolve_target_index(logits, target_class)

        # Backward pass (calculate gradients)
        self.model.zero_grad()
        logits[0, target_idx].backward()
        
        # Check if gradients and activations were captured
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check if target layer is correct.")
        if self.activations is None:
            raise RuntimeError("Activations not captured. Check if target layer is correct.")
        
        # Calculate weights based on global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))

        # Weight activation maps with gradients (vectorized; faster than Python loops)
        activations = self.activations
        assert activations is not None
        heatmap_t = torch.sum(
            activations * pooled_gradients[None, :, None, None],
            dim=1,
        ).squeeze()
        heatmap = heatmap_t.detach().cpu().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        else:
            # If heatmap is all zeros, create a fallback minimal heatmap
            logger.warning("Grad-CAM heatmap is all zeros; using a minimal fallback.")
            heatmap = np.ones_like(heatmap) * 0.1
        
        return heatmap, logits.detach()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            img: Original image (2D numpy array)
            heatmap: Heatmap from get_heatmap (2D numpy array)
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            overlaid_img: Image with overlaid heatmap
        """
        # Keep low-activation pixels mostly transparent (per-pixel alpha).
        # This avoids the full-image blue tint from constant-alpha overlays.
        return _overlay_colormap_on_gray(img, heatmap, alpha=alpha, colormap=colormap, gamma=1.0)
    
    def __del__(self):
        # Best-effort cleanup; prefer explicit `close()` via context manager.
        try:
            self.close()
        except Exception:
            pass
    
    def get_combined_heatmap(self, input_tensor, probability_threshold=0.5):
        """
        Generate a combined class activation heatmap for all pathologies above threshold
        
        Args:
            input_tensor: Input tensor of shape (1, C, H, W)
            probability_threshold: Minimum probability threshold (default: 0.5)
        
        Returns:
            combined_heatmap: Numpy array representing the combined heatmap
            selected_pathologies: List of pathology names and their probabilities above threshold
            output: Model output logits
        """
        # Single forward pass on logits (reuse graph for all backprops below).
        # Using logits avoids sigmoid/op-norm saturation in gradients.
        self.model.zero_grad()
        logits = self._safe_forward_logits(input_tensor)

        # Probabilities used only for thresholding/weighting.
        with torch.no_grad():
            probs_t = torch.sigmoid(logits)
            # Match TorchXRayVision `forward()` semantics when available (op_norm).
            # This keeps threshold-based selection consistent with what users see.
            base = _unwrap_model(self.model)
            op_norm = getattr(base, "op_norm", None)
            if callable(op_norm):
                try:
                    probs_t = op_norm(probs_t)
                except Exception:
                    # If op_norm isn't compatible for any reason, fall back to sigmoid.
                    pass
            probabilities = probs_t.squeeze().detach().cpu().numpy()

        pathology_names = _pathology_names_for_output(self.model, int(probabilities.shape[0]))
        # ResNet weight bundles in TorchXRayVision include a couple of labels that
        # are not trained and tend to sit at ~0.5 probability, which can dominate
        # threshold-based "combined" visualizations. Inference explicitly excludes
        # these labels; do the same here to keep explanations accurate.
        excluded_names: set[str] = set()
        try:
            base = _unwrap_model(self.model)
            if base.__class__.__name__.lower().startswith("resnet"):
                excluded_names = {
                    "Enlarged Cardiomediastinum",
                    "Enlarged_Cardiomediastinum",
                    "Lung Lesion",
                    "Lung_Lesion",
                }
        except Exception:
            excluded_names = set()
        
        # Find pathologies above threshold
        selected_pathologies = []
        selected_indices = []
        
        for i, (pathology, prob) in enumerate(zip(pathology_names, probabilities)):
            if pathology in excluded_names:
                continue
            if prob >= probability_threshold:
                selected_pathologies.append((pathology, prob))
                selected_indices.append(i)
        
        if not selected_indices:
            logger.info("No pathologies above threshold %.3f; using top-1 fallback.", probability_threshold)
            candidate_indices = [i for i, name in enumerate(pathology_names) if name not in excluded_names]
            if not candidate_indices:
                candidate_indices = list(range(int(probabilities.shape[0])))
            top_indices = np.array(candidate_indices, dtype=int)[np.argsort(probabilities[candidate_indices])[-3:][::-1]]
            for idx in top_indices:
                logger.debug("Top pathology candidate: %s=%.3f", pathology_names[idx], probabilities[idx])
            # Use the top pathology if none above threshold
            selected_indices = [top_indices[0]]
            selected_pathologies = [(pathology_names[top_indices[0]], probabilities[top_indices[0]])]
        
        logger.debug("Selected pathologies above %.3f threshold:", probability_threshold)
        for pathology, prob in selected_pathologies:
            logger.debug("  %s: %.3f", pathology, prob)
        
        # Generate combined heatmap
        combined_heatmap: np.ndarray | None = None

        for i, target_class in enumerate(selected_indices):
            # Reset gradients for each pathology
            self.model.zero_grad()

            # Backward pass for this specific pathology (reuse the same `logits` graph)
            logits[0, int(target_class)].backward(retain_graph=(i < len(selected_indices) - 1))
            
            # Calculate weights based on global average pooling of gradients
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients/activations not captured. Ensure backward hooks ran.")
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))

            activations = self.activations
            assert activations is not None
            heatmap_t = torch.sum(
                activations * pooled_gradients[None, :, None, None],
                dim=1,
            ).squeeze()
            heatmap = heatmap_t.detach().cpu().numpy()
            
            # ReLU on heatmap
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
            
            # Weight the heatmap by the probability
            pathology_prob = selected_pathologies[i][1]
            weighted_heatmap = heatmap * pathology_prob
            
            # Combine with existing heatmap
            if combined_heatmap is None:
                combined_heatmap = weighted_heatmap
            else:
                combined_heatmap += weighted_heatmap
        
        # Normalize the combined heatmap
        assert combined_heatmap is not None
        if np.max(combined_heatmap) > 0:
            combined_heatmap = combined_heatmap / np.max(combined_heatmap)
        
        return combined_heatmap, selected_pathologies, logits.detach()


class PixelLevelInterpretability:
    """
    Pixel-Level Interpretability (PLI) for chest X-ray analysis
    
    Uses guided backpropagation to generate high-resolution interpretable visualizations
    showing which pixels are most influential for each pathology prediction.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.handles = []
        
        # Set all ReLU operations to non-inplace before registering hooks
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.inplace = False
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for guided backpropagation"""
        def hook_fn(module, grad_in, grad_out):
            # Only pass positive gradients for positive activations
            if isinstance(grad_in, tuple) and len(grad_in) > 0 and grad_in[0] is not None:
                # Create a new tensor instead of modifying in-place
                pos_grad = torch.clone(grad_in[0])
                pos_grad[pos_grad < 0] = 0  # Zero out negative gradients without in-place operations
                return (pos_grad,) + grad_in[1:]
            return grad_in
        
        # Register hooks for all ReLU layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                self.handles.append(module.register_full_backward_hook(hook_fn))
    
    def _release_hooks(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def generate_saliency(self, input_tensor, target_class=None):
        """
        Generate a saliency map for the given input tensor
        
        Args:
            input_tensor: Input tensor with shape (1, C, H, W)
            target_class: Index of the target class
        
        Returns:
            saliency_map: Numpy array representing the saliency map
            output: Model output logits
        """
        # Create a copy of the input tensor that requires gradients
        input_grad = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass (prefer logits for stable gradients).
        self.model.zero_grad()
        logits = _forward_xrv_logits(self.model, input_grad)

        # Resolve target index (name or index).
        n_outputs = int(logits.shape[-1])
        if target_class is None:
            target_idx = int(torch.argmax(logits).item())
        elif isinstance(target_class, str):
            names = _pathology_names_for_output(self.model, n_outputs)
            candidates = [
                target_class,
                target_class.replace(" ", "_"),
                target_class.replace("_", " "),
            ]
            for cand in candidates:
                if cand in names:
                    target_idx = int(names.index(cand))
                    break
            else:
                lower_map = {n.lower(): i for i, n in enumerate(names)}
                idx = None
                for cand in candidates:
                    idx = lower_map.get(cand.lower())
                    if idx is not None:
                        break
                if idx is None:
                    raise ValueError(f"Pathology {target_class} not found in model outputs.")
                target_idx = int(idx)
        else:
            target_idx = int(target_class)
            if not (0 <= target_idx < n_outputs):
                raise ValueError(f"Target class index {target_idx} is out of range for {n_outputs} outputs.")

        # Backward pass
        self.model.zero_grad()
        logits[0, target_idx].backward()
        
        # Get gradients of the input
        grad_data = input_grad.grad.data.clone()  # Clone the gradient data
        
        # Use absolute value of gradients for better visualization
        saliency_map = grad_data.abs().squeeze().cpu().numpy()
        
        # Apply basic Gaussian blur to reduce noise
        saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)  # type: ignore
        
        # Normalize saliency map to [0, 1]
        if np.max(saliency_map) > 0:
            saliency_map = saliency_map / np.max(saliency_map)
        
        return saliency_map, logits.detach()
    
    def apply_smoothgrad(self, input_tensor, target_class=None, n_samples=8, noise_level=0.1):
        """
        Apply SmoothGrad technique to reduce visual noise in saliency maps
        
        Args:
            input_tensor: Input tensor with shape (1, C, H, W)
            target_class: Index of the target class
            n_samples: Number of noisy samples to generate
            noise_level: Standard deviation of Gaussian noise
        
        Returns:
            smooth_saliency: Smoothed saliency map
            output: Model output logits from the original input
        """
        # Get the original output logits (used only for selecting the target class).
        with torch.no_grad():
            output = _forward_xrv_logits(self.model, input_tensor.clone())

        # Resolve target index (name or index).
        n_outputs = int(output.shape[-1])
        if target_class is None:
            target_idx = int(torch.argmax(output).item())
        elif isinstance(target_class, str):
            names = _pathology_names_for_output(self.model, n_outputs)
            candidates = [
                target_class,
                target_class.replace(" ", "_"),
                target_class.replace("_", " "),
            ]
            for cand in candidates:
                if cand in names:
                    target_idx = int(names.index(cand))
                    break
            else:
                lower_map = {n.lower(): i for i, n in enumerate(names)}
                idx = None
                for cand in candidates:
                    idx = lower_map.get(cand.lower())
                    if idx is not None:
                        break
                if idx is None:
                    raise ValueError(f"Pathology {target_class} not found in model outputs.")
                target_idx = int(idx)
        else:
            target_idx = int(target_class)
            if not (0 <= target_idx < n_outputs):
                raise ValueError(f"Target class index {target_idx} is out of range for {n_outputs} outputs.")

        # Calculate the standard deviation for Gaussian noise.
        # NOTE: Inputs are in [-1024, 1024]. Using (max-min) makes noise enormous.
        # Using std(input) keeps noise scale reasonable and stable across images.
        stdev = float(noise_level) * float(input_tensor.detach().std().item() or 1.0)

        # Initialize an empty saliency map
        smooth_saliency = np.zeros_like(input_tensor.squeeze().cpu().numpy(), dtype=np.float32)
        
        # Generate multiple noisy samples and average their saliency maps
        for _ in range(n_samples):
            # Generate Gaussian noise
            noise = torch.normal(0, stdev, size=input_tensor.shape, device=input_tensor.device)
            
            # Add noise to the input (create a new tensor rather than modifying in-place)
            noisy_input = (input_tensor.clone() + noise).clamp(-1024.0, 1024.0)
            
            # Compute saliency map for the noisy input
            saliency, _ = self.generate_saliency(noisy_input, target_idx)
            
            # Add to the accumulated saliency map
            smooth_saliency = smooth_saliency + saliency
        
        # Average the saliency maps
        smooth_saliency = smooth_saliency / n_samples
        
        # Apply simple Gaussian blur for smoothing
        smooth_saliency = cv2.GaussianBlur(smooth_saliency, (3, 3), 0)  # type: ignore
        
        # Simple contrast enhancement
        if np.max(smooth_saliency) > 0:
            smooth_saliency = smooth_saliency / np.max(smooth_saliency)
        
        return smooth_saliency, output
    
    def __del__(self):
        # Release hooks when object is deleted
        self._release_hooks()
    
    def get_combined_saliency(self, input_tensor, probability_threshold=0.5, use_smoothgrad=True, n_samples=8, noise_level=0.1):
        """
        Generate a combined saliency map for all pathologies above threshold
        
        Args:
            input_tensor: Input tensor with shape (1, C, H, W)
            probability_threshold: Minimum probability threshold (default: 0.5)
            use_smoothgrad: Whether to use SmoothGrad technique (default: True)
            n_samples: Number of samples for SmoothGrad (default: 15)
            noise_level: Noise level for SmoothGrad (default: 0.1)
        
        Returns:
            combined_saliency: Numpy array representing the combined saliency map
            selected_pathologies: List of pathology names and their probabilities above threshold
            output: Model output probabilities (TorchXRayVision calibrated)
        """
        # Forward pass to get predictions
        with torch.no_grad():
            output = self.model(input_tensor.clone())
        
        # TorchXRayVision core models already output calibrated probabilities in [0, 1].
        probabilities = output.squeeze().cpu().detach().numpy()
        
        pathology_names = _pathology_names_for_output(self.model, int(probabilities.shape[0]))
        # Keep behavior consistent with inference: some ResNet bundles include
        # labels that are effectively unsupported (sit around 0.5). Exclude them
        # from combined explanations to avoid misleading highlights.
        excluded_names: set[str] = set()
        try:
            base = _unwrap_model(self.model)
            if base.__class__.__name__.lower().startswith("resnet"):
                excluded_names = {
                    "Enlarged Cardiomediastinum",
                    "Enlarged_Cardiomediastinum",
                    "Lung Lesion",
                    "Lung_Lesion",
                }
        except Exception:
            excluded_names = set()
        
        # Find pathologies above threshold
        selected_pathologies = []
        selected_indices = []
        
        for i, (pathology, prob) in enumerate(zip(pathology_names, probabilities)):
            if pathology in excluded_names:
                continue
            if prob >= probability_threshold:
                selected_pathologies.append((pathology, prob))
                selected_indices.append(i)
        
        if not selected_indices:
            logger.info("No pathologies above threshold %.3f; using top-1 fallback.", probability_threshold)
            candidate_indices = [i for i, name in enumerate(pathology_names) if name not in excluded_names]
            if not candidate_indices:
                candidate_indices = list(range(int(probabilities.shape[0])))
            top_indices = np.array(candidate_indices, dtype=int)[np.argsort(probabilities[candidate_indices])[-3:][::-1]]
            for idx in top_indices:
                logger.debug("Top pathology candidate: %s=%.3f", pathology_names[idx], probabilities[idx])
            # Use the top pathology if none above threshold
            selected_indices = [top_indices[0]]
            selected_pathologies = [(pathology_names[top_indices[0]], probabilities[top_indices[0]])]
        
        logger.debug("Selected pathologies above %.3f threshold:", probability_threshold)
        for pathology, prob in selected_pathologies:
            logger.debug("  %s: %.3f", pathology, prob)
        
        # Generate combined saliency map
        combined_saliency = None
        
        for i, target_class in enumerate(selected_indices):
            # Generate saliency map for this specific pathology
            if use_smoothgrad:
                saliency_map, _ = self.apply_smoothgrad(input_tensor, target_class, n_samples, noise_level)
            else:
                saliency_map, _ = self.generate_saliency(input_tensor, target_class)
            
            # Weight the saliency map by the probability
            pathology_prob = selected_pathologies[i][1]
            weighted_saliency = saliency_map * pathology_prob
            
            # Combine with existing saliency map
            if combined_saliency is None:
                combined_saliency = weighted_saliency
            else:
                combined_saliency += weighted_saliency
        
        # Normalize the combined saliency map
        assert combined_saliency is not None
        if np.max(combined_saliency) > 0:
            combined_saliency = combined_saliency / np.max(combined_saliency)
        
        return combined_saliency, selected_pathologies, output


def apply_gradcam(
    image_path: str,
    model_type: str = 'densenet',
    target_class: str | int | None = None,
) -> dict[str, Any]:
    """
    Apply Grad-CAM to an X-ray image
    """
    # Ensure deterministic results when multiple jobs run on the same cached model.
    with get_model_lock(model_type):
        try:
            # Shared setup
            wrapped_model, img_tensor, original_vis = _prepare_interpretability(image_path, model_type)

            # Initialize Grad-CAM and generate heatmap
            with GradCAM(wrapped_model) as gradcam:
                heatmap, output = gradcam.get_heatmap(img_tensor, target_class)

            # Resolve target name
            names = _pathology_names_for_output(wrapped_model, int(output.shape[-1]))
            if target_class is None:
                target_idx = int(torch.argmax(output, dim=-1).item())
                target_name: str | int = names[target_idx] if 0 <= target_idx < len(names) else target_idx
            elif isinstance(target_class, str):
                target_name = target_class
            else:
                idx = int(target_class)
                target_name = names[idx] if 0 <= idx < len(names) else idx
            
            # Shared postprocessing
            heatmap_full, overlaid_img = _postprocess_and_overlay(heatmap, original_vis)
            
            return {
                'original': original_vis,
                'heatmap': heatmap_full,
                'overlay': overlaid_img,
                'target_class': target_name
            }
            
        except Exception:
            logger.exception("Failed to apply Grad-CAM")
            raise


def apply_combined_gradcam(
    image_path: str,
    model_type: str = 'densenet',
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Apply combined interpretability to an X-ray image for all pathologies above threshold
    """
    with get_model_lock(model_type):
        try:
            # Shared setup
            wrapped_model, img_tensor, original_vis = _prepare_interpretability(image_path, model_type)
            
            # Initialize Grad-CAM and generate combined heatmap
            with GradCAM(wrapped_model) as gradcam:
                combined_heatmap, selected_pathologies, _predictions = gradcam.get_combined_heatmap(
                    img_tensor, probability_threshold=probability_threshold
                )
            
            # Shared postprocessing
            heatmap_full, overlaid_img = _postprocess_and_overlay(combined_heatmap, original_vis)
            
            # Create pathology summary string
            pathology_summary = ", ".join([f"{name} ({prob:.3f})" for name, prob in selected_pathologies])
            
            return {
                'original': original_vis,
                'heatmap': heatmap_full,
                'overlay': overlaid_img,
                'selected_pathologies': selected_pathologies,
                'pathology_summary': pathology_summary,
                'method': 'combined_gradcam',
                'threshold': probability_threshold
            }

        except Exception:
            logger.exception("Failed to apply combined Grad-CAM")
            raise


def apply_pixel_interpretability(
    image_path: str,
    model_type: str = 'densenet',
    target_class: str | int | None = None,
    use_smoothgrad: bool = True,
) -> dict[str, Any]:
    """
    Apply Pixel-Level Interpretability to an X-ray image
    """
    with get_model_lock(model_type):
        pli: PixelLevelInterpretability | None = None
        try:
            requested_target_class = target_class

            # Shared setup
            wrapped_model, img_tensor, original_vis = _prepare_interpretability(image_path, model_type)
            
            # If target_class is not provided, use the class with the highest probability
            # Note: We need a forward pass to determine this if not provided.
            if target_class is None:
                wrapped_model.eval()
                with torch.no_grad():
                    preds = wrapped_model(img_tensor)
                
                pred_idx = int(torch.argmax(preds).item())
                # Use shared helper to resolve pathology name
                n_outputs = int(preds.shape[-1])
                names = _pathology_names_for_output(wrapped_model, n_outputs)
                target_class = names[pred_idx] if 0 <= pred_idx < len(names) else str(pred_idx)
            
            # Normalize target_class string if needed
            elif isinstance(target_class, str):
                candidates = [target_class, target_class.replace(" ", "_"), target_class.replace("_", " ")]
                if model_type == 'resnet':
                    for cand in candidates:
                        if cand in xrv.datasets.default_pathologies:
                            target_class = cand
                            break
                else:
                    model_names = list(getattr(wrapped_model, "pathologies", []) or [])
                    for cand in candidates:
                        if cand in model_names:
                            target_class = cand
                            break

            # Initialize PixelLevelInterpretability
            pli = PixelLevelInterpretability(wrapped_model)
            
            # Generate saliency map
            if use_smoothgrad:
                saliency_map, _ = pli.apply_smoothgrad(img_tensor, target_class)
            else:
                saliency_map, _ = pli.generate_saliency(img_tensor, target_class)
            
            # Shared postprocessing (saliency_map is effectively a heatmap here)
            saliency_full, overlaid_img = _postprocess_and_overlay(saliency_map, original_vis, alpha=0.6)

            # Create colored version of saliency map (specific to PLI visualization needs)
            saliency_colored_bgr = cv2.applyColorMap((saliency_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
            saliency_colored = cv2.cvtColor(saliency_colored_bgr, cv2.COLOR_BGR2RGB)

            # Keep the user-visible label stable
            target_display = requested_target_class if isinstance(requested_target_class, str) else target_class
            
            return {
                'original': original_vis,
                'saliency_map': saliency_full,
                'saliency_colored': saliency_colored,
                'overlay': overlaid_img,
                'target_class': target_display,
                'method': 'pli'
            }

        except Exception:
            logger.exception("Error generating pixel interpretability")
            raise
        finally:
            # Always remove hooks deterministically; relying on GC/__del__ is risky
            # on long-running servers and can pollute subsequent Grad-CAM results.
            if pli is not None:
                try:
                    pli._release_hooks()
                except Exception:
                    pass


def apply_combined_pixel_interpretability(
    image_path: str,
    model_type: str = 'densenet',
    probability_threshold: float = 0.5,
    use_smoothgrad: bool = True,
) -> dict[str, Any]:
    """
    Apply combined Pixel-Level Interpretability to an X-ray image
    """
    with get_model_lock(model_type):
        pli: PixelLevelInterpretability | None = None
        try:
            # Shared setup
            wrapped_model, img_tensor, original_vis = _prepare_interpretability(image_path, model_type)
            
            # Initialize Pixel-Level Interpretability
            pli = PixelLevelInterpretability(wrapped_model)
            
            # Get combined saliency map
            combined_saliency, selected_pathologies, predictions = pli.get_combined_saliency(
                img_tensor, probability_threshold=probability_threshold, use_smoothgrad=use_smoothgrad
            )
            
            # Shared postprocessing
            saliency_full, overlaid_img = _postprocess_and_overlay(combined_saliency, original_vis, alpha=0.4)

            # Colored saliency
            saliency_colored_bgr = cv2.applyColorMap((saliency_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
            saliency_colored = cv2.cvtColor(saliency_colored_bgr, cv2.COLOR_BGR2RGB)
            
            # Create pathology summary string
            pathology_summary = ", ".join([f"{name} ({prob:.3f})" for name, prob in selected_pathologies])
            
            return {
                'original': original_vis,
                'saliency_map': saliency_full,
                'saliency_colored': saliency_colored,
                'overlay': overlaid_img,
                'selected_pathologies': selected_pathologies,
                'pathology_summary': pathology_summary,
                'method': 'combined_pli',
                'threshold': probability_threshold
            }

        except Exception:
            logger.exception("Failed to apply combined PLI")
            raise
        finally:
            if pli is not None:
                try:
                    pli._release_hooks()
                except Exception:
                    pass
