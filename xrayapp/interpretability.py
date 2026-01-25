from __future__ import annotations

import logging
import numpy as np
import torch
import torchvision
import skimage
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
from typing import Any, Optional
import torchxrayvision as xrv
from .model_loader import load_model

logger = logging.getLogger(__name__)


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
        

class GradCAM:
    """
    Grad-CAM implementation for DenseNet-121 and other convolutional networks
    Based on: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Check if model is wrapped with NoInplaceReLU
        actual_model = model.model if isinstance(model, NoInplaceReLU) else model
        
        # For DenseNet-121, the default target layer is the last convolutional layer
        if target_layer is None:
            # TorchXRayVision DenseNet models have features directly accessible
            if hasattr(actual_model, 'features') and hasattr(actual_model.features, 'denseblock4'):
                # For DenseNet models - use the last conv layer before the classifier
                if hasattr(actual_model.features.denseblock4, 'denselayer16'):
                    if hasattr(actual_model.features.denseblock4.denselayer16, 'conv2'):
                        self.target_layer = actual_model.features.denseblock4.denselayer16.conv2
                    else:
                        # Fallback to relu2 if conv2 doesn't exist
                        self.target_layer = actual_model.features.denseblock4.denselayer16.relu2
                else:
                    # If denselayer16 doesn't exist, use the transition3 conv layer
                    self.target_layer = actual_model.features.transition3.conv
            elif hasattr(actual_model, 'model'):
                # Some models might have a nested .model attribute
                inner_model = actual_model.model
                if hasattr(inner_model, 'features') and hasattr(inner_model.features, 'denseblock4'):
                    # For nested DenseNet models
                    if hasattr(inner_model.features.denseblock4, 'denselayer16'):
                        if hasattr(inner_model.features.denseblock4.denselayer16, 'conv2'):
                            self.target_layer = inner_model.features.denseblock4.denselayer16.conv2
                        else:
                            self.target_layer = inner_model.features.denseblock4.denselayer16.relu2
                    else:
                        self.target_layer = inner_model.features.transition3.conv
                elif hasattr(inner_model, 'layer4'):
                    # For ResNet models
                    self.target_layer = inner_model.layer4[-1]
                else:
                    raise ValueError("Could not determine target layer for inner model")
            elif hasattr(actual_model, 'layer4'):
                # Direct ResNet model
                self.target_layer = actual_model.layer4[-1]
            else:
                # Log model structure to help with debugging (avoid stdout in production).
                logger.debug("Unable to auto-detect Grad-CAM target layer; model structure follows.")
                for name, _module in actual_model.named_modules():
                    logger.debug("Module: %s", name)
                    if 'conv' in name.lower() or 'layer' in name.lower():
                        logger.debug("Potential target: %s", name)
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

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

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
            try:
                return int(names.index(target_class))
            except ValueError as e:
                raise ValueError(f"Pathology {target_class} not found in model outputs.") from e

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
        # Forward pass
        self.model.zero_grad()
        output = self._safe_forward(input_tensor)
        target_idx = self._resolve_target_index(output, target_class)
        
        # Backward pass (calculate gradients)
        one_hot = torch.zeros_like(output)
        one_hot[0, target_idx] = 1
        output.backward(gradient=one_hot)
        
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
        heatmap_t = torch.mean(
            activations * pooled_gradients[None, :, None, None],
            dim=1,
        ).squeeze()
        heatmap = heatmap_t.cpu().detach().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        else:
            # If heatmap is all zeros, create a fallback minimal heatmap
            logger.warning("Grad-CAM heatmap is all zeros; using a minimal fallback.")
            heatmap = np.ones_like(heatmap) * 0.1
        
        return heatmap, output.detach()
    
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
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)  # type: ignore
        
        # Convert grayscale image to RGB if needed
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2RGB)  # type: ignore
        else:
            img_rgb = np.uint8(img * 255)
        
        # Overlay heatmap on original image
        overlaid_img = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_colored, alpha, 0)  # type: ignore
        
        return overlaid_img
    
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
        # Single forward pass (we reuse `output` for all backprops below).
        self.model.zero_grad()
        output = self._safe_forward(input_tensor)

        # Convert output to probabilities using sigmoid (multi-label); no graph needed here.
        with torch.no_grad():
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()

        pathology_names = _pathology_names_for_output(self.model, int(probabilities.shape[0]))
        
        # Find pathologies above threshold
        selected_pathologies = []
        selected_indices = []
        
        for i, (pathology, prob) in enumerate(zip(pathology_names, probabilities)):
            if prob >= probability_threshold:
                selected_pathologies.append((pathology, prob))
                selected_indices.append(i)
        
        if not selected_indices:
            logger.info("No pathologies above threshold %.3f; using top-1 fallback.", probability_threshold)
            top_indices = np.argsort(probabilities)[-3:][::-1]
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

            # Backward pass for this specific pathology (reuse the same `output` graph)
            one_hot = torch.zeros_like(output)
            one_hot[0, int(target_class)] = 1
            output.backward(gradient=one_hot, retain_graph=(i < len(selected_indices) - 1))
            
            # Calculate weights based on global average pooling of gradients
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients/activations not captured. Ensure backward hooks ran.")
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))

            activations = self.activations
            assert activations is not None
            heatmap_t = torch.mean(
                activations * pooled_gradients[None, :, None, None],
                dim=1,
            ).squeeze()
            heatmap = heatmap_t.cpu().detach().numpy()
            
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
        
        return combined_heatmap, selected_pathologies, output.detach()


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
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_grad)
        
        # If target_class is None, get class with highest score
        if target_class is None:
            target_class = torch.argmax(output).item()
        elif isinstance(target_class, str):
            # If target_class is a string (pathology name), get its index
            # For ResNet model, use default_pathologies for correct mapping
            if hasattr(self.model, 'pathologies') and 'Lung Opacity' in self.model.pathologies:
                # This is ResNet model, use default_pathologies for indexing
                if target_class in xrv.datasets.default_pathologies:
                    target_class = xrv.datasets.default_pathologies.index(target_class)
                else:
                    raise ValueError(f"Pathology {target_class} not found in default pathologies.")
            elif target_class in self.model.pathologies:
                target_class = self.model.pathologies.index(target_class)
            else:
                raise ValueError(f"Pathology {target_class} not found in model.")
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        target_idx: int = int(target_class)
        one_hot[0, target_idx] = 1
        output.backward(gradient=one_hot)
        
        # Get gradients of the input
        grad_data = input_grad.grad.data.clone()  # Clone the gradient data
        
        # Use absolute value of gradients for better visualization
        saliency_map = grad_data.abs().squeeze().cpu().numpy()
        
        # Apply basic Gaussian blur to reduce noise
        saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)  # type: ignore
        
        # Normalize saliency map to [0, 1]
        if np.max(saliency_map) > 0:
            saliency_map = saliency_map / np.max(saliency_map)
        
        return saliency_map, output
    
    def apply_smoothgrad(self, input_tensor, target_class=None, n_samples=15, noise_level=0.1):
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
        # Get the original output for the input
        with torch.no_grad():
            output = self.model(input_tensor.clone())
        
        # Determine target class if not provided
        if target_class is None:
            target_class = torch.argmax(output).item()
        elif isinstance(target_class, str):
            # If target_class is a string (pathology name), get its index
            # For ResNet model, use default_pathologies for correct mapping
            if hasattr(self.model, 'pathologies') and 'Lung Opacity' in self.model.pathologies:
                # This is ResNet model, use default_pathologies for indexing
                if target_class in xrv.datasets.default_pathologies:
                    target_class = xrv.datasets.default_pathologies.index(target_class)
                else:
                    raise ValueError(f"Pathology {target_class} not found in default pathologies.")
            elif target_class in self.model.pathologies:
                target_class = self.model.pathologies.index(target_class)
            else:
                raise ValueError(f"Pathology {target_class} not found in model.")
        
        # Calculate the standard deviation for Gaussian noise
        stdev = noise_level * (torch.max(input_tensor) - torch.min(input_tensor)).item()
        
        # Initialize an empty saliency map
        smooth_saliency = np.zeros_like(input_tensor.squeeze().cpu().numpy())
        
        # Generate multiple noisy samples and average their saliency maps
        for _ in range(n_samples):
            # Generate Gaussian noise
            noise = torch.normal(0, stdev, size=input_tensor.shape, device=input_tensor.device)
            
            # Add noise to the input (create a new tensor rather than modifying in-place)
            noisy_input = input_tensor.clone() + noise
            
            # Compute saliency map for the noisy input
            saliency, _ = self.generate_saliency(noisy_input, target_class)
            
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
    
    def get_combined_saliency(self, input_tensor, probability_threshold=0.5, use_smoothgrad=True, n_samples=15, noise_level=0.1):
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
            output: Model output logits
        """
        # Forward pass to get predictions
        with torch.no_grad():
            output = self.model(input_tensor.clone())
        
        # Convert output to probabilities using sigmoid (for multi-label classification)
        probabilities = torch.sigmoid(output).squeeze().cpu().detach().numpy()
        
        # Determine pathology names based on model type
        if hasattr(self.model, 'pathologies') and 'Lung Opacity' in self.model.pathologies:
            # This is ResNet model, use default_pathologies for correct mapping
            pathology_names = xrv.datasets.default_pathologies
        else:
            # This is DenseNet or other model
            pathology_names = self.model.pathologies
        
        # Find pathologies above threshold
        selected_pathologies = []
        selected_indices = []
        
        for i, (pathology, prob) in enumerate(zip(pathology_names, probabilities)):
            if prob >= probability_threshold:
                selected_pathologies.append((pathology, prob))
                selected_indices.append(i)
        
        if not selected_indices:
            logger.info("No pathologies above threshold %.3f; using top-1 fallback.", probability_threshold)
            top_indices = np.argsort(probabilities)[-3:][::-1]
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
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        target_class: Target class for Grad-CAM visualization
        
    Returns:
        Dictionary with visualization results
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Load model via shared cache
        logger.info(f"Loading model for Grad-CAM: {model_type}")
        model, resize_dim = load_model(model_type)
        logger.info(f"Model loaded successfully, resize_dim: {resize_dim}")
    except Exception as e:
        logger.error(f"Failed to load model for Grad-CAM: {e}")
        raise
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Load and preprocess the image
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    
    # Preserve original image for visualization
    original_img = img.copy()
    
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]  # Take first channel instead of averaging
    if len(img.shape) < 2:
        raise ValueError("Input image must have at least 2 dimensions")
    
    # Add channel dimension
    img = img[None, :, :]
    
    # IMPORTANT: Keep preprocessing consistent with inference (`xrayapp/utils.py`).
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(resize_dim),
    ])
    
    # Apply transforms
    img = transform(img)
    
    # To tensor on the same device as the cached model (prevents CPU/GPU mismatch).
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(_model_device(wrapped_model))

    # Initialize Grad-CAM (auto-detect target layer) and generate heatmap.
    with GradCAM(wrapped_model) as gradcam:
        heatmap, output = gradcam.get_heatmap(img_tensor, target_class)

    # Return a stable, user-facing pathology name (even if caller passed an index).
    names = _pathology_names_for_output(wrapped_model, int(output.shape[-1]))
    if target_class is None:
        target_idx = int(torch.argmax(output, dim=-1).item())
        target_name: str | int = names[target_idx] if 0 <= target_idx < len(names) else target_idx
    elif isinstance(target_class, str):
        target_name = target_class
    else:
        idx = int(target_class)
        target_name = names[idx] if 0 <= idx < len(names) else idx
    
    # Convert original image to a suitable format for visualization
    original_vis = original_img
    if len(original_vis.shape) > 2:
        original_vis = original_vis[:, :, 0]  # Take first channel for visualization
    
    # Scale to [0, 1] for visualization
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min() + 1e-8)
    
    # Overlay heatmap on original image
    overlaid_img = gradcam.overlay_heatmap(original_vis, heatmap)
    
    # Return results
    return {
        'original': original_vis,
        'heatmap': heatmap,
        'overlay': overlaid_img,
        'target_class': target_name
    }


def apply_combined_gradcam(
    image_path: str,
    model_type: str = 'densenet',
    probability_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Apply combined interpretability to an X-ray image for all pathologies above threshold
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        probability_threshold: Minimum probability threshold (default: 0.5)
        
    Returns:
        Dictionary with combined visualization results
    """
    # Load model via shared cache
    model, resize_dim = load_model(model_type)
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Load and preprocess the image
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    
    # Preserve original image for visualization
    original_img = img.copy()
    
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]  # Take first channel instead of averaging
    if len(img.shape) < 2:
        raise ValueError("Input image must have at least 2 dimensions")
    
    # Add channel dimension
    img = img[None, :, :]
    
    # IMPORTANT: Keep preprocessing consistent with inference (`xrayapp/utils.py`).
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(resize_dim),
    ])
    
    # Apply transforms
    img = transform(img)
    
    # To tensor on the same device as the cached model (prevents CPU/GPU mismatch).
    img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(_model_device(wrapped_model))

    # Initialize Grad-CAM (auto-detect target layer) and generate heatmap.
    with GradCAM(wrapped_model) as gradcam:
        combined_heatmap, selected_pathologies, _predictions = gradcam.get_combined_heatmap(
            img_tensor, probability_threshold=probability_threshold
        )
    
    # Convert original image to a suitable format for visualization
    original_vis = original_img
    if len(original_vis.shape) > 2:
        original_vis = original_vis[:, :, 0]  # Take first channel for visualization
    
    # Scale to [0, 1] for visualization
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min() + 1e-8)
    
    # Overlay heatmap on original image
    overlaid_img = gradcam.overlay_heatmap(original_vis, combined_heatmap)
    
    # Create pathology summary string
    pathology_summary = ", ".join([f"{name} ({prob:.3f})" for name, prob in selected_pathologies])
    
    # Return results
    return {
        'original': original_vis,
        'heatmap': combined_heatmap,
        'overlay': overlaid_img,
        'selected_pathologies': selected_pathologies,
        'pathology_summary': pathology_summary,
        'method': 'combined_gradcam',
        'threshold': probability_threshold
    }


def apply_pixel_interpretability(
    image_path: str,
    model_type: str = 'densenet',
    target_class: str | int | None = None,
    use_smoothgrad: bool = True,
) -> dict[str, Any]:
    """
    Apply Pixel-Level Interpretability to an X-ray image
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        target_class: Target class for the visualization
        use_smoothgrad: Whether to use SmoothGrad for better visualization
        
    Returns:
        Dictionary with visualization results
    """
    # Load model via shared cache
    model, resize_dim = load_model(model_type)
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Load and preprocess the image
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    
    # Preserve original image for visualization
    original_img = img.copy()
    
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]  # Take first channel instead of averaging
    if len(img.shape) < 2:
        raise ValueError("Input image must have at least 2 dimensions")
    
    # Add channel dimension
    img = img[None, :, :]
    
    # Set up transformation pipeline
    if model_type == 'densenet':
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
        resize_dim = 224
    else:  # resnet
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayResizer(512),
            xrv.datasets.XRayCenterCrop()
        ])
        resize_dim = 512
    
    # Apply transforms
    img = transform(img)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img)
    
    # Add batch dimension
    if len(img_tensor.shape) < 3:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Get model predictions to determine target class if not provided
    wrapped_model.eval()
    with torch.no_grad():
        try:
            preds = wrapped_model(img_tensor)
        except RuntimeError as e:
            if "could not create a primitive" in str(e):
                logger.warning("PyTorch primitive error; using CPU fallback.")
                wrapped_model = wrapped_model.cpu()
                img_tensor = img_tensor.cpu()
                preds = wrapped_model(img_tensor)
            else:
                raise e
    
    # If target_class is not provided, use the class with the highest probability
    if target_class is None:
        pred_idx = int(torch.argmax(preds).item())
        # For ResNet, we need to get the pathology name from default_pathologies
        if model_type == 'resnet':
            target_class = xrv.datasets.default_pathologies[pred_idx]
        else:
            target_class = wrapped_model.pathologies[pred_idx]
    elif isinstance(target_class, str):
        # If target_class is a pathology name, validate it exists
        if model_type == 'resnet':
            if target_class not in xrv.datasets.default_pathologies:
                logger.warning("Pathology %s not found; using highest probability class.", target_class)
                pred_idx = int(torch.argmax(preds).item())
                target_class = xrv.datasets.default_pathologies[pred_idx]
        else:
            try:
                target_class_idx = wrapped_model.pathologies.index(target_class)
            except ValueError:
                logger.warning("Pathology %s not found in model; using highest probability class.", target_class)
                pred_idx = int(torch.argmax(preds).item())
                target_class = wrapped_model.pathologies[pred_idx]
    
    # Initialize PixelLevelInterpretability
    try:
        pli = PixelLevelInterpretability(wrapped_model)
        
        # Generate saliency map
        if use_smoothgrad:
            saliency_map, _ = pli.apply_smoothgrad(img_tensor, target_class)
        else:
            saliency_map, _ = pli.generate_saliency(img_tensor, target_class)
    except Exception as e:
        logger.exception("Error generating pixel interpretability")
        # Return empty saliency map in case of error
        saliency_map = np.zeros((img.shape[1], img.shape[2]))
    
    # Convert original image to a suitable format for visualization
    original_vis = original_img
    if len(original_vis.shape) > 2:
        original_vis = original_vis[:, :, 0]  # Take first channel for visualization
    
    # Scale to [0, 1] for visualization
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min() + 1e-8)
    
    # Resize saliency map to match original image size
    saliency_map_resized = cv2.resize(saliency_map, (original_vis.shape[1], original_vis.shape[0]))  # type: ignore
    
    # Create colored representation of saliency map (using JET colormap for better contrast)
    saliency_colored = cv2.applyColorMap(np.uint8(255 * saliency_map_resized), cv2.COLORMAP_JET)  # type: ignore[arg-type]
    
    # Create basic overlay on the original image
    alpha = 0.6  # Reduce opacity
    overlay = np.uint8(255 * original_vis)
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)  # type: ignore
    
    saliency_overlay = cv2.addWeighted(overlay, 1 - alpha, cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB), alpha, 0)  # type: ignore[arg-type]
    
    # Return results
    return {
        'original': original_vis,
        'saliency_map': saliency_map_resized,
        'saliency_colored': cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB),
        'overlay': saliency_overlay,
        'target_class': target_class,
        'method': 'pli'
    }


def apply_combined_pixel_interpretability(
    image_path: str,
    model_type: str = 'densenet',
    probability_threshold: float = 0.5,
    use_smoothgrad: bool = True,
) -> dict[str, Any]:
    """
    Apply combined Pixel-Level Interpretability to an X-ray image for all pathologies above threshold
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        probability_threshold: Minimum probability threshold (default: 0.5)
        use_smoothgrad: Whether to use SmoothGrad technique (default: True)
        
    Returns:
        Dictionary with combined visualization results
    """
    # Load model via shared cache
    model, _ = load_model(model_type)
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Load and preprocess the image
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)
    
    # Preserve original image for visualization
    original_img = img.copy()
    
    # Check that images are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]  # Take first channel instead of averaging
    if len(img.shape) < 2:
        raise ValueError("Input image must have at least 2 dimensions")
    
    # Add channel dimension
    img = img[None, :, :]
    
    # Set up transformation pipeline
    if model_type == 'densenet':
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224)
        ])
    else:  # resnet
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayResizer(512),
            xrv.datasets.XRayCenterCrop()
        ])
    
    # Apply transforms
    img = transform(img)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img)
    
    # Add batch dimension
    if len(img_tensor.shape) < 3:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Initialize Pixel-Level Interpretability
    pli = PixelLevelInterpretability(wrapped_model)
    
    # Get combined saliency map for pathologies above threshold
    combined_saliency, selected_pathologies, predictions = pli.get_combined_saliency(
        img_tensor, probability_threshold=probability_threshold, use_smoothgrad=use_smoothgrad
    )
    
    # Convert original image to a suitable format for visualization
    original_vis = original_img
    if len(original_vis.shape) > 2:
        original_vis = original_vis[:, :, 0]  # Take first channel for visualization
    
    # Scale to [0, 1] for visualization
    original_vis = (original_vis - original_vis.min()) / (original_vis.max() - original_vis.min() + 1e-8)
    
    # Create colored saliency map
    saliency_colored = cv2.applyColorMap(np.uint8(255 * combined_saliency), cv2.COLORMAP_JET)  # type: ignore[arg-type]
    saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)  # type: ignore
    
    # Create overlay visualization (saliency over original image)
    # Resize saliency to match original image dimensions
    original_shape = original_vis.shape
    saliency_resized = cv2.resize(combined_saliency, (original_shape[1], original_shape[0]))  # type: ignore
    
    # Convert original to RGB for overlay
    original_rgb = cv2.cvtColor(np.uint8(original_vis * 255), cv2.COLOR_GRAY2RGB)  # type: ignore
    
    # Apply colormap to saliency
    saliency_colored_resized = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)  # type: ignore[arg-type]
    saliency_colored_resized = cv2.cvtColor(saliency_colored_resized, cv2.COLOR_BGR2RGB)  # type: ignore
    
    # Create overlay with transparency
    alpha = 0.4  # Transparency factor
    overlaid_img = cv2.addWeighted(original_rgb, 1 - alpha, saliency_colored_resized, alpha, 0)  # type: ignore[arg-type]
    
    # Create pathology summary string
    pathology_summary = ", ".join([f"{name} ({prob:.3f})" for name, prob in selected_pathologies])
    
    # Return results
    return {
        'original': original_vis,
        'saliency_map': combined_saliency,
        'saliency_colored': saliency_colored,
        'overlay': overlaid_img,
        'selected_pathologies': selected_pathologies,
        'pathology_summary': pathology_summary,
        'method': 'combined_pli',
        'threshold': probability_threshold
    } 