import numpy as np
import torch
import torchvision
import skimage
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter threading issues
from typing import Optional
import torchxrayvision as xrv


def disable_inplace_relu(model):
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
        
        # For DenseNet-121, the default target layer is the last convolutional layer
        if target_layer is None:
            if hasattr(model, 'features') and hasattr(model.features, 'denseblock4'):
                # For torchxrayvision DenseNet models
                self.target_layer = model.features.denseblock4.denselayer16.norm2
            elif hasattr(model, 'model') and hasattr(model.model, 'layer4'):
                # For torchxrayvision ResNet models (which have a nested model structure)
                self.target_layer = model.model.layer4[-1]
            elif hasattr(model, 'layer4'):
                # For standard ResNet models
                self.target_layer = model.layer4[-1]
            else:
                # Print model structure to help with debugging
                print("Model structure:")
                for name, module in model.named_modules():
                    print(f"Module: {name}")
                raise ValueError("Could not determine the target layer. Please specify explicitly.")
        else:
            self.target_layer = target_layer
        
        # Hooks populate these; guard before use so type checker narrows from Optional
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks to capture activations and gradients
        self.hook_forward = self.target_layer.register_forward_hook(forward_hook)
        self.hook_backward = self.target_layer.register_full_backward_hook(backward_hook)
    
    def _release_hooks(self):
        self.hook_forward.remove()
        self.hook_backward.remove()
    
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
        output = self.model(input_tensor)
        
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
        
        # Backward pass (calculate gradients)
        one_hot = torch.zeros_like(output)
        target_idx: int = int(target_class)
        one_hot[0, target_idx] = 1
        output.backward(gradient=one_hot)
        
        # Calculate weights based on global average pooling of gradients
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients/activations not captured. Ensure backward hooks ran.")
        pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
        
        # Create a copy of activations to avoid in-place modifications
        # of tensors that are part of the computation graph
        activations = self.activations.clone()
        
        # Weight activation maps with gradients (without in-place operations)
        weighted_activations = torch.zeros_like(activations)
        for i in range(pooled_gradients.shape[0]):
            weighted_activations[:, i, :, :] = activations[:, i, :, :] * pooled_gradients[i]
        
        # Global average pooling of weighted activation maps
        heatmap = torch.mean(weighted_activations, dim=1).squeeze().cpu().detach().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap, output
    
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
        # Release hooks when object is deleted
        if hasattr(self, 'hook_forward') and hasattr(self, 'hook_backward'):
            self._release_hooks()
    
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
        # Forward pass to get predictions
        self.model.zero_grad()
        output = self.model(input_tensor)
        
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
            print(f"No pathologies found above threshold {probability_threshold}")
            print("Top 3 pathologies:")
            top_indices = np.argsort(probabilities)[-3:][::-1]
            for idx in top_indices:
                print(f"  {pathology_names[idx]}: {probabilities[idx]:.3f}")
            # Use the top pathology if none above threshold
            selected_indices = [top_indices[0]]
            selected_pathologies = [(pathology_names[top_indices[0]], probabilities[top_indices[0]])]
        
        print(f"Selected pathologies above {probability_threshold} threshold:")
        for pathology, prob in selected_pathologies:
            print(f"  {pathology}: {prob:.3f}")
        
        # Generate combined heatmap
        combined_heatmap = None
        
        for i, target_class in enumerate(selected_indices):
            # Reset gradients for each pathology
            self.model.zero_grad()
            
            # Forward pass (reuse the same output for efficiency)
            if i == 0:
                current_output = self.model(input_tensor)
            else:
                current_output = self.model(input_tensor)
            
            # Backward pass for this specific pathology
            one_hot = torch.zeros_like(current_output)
            one_hot[0, int(target_class)] = 1
            current_output.backward(gradient=one_hot, retain_graph=(i < len(selected_indices) - 1))
            
            # Calculate weights based on global average pooling of gradients
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Gradients/activations not captured. Ensure backward hooks ran.")
            pooled_gradients = torch.mean(self.gradients, dim=(0, 2, 3))
            
            # Create a copy of activations to avoid in-place modifications
            activations = self.activations.clone()
            
            # Weight activation maps with gradients
            weighted_activations = torch.zeros_like(activations)
            for j in range(pooled_gradients.shape[0]):
                weighted_activations[:, j, :, :] = activations[:, j, :, :] * pooled_gradients[j]
            
            # Global average pooling of weighted activation maps
            heatmap = torch.mean(weighted_activations, dim=1).squeeze().cpu().detach().numpy()
            
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
        
        return combined_heatmap, selected_pathologies, output


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
            print(f"No pathologies found above threshold {probability_threshold}")
            print("Top 3 pathologies:")
            top_indices = np.argsort(probabilities)[-3:][::-1]
            for idx in top_indices:
                print(f"  {pathology_names[idx]}: {probabilities[idx]:.3f}")
            # Use the top pathology if none above threshold
            selected_indices = [top_indices[0]]
            selected_pathologies = [(pathology_names[top_indices[0]], probabilities[top_indices[0]])]
        
        print(f"Selected pathologies above {probability_threshold} threshold:")
        for pathology, prob in selected_pathologies:
            print(f"  {pathology}: {prob:.3f}")
        
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


def apply_gradcam(image_path, model_type='densenet', target_class=None):
    """
    Apply Grad-CAM to an X-ray image
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        target_class: Target class for Grad-CAM visualization
        
    Returns:
        Dictionary with visualization results
    """
    # Load model
    if model_type == 'resnet':
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        resize_dim = 512
    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        resize_dim = 224
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Let GradCAM auto-detect the appropriate target layer
    
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
        preds = wrapped_model(img_tensor)
    
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
                print(f"Pathology {target_class} not found. Using highest probability class.")
                pred_idx = int(torch.argmax(preds).item())
                target_class = xrv.datasets.default_pathologies[pred_idx]
        else:
            try:
                target_class_idx = wrapped_model.pathologies.index(target_class)
            except ValueError:
                print(f"Pathology {target_class} not found in model. Using highest probability class.")
                pred_idx = int(torch.argmax(preds).item())
                target_class = wrapped_model.pathologies[pred_idx]
    
    # Initialize Grad-CAM (auto-detect target layer)
    gradcam = GradCAM(wrapped_model)
    
    # Get heatmap
    heatmap, _ = gradcam.get_heatmap(img_tensor, target_class)
    
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
        'target_class': target_class
    }


def apply_combined_gradcam(image_path, model_type='densenet', probability_threshold=0.5):
    """
    Apply combined interpretability to an X-ray image for all pathologies above threshold
    
    Args:
        image_path: Path to the image
        model_type: 'densenet' or 'resnet'
        probability_threshold: Minimum probability threshold (default: 0.5)
        
    Returns:
        Dictionary with combined visualization results
    """
    # Load model
    if model_type == 'resnet':
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        resize_dim = 512
    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        resize_dim = 224
    
    # Wrap model to prevent in-place operations
    wrapped_model = NoInplaceReLU(model)
    
    # Let GradCAM auto-detect the appropriate target layer
    
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
    
    # Initialize Grad-CAM (auto-detect target layer)
    gradcam = GradCAM(wrapped_model)
    
    # Get combined heatmap for pathologies above threshold
    combined_heatmap, selected_pathologies, predictions = gradcam.get_combined_heatmap(
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


def apply_pixel_interpretability(image_path, model_type='densenet', target_class=None, use_smoothgrad=True):
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
    # Load model
    if model_type == 'resnet':
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        resize_dim = 512
    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        resize_dim = 224
    
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
        preds = wrapped_model(img_tensor)
    
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
                print(f"Pathology {target_class} not found. Using highest probability class.")
                pred_idx = int(torch.argmax(preds).item())
                target_class = xrv.datasets.default_pathologies[pred_idx]
        else:
            try:
                target_class_idx = wrapped_model.pathologies.index(target_class)
            except ValueError:
                print(f"Pathology {target_class} not found in model. Using highest probability class.")
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
        print(f"Error generating pixel interpretability: {str(e)}")
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


def apply_combined_pixel_interpretability(image_path, model_type='densenet', probability_threshold=0.5, use_smoothgrad=True):
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
    # Load model
    if model_type == 'resnet':
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        resize_dim = 512
    else:
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        resize_dim = 224
    
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
    assert combined_saliency is not None
    saliency_colored = cv2.applyColorMap(np.uint8(255 * combined_saliency), cv2.COLORMAP_JET)  # type: ignore
    saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)  # type: ignore
    
    # Create overlay visualization (saliency over original image)
    # Resize saliency to match original image dimensions
    original_shape = original_vis.shape
    saliency_resized = cv2.resize(combined_saliency, (original_shape[1], original_shape[0]))  # type: ignore
    
    # Convert original to RGB for overlay
    original_rgb = cv2.cvtColor(np.uint8(original_vis * 255), cv2.COLOR_GRAY2RGB)  # type: ignore
    
    # Apply colormap to saliency
    saliency_colored_resized = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)  # type: ignore
    saliency_colored_resized = cv2.cvtColor(saliency_colored_resized, cv2.COLOR_BGR2RGB)  # type: ignore
    
    # Create overlay with transparency
    alpha = 0.4  # Transparency factor
    overlaid_img = cv2.addWeighted(original_rgb, 1 - alpha, saliency_colored_resized, alpha, 0)  # type: ignore
    
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