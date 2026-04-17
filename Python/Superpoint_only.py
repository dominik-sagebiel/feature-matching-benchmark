"""
detect keypoints with rpautrat superpoint
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time 
from datetime import datetime

# ============================================================================
# AUTO PATH CONFIGURATION 
# ============================================================================
def setup_paths():
    """Automatically detect all paths based on script location"""
    # Get current script directory (myproject/Python/)
    script_dir = Path(__file__).resolve().parent
    # Get project root (myproject/)
    project_root = script_dir.parent
    
    # Define all paths relative to project root
    images_dir = project_root / 'Images'
    repos_dir = project_root / 'Repos'
    results_dir = project_root / 'Results' / 'Python' / 'rpautratSuperpoint'
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find SuperPoint in Repos directory
    superpoint_path = None
    if repos_dir.exists():
        # Look for any directory containing 'SuperPoint' (case insensitive)
        sp_candidates = [d for d in repos_dir.iterdir() 
                        if d.is_dir() and 'superpoint' in d.name.lower()]
        if sp_candidates:
            superpoint_path = sp_candidates[0]
            print(f"✓ Found SuperPoint at: {superpoint_path}")
    
    if superpoint_path is None:
        raise FileNotFoundError(
            f"\n❌ SuperPoint not found!\n"
            f"Expected location: {repos_dir}/SuperPoint or similar\n"
            f"Please clone it:\n"
            f"  cd {repos_dir}\n"
            f"  git clone https://github.com/rpautrat/SuperPoint.git\n"
        )
    
    # Add SuperPoint to Python path
    sys.path.insert(0, str(superpoint_path))
    
    # Find the weights file (try common locations)
    weights_path = None
    weight_candidates = [
        superpoint_path / 'weights' / 'superpoint_v6_from_tf.pth',
        superpoint_path / 'weights' / 'superpoint_v1.pth',
        superpoint_path / 'superpoint_v6_from_tf.pth',
        superpoint_path / 'superpoint_v1.pth',
        project_root / 'Models' / 'superpoint_v6_from_tf.pth',  # Alternative location
    ]
    
    for candidate in weight_candidates:
        if candidate.exists():
            weights_path = candidate
            print(f"✓ Found weights at: {weights_path}")
            break
    
    if weights_path is None:
        raise FileNotFoundError(
            f"\n❌ SuperPoint weights not found!\n"
            f"Searched in:\n" + 
            "\n".join(f"  - {p}" for p in weight_candidates) +
            f"\n\nPlease ensure weights file is in the SuperPoint/weights/ directory"
        )
    
    return {
        'project_root': project_root,
        'images_dir': images_dir,
        'results_dir': results_dir,
        'superpoint_path': superpoint_path,
        'weights_path': weights_path
    }

# ============================================================================
# Run path setup (make paths available globally)
# ============================================================================
paths = setup_paths()

# Import SuperPoint after path is set
from superpoint_pytorch import SuperPoint

# ============================================================================
# IMAGE RESIZING FOR SUPEROINT (Memory Safe)
# ============================================================================
def resize_for_superpoint(image, max_pixels=4000000, stride=8):
    """
    Resize image to target megapixels while preserving aspect ratio.
    Ensures dimensions are multiples of stride (8).
    
    Args:
        image: Input image (grayscale or color)
        max_pixels: Maximum number of pixels (e.g., 4_000_000 for 4MP)
        stride: Required stride for SuperPoint (default 8)
    
    Returns:
        Resized image with dimensions multiple of stride
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    
    if total_pixels <= max_pixels:
        # Still need to ensure dimensions are multiples of stride
        new_h = (h // stride) * stride
        new_w = (w // stride) * stride
        
        if new_h != h or new_w != w:
            # Crop from center to make dimensions divisible by stride
            offset_y = (h - new_h) // 2
            offset_x = (w - new_w) // 2
            image = image[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
            print(f"   Cropped from {w}x{h} to {new_w}x{new_h} (center crop for stride {stride})")
        
        return image
    
    # Calculate scale factor to reach target pixels
    scale = np.sqrt(max_pixels / total_pixels)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Round to nearest multiple of stride
    new_h = (new_h // stride) * stride
    new_w = (new_w // stride) * stride
    
    print(f"   Original: {w}x{h} ({total_pixels/1e6:.1f}MP)")
    print(f"   Resizing to: {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")
    
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image

#=========================================
#superpoint pipeline
#=========================================
# Initialize model
model = SuperPoint(
    nms_radius=4,
    max_num_keypoints=20000, 
    detection_threshold=0.005,
    remove_borders=4,
    descriptor_dim=256,
    channels=[64, 64, 128, 128, 256]
)

# Load weights
weights_path = r'c:\Users\domin\Nextcloud\Uni\FU Ba. BioInf\10. Sem\Praktikum MDC\git\SuperPoint\weights\superpoint_v6_from_tf.pth'
state_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

print(" SuperPoint model loaded successfully!")

start_time = time.time()
# Load and prepare the image

image_path=Path(paths["images_dir"]) / "HEstain.png"

if not Path(image_path).exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image as grayscale
print(f" Loading image: {image_path}")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Could not load image from {image_path}")

print(f"   Original image shape: {img.shape}")

img_resized = resize_for_superpoint(img, max_pixels=4000000)

# Store resized image for visualization
img_display = img_resized

# Normalize to [0, 1] for model input
img_normalized = img_resized.astype(np.float32) / 255.0

preprocessing_time = time.time() - start_time 

start_time = time.time()
# Convert to tensor
input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)
img_to_tensor_time = time.time() - start_time
print(f"   Input tensor shape: {input_tensor.shape}")

# Run inference
print("\n Running SuperPoint inference...")
start_time = time.time()
with torch.no_grad():
    output = model({"image": input_tensor})
inference_time = time.time() - start_time

# Extract results
keypoints = output['keypoints'][0].cpu().numpy()
descriptors = output['descriptors'][0].cpu().numpy()
scores = output['keypoint_scores'][0].cpu().numpy()

# Create visualization
plt.figure(figsize=(12, 8))
plt.imshow(img_display, cmap='gray')

if len(keypoints) > 0:
    plt.scatter(keypoints[:, 0], keypoints[:, 1], 
                c='lime',           # Bright green
                s=15,                # Small, consistent size
                alpha=0.8,            # Slightly transparent
                marker='o',           # Circle marker
                edgecolors=None)      # No outline

plt.title(f'SuperPoint Keypoints: {len(keypoints)}', fontsize=14)
plt.axis('off')
plt.tight_layout()

# Get current timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Save visualization FIRST (before showing)
viz_filename = f"superpoint_keypoints_{timestamp}.png"
output_path = Path(paths["results_dir"]) / viz_filename
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Visualization saved to: {output_path}")

# THEN show the plot
plt.show()
plt.close()

from datetime import datetime

# Create filename with timestamp
filename = f"superpoint_timings_{timestamp}.txt"

txt_path = Path(paths["results_dir"]) / filename

# Save results
with open(txt_path, 'w') as f:
    f.write("SuperPoint Performance Timings (Python)\n")
    f.write("========================================\n\n")
    f.write(f"Image: {image_path}\n")
    f.write(f"Image original shape: {img.shape}\n\n")
    f.write(f"Image shape (resized): {img_resized.shape}\n\n")
    f.write(f"Keypoints detected: {len(keypoints)}\n\n")
    f.write("Timings:\n")
    f.write(f"  Preprocessing time: {preprocessing_time:.4f} seconds\n")
    f.write(f"  Image to tensor time: {img_to_tensor_time:.4f} seconds\n")
    f.write(f"  Inference time: {inference_time:.4f} seconds\n")
    f.write(f"  Total time: {preprocessing_time + img_to_tensor_time + inference_time:.4f} seconds\n")

print(f" Results saved to: {output_path}")
