import cv2 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def setup_paths():
    """Automatically detect all paths based on script location"""
    # Get current script directory (myproject/Python/)
    script_dir = Path(__file__).resolve().parent
    # Get project root (myproject/)
    project_root = script_dir.parent

    images_dir = project_root / 'Images'

    return {
       'project_root': project_root,
       'images_dir': images_dir
    }


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
        #ensure dimensions are multiples of stride
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


#Image preprocessing pipeline

paths = setup_paths()
image_path=Path(paths["images_dir"]) / "HEstain.png"

if not Path(image_path).exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

# Load image as grayscale
print(f" Loading image: {image_path}")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Could not load image from {image_path}")

print(f"   Original image shape: {img.shape}")

img_resized = resize_for_superpoint(img, max_pixels=8000000)

# Normalize to [0, 1] for model input
#img_normalized = img_resized.astype(np.float32) / 255.0

cv2.imwrite(paths['images_dir'] / "Resized_Image.png", img_resized)
