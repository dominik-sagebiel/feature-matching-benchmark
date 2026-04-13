"""
Compare SIFT vs SuperPoint registration using FLANN + RANSAC
Evaluate both on a 20-point grid
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
    results_dir = project_root / 'Results' / 'Python' / 'twoway_comparison'
    
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

# ============================================================================
# Point Selection
# ============================================================================
def select_grid_points(image, grid_rows=4, grid_cols=5, margin=50):
    """
    Select evenly spaced grid points (for evaluation only)
    """
    h, w = image.shape
    step_x = (w - 2 * margin) / (grid_cols - 1)
    step_y = (h - 2 * margin) / (grid_rows - 1)
    
    points = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = margin + col * step_x
            y = margin + row * step_y
            points.append([x, y])
    
    points = np.array(points)
    print(f"Generated {len(points)} grid points ({grid_rows}x{grid_cols})")
    return points

# ============================================================================
# Image Rotation (Ground Truth)
# ============================================================================
def rotate_image_and_points(image, points, angle=45):
    """
    Rotate image and transform points (ground truth)
    """
    h, w = image.shape
    center = (w/2, h/2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
    
    # Transform points
    transformed_points = []
    for pt in points:
        x, y = pt
        x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        transformed_points.append([x_new, y_new])
    
    transformed_points = np.array(transformed_points)
    
    print(f"   Original size: {w}x{h}")
    print(f"   Rotated size: {new_w}x{new_h}")
    
    return rotated_image, transformed_points, M

# ============================================================================
# SIFT Feature Extraction
# ============================================================================
def extract_sift_features(image, max_keypoints=5000):
    """
    Extract SIFT keypoints and descriptors
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    if keypoints is None:
        return None
    
    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    scores = np.array([kp.response for kp in keypoints])
    
    print(f"   Found {len(kp_array)} keypoints")
    
    return {
        'keypoints': kp_array,
        'descriptors': descriptors,
        'scores': scores,
        'num_keypoints': len(kp_array)
    }

# ============================================================================
# SuperPoint Setup
# ============================================================================
def load_superpoint_model():
    """Load SuperPoint model using auto-detected paths"""
    model = SuperPoint(
        nms_radius=2,
        max_num_keypoints=5000,
        detection_threshold=0.001,
        remove_borders=4,
        descriptor_dim=256,
        channels=[64, 64, 128, 128, 256]
    )
    
    # Use the auto-detected weights path
    state_dict = torch.load(paths['weights_path'], map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model

def extract_superpoint_features(model, image, max_pixels=4000000):
    """
    Extract SuperPoint keypoints and descriptors
    Includes automatic downsampling for large images
    """
    # Apply memory-safe resizing (4MP default)
    original_h, original_w = image.shape
    image_resized = resize_for_superpoint(image, max_pixels=max_pixels, stride=8)
    h, w = image_resized.shape
    
    # Normalize and convert to tensor
    img_normalized = image_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        output = model({"image": input_tensor})
    inference_time = time.time() - start_time
    
    # Extract results
    keypoints = output['keypoints'][0].cpu().numpy()
    descriptors = output['descriptors'][0].cpu().numpy()
    scores = output['keypoint_scores'][0].cpu().numpy()
    
    # Scale keypoints back to original image coordinates if resized
    if h != original_h or w != original_w:
        scale_x = original_w / w
        scale_y = original_h / h
        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y
        print(f"   Scaled keypoints back to original resolution: {original_w}x{original_h}")
    
    print(f"   SuperPoint inference: {inference_time*1000:.1f} ms")
    print(f"   Found {len(keypoints)} keypoints")
    
    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'scores': scores,
        'num_keypoints': len(keypoints),
        'time': inference_time
    }

# ============================================================================
# FLANN Matcher
# ============================================================================
def get_flann_matcher():
    """Get FLANN matcher"""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def match_features_flann(desc1, desc2, flann_matcher, lowes_ratio=0.8):
    """
    Match features using FLANN with Lowe's ratio test
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    
    start_time = time.time()
    matches = flann_matcher.knnMatch(desc1, desc2, k=2)
    matching_time = time.time() - start_time
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowes_ratio * n.distance:
                good_matches.append(m)
    
    print(f"   FLANN matching: {matching_time*1000:.1f} ms")
    print(f"   Found {len(good_matches)} good matches")
    
    return good_matches

# ============================================================================
# RANSAC Filtering
# ============================================================================
def filter_with_ransac(matches, kp1, kp2, ransac_thresh=5.0):
    """
    Filter matches using RANSAC to find geometrically consistent ones
    """
    if len(matches) < 4:
        print(f"   RANSAC: Need at least 4 matches, have {len(matches)}")
        return [], None, None
    
    src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    
    if H is not None:
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        print(f"   RANSAC: {len(inliers)}/{len(matches)} inliers ({len(inliers)/len(matches)*100:.1f}%)")
        return inliers, H, mask
    else:
        print(f"   RANSAC: Failed to find homography")
        return [], None, None

# ============================================================================
# Evaluation
# ============================================================================
def evaluate_grid_points(grid_points, ground_truth_points, H_auto):
    """
    Evaluate how well automatic transformation works on grid points
    """
    if H_auto is None:
        return np.array([np.inf] * len(grid_points)), None
    
    # Ensure points are float32
    pts = grid_points.astype(np.float32).reshape(-1, 1, 2)
    
    # Apply automatic transformation
    pts_auto = cv2.perspectiveTransform(pts, H_auto).reshape(-1, 2)
    
    # Calculate errors against ground truth
    errors = np.linalg.norm(pts_auto - ground_truth_points, axis=1)
    
    return errors, pts_auto

# ============================================================================
# Visualization
# ============================================================================
def create_output_dir():
    """Create timestamped output directory inside Results/Python/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths['results_dir'] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    return output_dir

def draw_matches(img1, img2, kp1, kp2, matches, inliers, title, save_path):
    """
    Draw matches with inliers in green, outliers in red
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    match_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    match_img[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Create set of inlier query indices
    inlier_set = set()
    for m in inliers:
        inlier_set.add(m.queryIdx)
    
    for m in matches:
        pt1 = (int(kp1[m.queryIdx][0]), int(kp1[m.queryIdx][1]))
        pt2 = (int(kp2[m.trainIdx][0] + w1), int(kp2[m.trainIdx][1]))
        
        if m.queryIdx in inlier_set:
            color = (0, 255, 0)  # Green
            thickness = 2
        else:
            color = (0, 0, 255)  # Red
            thickness = 1
        
        cv2.line(match_img, pt1, pt2, color, thickness)
        cv2.circle(match_img, pt1, 3, color, -1)
        cv2.circle(match_img, pt2, 3, color, -1)
    
    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} - {len(inliers)} inliers / {len(matches)} total')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to avoid memory issues

def draw_grid_evaluation(img1, img2, grid_points, ground_truth, transformed_points, errors, title, save_path):
    """
    Draw grid point evaluation
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2
    
    plt.figure(figsize=(16, 9))
    plt.imshow(combined, cmap='gray')
    
    # Draw grid points and their transformations
    for i, (orig, auto, truth) in enumerate(zip(grid_points, transformed_points, ground_truth)):
        # Original point (red circle)
        plt.plot(orig[0], orig[1], 'ro', markersize=8, markeredgecolor='white')
        # Ground truth (blue cross)
        plt.plot(truth[0] + w1, truth[1], 'b+', markersize=12, linewidth=2)
        # Auto transformation (green circle)
        plt.plot(auto[0] + w1, auto[1], 'go', markersize=8, markeredgecolor='white')
        # Connect auto to ground truth
        plt.plot([auto[0] + w1, truth[0] + w1], [auto[1], truth[1]], 'y-', linewidth=1, alpha=0.7)
    
    plt.title(f'{title}\nMean error: {errors.mean():.2f}px, Points <10px: {sum(e < 10 for e in errors)}/{len(grid_points)}')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to avoid memory issues

# ============================================================================
# Main Function
# ============================================================================
def main():
    # Find first image in Images directory (no hardcoded path!)
    image_files = list(paths['images_dir'].glob('*.png')) + \
                  list(paths['images_dir'].glob('*.jpg')) + \
                  list(paths['images_dir'].glob('*.tif'))
    
    if not image_files:
        raise FileNotFoundError(
            f"\n❌ No images found in {paths['images_dir']}\n"
            f"Please add .png, .jpg, or .tif files to the Images folder"
        )
    
    # Use the first image found (or you can modify to use a specific one)
    image_path = image_files[0]
    print(f"\n📷 Using image: {image_path.name}")
    
    rotation_angle = 45
    
    print("=" * 70)
    print("SIFT vs SuperPoint Registration Comparison")
    print("=" * 70)
    
    # Create output directory
    output_dir = create_output_dir()
    
    # Load original image
    img_original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img_original is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"\n[1] Loading original image: {img_original.shape}")
    
    # ========================================================================
    # Generate Grid Points (for evaluation only - not used in matching)
    # ========================================================================
    print("\n[2] Generating evaluation grid points...")
    grid_points = select_grid_points(img_original, grid_rows=4, grid_cols=5, margin=50)
    
    # ========================================================================
    # Rotate Image (ground truth)
    # ========================================================================
    print(f"\n[3] Rotating image by {rotation_angle}°...")
    img_rotated, ground_truth_points, rot_matrix = rotate_image_and_points(
        img_original, grid_points, angle=rotation_angle
    )
    
    # ========================================================================
    # SIFT Processing
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIFT REGISTRATION")
    print("=" * 70)
    
    print("\n[4a] Extracting SIFT features...")
    sift_orig = extract_sift_features(img_original, max_keypoints=5000)
    sift_rot = extract_sift_features(img_rotated, max_keypoints=5000)
    
    print("\n[4b] FLANN matching...")
    flann = get_flann_matcher()
    sift_matches = match_features_flann(sift_orig['descriptors'], sift_rot['descriptors'], flann)
    
    print("\n[4c] RANSAC filtering...")
    sift_inliers, sift_H, sift_mask = filter_with_ransac(
        sift_matches, sift_orig['keypoints'], sift_rot['keypoints'], ransac_thresh=5.0
    )
    
    print("\n[4d] Evaluating on grid points...")
    sift_errors, sift_transformed = evaluate_grid_points(grid_points, ground_truth_points, sift_H)
    
    # ========================================================================
    # SuperPoint Processing
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUPERPOINT REGISTRATION")
    print("=" * 70)
    
    print("\n[5a] Loading SuperPoint model...")
    sp_model = load_superpoint_model()
    
    print("\n[5b] Extracting SuperPoint features...")
    # Set max_pixels to 4_000_000 (4MP) or 8_000_000 (8MP)
    # You can change this to 8_000_000 for higher resolution
    sp_orig = extract_superpoint_features(sp_model, img_original, max_pixels=4000000)
    sp_rot = extract_superpoint_features(sp_model, img_rotated, max_pixels=4000000)
    
    print("\n[5c] FLANN matching...")
    sp_matches = match_features_flann(sp_orig['descriptors'], sp_rot['descriptors'], flann)
    
    print("\n[5d] RANSAC filtering...")
    sp_inliers, sp_H, sp_mask = filter_with_ransac(
        sp_matches, sp_orig['keypoints'], sp_rot['keypoints'], ransac_thresh=5.0
    )
    
    print("\n[5e] Evaluating on grid points...")
    sp_errors, sp_transformed = evaluate_grid_points(grid_points, ground_truth_points, sp_H)
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'SIFT':<20} {'SuperPoint':<20}")
    print("-" * 70)
    print(f"{'Keypoints (original)':<30} {sift_orig['num_keypoints']:<20} {sp_orig['num_keypoints']:<20}")
    print(f"{'Keypoints (rotated)':<30} {sift_rot['num_keypoints']:<20} {sp_rot['num_keypoints']:<20}")
    print(f"{'FLANN matches':<30} {len(sift_matches):<20} {len(sp_matches):<20}")
    print(f"{'RANSAC inliers':<30} {len(sift_inliers):<20} {len(sp_inliers):<20}")
    print(f"{'Inlier % (of matches)':<30} {len(sift_inliers)/max(len(sift_matches),1)*100:<20.1f} {len(sp_inliers)/max(len(sp_matches),1)*100:<20.1f}")
    print(f"\n{'Grid Point Evaluation':<30} {'':<20} {'':<20}")
    print(f"{'Mean error (px)':<30} {sift_errors.mean():<20.2f} {sp_errors.mean():<20.2f}")
    print(f"{'Std error (px)':<30} {sift_errors.std():<20.2f} {sp_errors.std():<20.2f}")
    print(f"{'Max error (px)':<30} {sift_errors.max():<20.2f} {sp_errors.max():<20.2f}")
    print(f"{'Points within 10px':<30} {sum(sift_errors < 10)}/{len(grid_points):<17} {sum(sp_errors < 10)}/{len(grid_points)}")
    
    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n[6] Generating visualizations...")
    
    # SIFT visualizations
    draw_matches(
        img_original, img_rotated,
        sift_orig['keypoints'], sift_rot['keypoints'],
        sift_matches, sift_inliers,
        'SIFT',
        output_dir / "sift_matches.png"
    )
    
    draw_grid_evaluation(
        img_original, img_rotated,
        grid_points, ground_truth_points, sift_transformed, sift_errors,
        'SIFT',
        output_dir / "sift_grid_evaluation.png"
    )
    
    # SuperPoint visualizations
    draw_matches(
        img_original, img_rotated,
        sp_orig['keypoints'], sp_rot['keypoints'],
        sp_matches, sp_inliers,
        'SuperPoint',
        output_dir / "superpoint_matches.png"
    )
    
    draw_grid_evaluation(
        img_original, img_rotated,
        grid_points, ground_truth_points, sp_transformed, sp_errors,
        'SuperPoint',
        output_dir / "superpoint_grid_evaluation.png"
    )
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n[7] Saving results...")
    
    np.savez(
        output_dir / "results.npz",
        grid_points=grid_points,
        ground_truth_points=ground_truth_points,
        sift_H=sift_H,
        sift_errors=sift_errors,
        sift_transformed=sift_transformed,
        sp_H=sp_H,
        sp_errors=sp_errors,
        sp_transformed=sp_transformed
    )
    
    # Save summary
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SIFT vs SuperPoint Registration Comparison\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Input image: {image_path.name}\n")
        f.write(f"Rotation angle: {rotation_angle}°\n")
        f.write(f"Grid points: {len(grid_points)} (4x5)\n\n")
        
        f.write("SIFT RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Keypoints (original): {sift_orig['num_keypoints']}\n")
        f.write(f"Keypoints (rotated): {sift_rot['num_keypoints']}\n")
        f.write(f"FLANN matches: {len(sift_matches)}\n")
        f.write(f"RANSAC inliers: {len(sift_inliers)} ({len(sift_inliers)/max(len(sift_matches),1)*100:.1f}%)\n")
        f.write(f"Grid point mean error: {sift_errors.mean():.2f} px\n")
        f.write(f"Grid points within 10px: {sum(sift_errors < 10)}/{len(grid_points)}\n\n")
        
        f.write("SUPERPOINT RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Keypoints (original): {sp_orig['num_keypoints']}\n")
        f.write(f"Keypoints (rotated): {sp_rot['num_keypoints']}\n")
        f.write(f"FLANN matches: {len(sp_matches)}\n")
        f.write(f"RANSAC inliers: {len(sp_inliers)} ({len(sp_inliers)/max(len(sp_matches),1)*100:.1f}%)\n")
        f.write(f"Grid point mean error: {sp_errors.mean():.2f} px\n")
        f.write(f"Grid points within 10px: {sum(sp_errors < 10)}/{len(grid_points)}\n")
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Images: sift_matches.png, sift_grid_evaluation.png")
    print(f"   - Images: superpoint_matches.png, superpoint_grid_evaluation.png")
    print(f"   - Data: results.npz")
    print(f"   - Summary: comparison_summary.txt")
    print("\n✨ Done!")

if __name__ == "__main__":
    main()