"""
Five‑way Registration Comparison on a 20‑point Grid

"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from datetime import datetime

# -----------------------------------------------------------------------------
# Paths – adjust these to your environment
# -----------------------------------------------------------------------------
"""Automatically detect all paths based on script location"""
# Get current script directory (myproject/Python/)
script_dir = Path(__file__).resolve().parent
# Get project root (myproject/)
project_root = script_dir.parent

# Define import Paths
SUPERGLUE_PATH = project_root / 'Repos' / 'SuperGluePretrainedNetwork'
SUPERPOINT_PATH = project_root / 'Repos' / 'SuperPoint'

# Importing Superglue Superpoint
sys.path.append(str(SUPERGLUE_PATH))
sys.path.append(str(Path(SUPERGLUE_PATH) / 'models'))
sys.path.append(str(SUPERPOINT_PATH))


from superpoint_pytorch import SuperPoint as RPAutratSuperPoint

# Official SuperGlue imports
from models.superpoint import SuperPoint as OfficialSuperPoint
from models.superglue import SuperGlue

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# -----------------------------------------------------------------------------
# Global parameters
# -----------------------------------------------------------------------------
ROTATION_ANGLE = 45
MAX_DIM = 1600                # resize largest side to this for SuperPoint methods
MAX_KP_SIFT = 5000
MAX_KP_SP_FLANN = 2000        # keypoints for SuperPoint + FLANN (after downscaling)
MAX_KP_SP_SG = 2000           # keypoints for SuperGlue (official)
RANSAC_THRESH = 5.0
LOWES_RATIO = 0.8

# ============================================================================
# Helper functions (grid, rotation, evaluation, visualisation)
# ============================================================================
def select_grid_points(image, grid_rows=4, grid_cols=5, margin=50):
    """Evenly spaced grid points (for evaluation only)."""
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

def rotate_image_and_points(image, points, angle=ROTATION_ANGLE):
    """Rotate image and transform points (ground truth)."""
    h, w = image.shape
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    transformed = []
    for pt in points:
        x, y = pt
        x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        transformed.append([x_new, y_new])
    transformed = np.array(transformed)
    print(f"   Original size: {w}x{h} -> rotated size: {new_w}x{new_h}")
    return rotated, transformed, M

def evaluate_grid_points(grid_points, ground_truth, H):
    """Apply homography H to grid points and compute errors against ground truth."""
    if H is None:
        return np.full(len(grid_points), np.inf), None
    pts = grid_points.astype(np.float32).reshape(-1, 1, 2)
    pts_transformed = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    errors = np.linalg.norm(pts_transformed - ground_truth, axis=1)
    return errors, pts_transformed

def draw_matches(img1, img2, kp1, kp2, matches, inliers, title, save_path):
    """Draw matches: inliers green, outliers red."""
    h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    inlier_set = {m.queryIdx for m in inliers}
    for m in matches:
        pt1 = (int(kp1[m.queryIdx][0]), int(kp1[m.queryIdx][1]))
        pt2 = (int(kp2[m.trainIdx][0] + w1), int(kp2[m.trainIdx][1]))
        color = (0, 255, 0) if m.queryIdx in inlier_set else (0, 0, 255)
        cv2.line(canvas, pt1, pt2, color, 2 if m.queryIdx in inlier_set else 1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)

    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} - {len(inliers)} inliers / {len(matches)} matches")
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def draw_grid_evaluation(img1, img2, grid, ground_truth, transformed, errors, title, save_path):
    """Visualise ground truth (blue crosses) vs automatic (green circles)."""
    if transformed is None:
        print("Überspringe Zeichnung: 'transformed' ist None.")
        return
    h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
    combined = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2

    plt.figure(figsize=(16, 9))
    plt.imshow(combined, cmap='gray')
    for i, (orig, auto, truth) in enumerate(zip(grid, transformed, ground_truth)):
        plt.plot(orig[0], orig[1], 'ro', markersize=8, markeredgecolor='white')
        plt.plot(truth[0] + w1, truth[1], 'b+', markersize=12, linewidth=2)
        plt.plot(auto[0] + w1, auto[1], 'go', markersize=8, markeredgecolor='white')
        plt.plot([auto[0] + w1, truth[0] + w1], [auto[1], truth[1]], 'y-', linewidth=1, alpha=0.7)
    plt.title(f"{title}\nMean error: {errors.mean():.2f}px, <10px: {np.sum(errors < 10)}/{len(grid)}")
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# FLANN + RANSAC utilities (shared)
# ============================================================================
def get_flann_matcher():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

def match_flann(desc1, desc2, matcher, lowes_ratio=LOWES_RATIO):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)
    t0 = time.time()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    t = time.time() - t0
    good = []
    for m, n in matches:
        if m.distance < lowes_ratio * n.distance:
            good.append(m)
    print(f"   FLANN matching: {t*1000:.1f} ms, {len(good)} matches")
    return good

def ransac_homography(matches, kp1, kp2, thresh=RANSAC_THRESH):
    if len(matches) < 4:
        return [], None
    src = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1,2)
    dst = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    if H is not None:
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        print(f"   RANSAC: {len(inliers)}/{len(matches)} inliers ({100*len(inliers)/len(matches):.1f}%)")
        return inliers, H
    else:
        print("   RANSAC failed")
        return [], None

# ============================================================================
# Pipeline 1: SIFT + FLANN + RANSAC
# ============================================================================
def extract_sift(image, max_kp=MAX_KP_SIFT):
    sift = cv2.SIFT_create(nfeatures=max_kp)
    kp, desc = sift.detectAndCompute(image, None)
    if kp is None:
        return None
    kp_arr = np.array([[p.pt[0], p.pt[1]] for p in kp])
    scores = np.array([p.response for p in kp])
    print(f"   SIFT: {len(kp_arr)} keypoints")
    return {'keypoints': kp_arr, 'descriptors': desc, 'scores': scores, 'num': len(kp_arr)}

def pipeline_sift(image_orig, image_rot):
    data_orig = extract_sift(image_orig)
    data_rot  = extract_sift(image_rot)
    if data_orig is None or data_rot is None:
        return None, None, None, None, None
    flann = get_flann_matcher()
    matches = match_flann(data_orig['descriptors'], data_rot['descriptors'], flann)
    inliers, H = ransac_homography(matches, data_orig['keypoints'], data_rot['keypoints'])
    return data_orig['keypoints'], data_rot['keypoints'], matches, inliers, H

# ============================================================================
# Pipeline 2: rpautrat SuperPoint + FLANN + RANSAC
# ============================================================================
def load_rpautrat_sp():
    model = RPAutratSuperPoint(
        nms_radius=2, max_num_keypoints=MAX_KP_SP_FLANN,
        detection_threshold=0.001, remove_borders=4,
        descriptor_dim=256, channels=[64, 64, 128, 128, 256]
    )
    w = str(Path(SUPERPOINT_PATH) / 'weights' / 'superpoint_v6_from_tf.pth')
    state = torch.load(w, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_rpautrat_sp(model, image, max_dim=MAX_DIM):
    orig_h, orig_w = image.shape
    # downscale
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = cv2.resize(image, (new_w, new_h))
    else:
        scale = 1.0
        new_h, new_w = orig_h, orig_w

    # divisible by 8
    h, w = image.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h))
        # scaling factor changes again
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
    else:
        scale_x = orig_w / w
        scale_y = orig_h / h

    tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model({'image': tensor})
    t = time.time() - t0
    kp = out['keypoints'][0].cpu().numpy()          # coordinates in processed image
    desc = out['descriptors'][0].cpu().numpy()
    scores = out['keypoint_scores'][0].cpu().numpy()
    # limit keypoints
    if len(kp) > MAX_KP_SP_FLANN:
        idx = np.argsort(scores)[-MAX_KP_SP_FLANN:]
        kp = kp[idx]
        desc = desc[idx]
        scores = scores[idx]
    # scale keypoints back to original image coordinates
    kp_orig = kp.copy()
    kp_orig[:, 0] *= scale_x
    kp_orig[:, 1] *= scale_y
    print(f"   rpautrat SP: {len(kp)} keypoints, {t*1000:.1f} ms")
    return {'keypoints': kp_orig, 'descriptors': desc, 'scores': scores, 'num': len(kp_orig)}

def pipeline_rpautrat_sp(model, image_orig, image_rot):
    data_orig = extract_rpautrat_sp(model, image_orig)
    data_rot  = extract_rpautrat_sp(model, image_rot)
    flann = get_flann_matcher()
    matches = match_flann(data_orig['descriptors'], data_rot['descriptors'], flann)
    inliers, H = ransac_homography(matches, data_orig['keypoints'], data_rot['keypoints'])
    return data_orig['keypoints'], data_rot['keypoints'], matches, inliers, H

# ============================================================================
# Pipeline 3: MagicLeap SuperPoint + FLANN + RANSAC
# ============================================================================
def load_magicleap_sp(max_kp=MAX_KP_SP_FLANN):
    config = {'nms_radius': 4, 'keypoint_threshold': 0.005,
              'max_keypoints': max_kp, 'remove_borders': 4}
    model = OfficialSuperPoint(config)
    w = Path(SUPERGLUE_PATH) / 'models' / 'weights' / 'superpoint_v1.pth'
    state = torch.load(w, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_magicleap_sp(model, image, max_dim=MAX_DIM):
    # downscale
    h, w = image.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    # divisible by 8
    h, w = image.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h))
    tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model({'image': tensor})
    t = time.time() - t0
    kp = out['keypoints'][0].cpu().numpy()
    desc = out['descriptors'][0].cpu().numpy().T   # [N,256]
    scores = out['scores'][0].cpu().numpy()
    # already limited by max_keypoints
    print(f"   MagicLeap SP: {len(kp)} keypoints, {t*1000:.1f} ms")
    return {'keypoints': kp, 'descriptors': desc, 'scores': scores, 'num': len(kp)}

def pipeline_magicleap_flann(sp_model, image_orig, image_rot):
    data_orig = extract_magicleap_sp(sp_model, image_orig)
    data_rot  = extract_magicleap_sp(sp_model, image_rot)
    flann = get_flann_matcher()
    matches = match_flann(data_orig['descriptors'], data_rot['descriptors'], flann)
    inliers, H = ransac_homography(matches, data_orig['keypoints'], data_rot['keypoints'])
    return data_orig['keypoints'], data_rot['keypoints'], matches, inliers, H

# ============================================================================
# Pipeline 4: MagicLeap SuperPoint + SuperGlue (official)
# ============================================================================
def load_magicleap_superglue(model_type='outdoor'):
    config = {'descriptor_dim':256, 'keypoint_encoder':[32,64,128,256],
              'GNN_layers':['self','cross']*9, 'sinkhorn_iterations':100, 'match_threshold':0.2}
    model = SuperGlue(config)
    if model_type == 'indoor':
        w = Path(SUPERGLUE_PATH) / 'models' / 'weights' / 'superglue_indoor.pth'
    else:
        w = Path(SUPERGLUE_PATH) / 'models' / 'weights' / 'superglue_outdoor.pth'
    state = torch.load(w, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    return model

def extract_magicleap_sp_for_sg(model, image, max_dim=MAX_DIM):
    orig_h, orig_w = image.shape
    # downscale
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = cv2.resize(image, (new_w, new_h))
    else:
        scale = 1.0
        new_h, new_w = orig_h, orig_w

    # divisible by 8
    h, w = image.shape
    new_h = (h // 8) * 8
    new_w = (w // 8) * 8
    if new_h != h or new_w != w:
        image = cv2.resize(image, (new_w, new_h))
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
    else:
        scale_x = orig_w / w
        scale_y = orig_h / h

    tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        out = model({'image': tensor})
    t = time.time() - t0
    kp = out['keypoints'][0].cpu().numpy()
    desc = out['descriptors'][0].cpu().numpy().T
    scores = out['scores'][0].cpu().numpy()
    # limit
    if len(kp) > MAX_KP_SP_SG:
        idx = np.argsort(scores)[-MAX_KP_SP_SG:]
        kp = kp[idx]
        desc = desc[idx]
        scores = scores[idx]
    kp_orig = kp.copy()
    kp_orig[:, 0] *= scale_x
    kp_orig[:, 1] *= scale_y
    print(f"   MagicLeap SP (for SG): {len(kp)} keypoints, {t*1000:.1f} ms")
    return {'keypoints': kp_orig, 'descriptors': desc, 'scores': scores, 'num': len(kp_orig), 'tensor': tensor, 'scale_x': scale_x, 'scale_y': scale_y}

def match_superglue(sg_model, data0, data1, img_tensor0, img_tensor1):
    # prepare tensors
    kp0 = torch.from_numpy(data0['keypoints']).float().unsqueeze(0).to(DEVICE)
    scr0 = torch.from_numpy(data0['scores']).float().unsqueeze(0).to(DEVICE)
    desc0 = torch.from_numpy(data0['descriptors']).float().unsqueeze(0).permute(0,2,1).contiguous().to(DEVICE)
    kp1 = torch.from_numpy(data1['keypoints']).float().unsqueeze(0).to(DEVICE)
    scr1 = torch.from_numpy(data1['scores']).float().unsqueeze(0).to(DEVICE)
    desc1 = torch.from_numpy(data1['descriptors']).float().unsqueeze(0).permute(0,2,1).contiguous().to(DEVICE)

    pred = {
        'keypoints0': kp0, 'scores0': scr0, 'descriptors0': desc0, 'image0': img_tensor0,
        'keypoints1': kp1, 'scores1': scr1, 'descriptors1': desc1, 'image1': img_tensor1,
    }
    t0 = time.time()
    with torch.no_grad():
        out = sg_model(pred)
    t = time.time() - t0
    matches0 = out['matches0'][0].cpu().numpy()
    match_scores = out['matching_scores0'][0].cpu().numpy()
    matches = []
    for i, j in enumerate(matches0):
        if j > -1:
            matches.append(cv2.DMatch(i, int(j), 0, 1.0 - match_scores[i]))
    print(f"   SuperGlue matching: {t*1000:.1f} ms, {len(matches)} matches")
    return matches

def pipeline_magicleap_superglue(sp_model, sg_model, image_orig, image_rot):
    data_orig = extract_magicleap_sp_for_sg(sp_model, image_orig)
    data_rot  = extract_magicleap_sp_for_sg(sp_model, image_rot)
    matches = match_superglue(sg_model, data_orig, data_rot, data_orig['tensor'], data_rot['tensor'])
    # compute homography from the matches (RANSAC)
    _, H = ransac_homography(matches, data_orig['keypoints'], data_rot['keypoints'])
    return data_orig['keypoints'], data_rot['keypoints'], matches, matches, H   # inliers = matches (all are used for homography)

# ============================================================================
# Pipeline 5: SIFT + SuperGlue (neu)
# ============================================================================
def extract_sift_for_sg(image, max_kp=MAX_KP_SP_SG, max_dim=MAX_DIM):
    """Extrahiert SIFT‑Keypoints und ‑Deskriptoren, padet auf 256 Dimensionen,
       und erzeugt den Bild‑Tensor für SuperGlue."""
    orig_h, orig_w = image.shape

    # Downscaling (optional, analog zu SuperPoint)
    if max(orig_h, orig_w) > max_dim:
        scale = max_dim / max(orig_h, orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = cv2.resize(image, (new_w, new_h))
    else:
        scale = 1.0
        new_h, new_w = orig_h, orig_w

    # SIFT‑Extraktion
    sift = cv2.SIFT_create(nfeatures=max_kp)
    kp, desc = sift.detectAndCompute(image, None)
    if kp is None:
        return None
    kp_arr = np.array([[p.pt[0], p.pt[1]] for p in kp])
    scores = np.array([p.response for p in kp])

    # Keypoints begrenzen
    if len(kp_arr) > max_kp:
        idx = np.argsort(scores)[-max_kp:]
        kp_arr = kp_arr[idx]
        desc = desc[idx]
        scores = scores[idx]

    # Skalierung zurück auf Originalgröße
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h
    kp_orig = kp_arr.copy()
    kp_orig[:, 0] *= scale_x
    kp_orig[:, 1] *= scale_y

    # Deskriptoren auf 256 Dimensionen auffüllen (mit Nullen)
    if desc.shape[1] < 256:
        pad = np.zeros((desc.shape[0], 256 - desc.shape[1]), dtype=desc.dtype)
        desc = np.hstack([desc, pad])

    # Bild‑Tensor für SuperGlue (wichtig für Normalisierung)
    tensor = torch.from_numpy(image.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print(f"   SIFT (for SG): {len(kp_orig)} keypoints")
    return {'keypoints': kp_orig, 'descriptors': desc, 'scores': scores,
            'num': len(kp_orig), 'tensor': tensor,
            'scale_x': scale_x, 'scale_y': scale_y}

def pipeline_sift_superglue(sg_model, image_orig, image_rot):
    """SIFT + SuperGlue: extrahiert SIFT‑Keypoints und führt Matching mit SuperGlue durch."""
    data_orig = extract_sift_for_sg(image_orig)
    data_rot  = extract_sift_for_sg(image_rot)
    if data_orig is None or data_rot is None:
        return None, None, None, None, None
    matches = match_superglue(sg_model, data_orig, data_rot, data_orig['tensor'], data_rot['tensor'])
    _, H = ransac_homography(matches, data_orig['keypoints'], data_rot['keypoints'])
    return data_orig['keypoints'], data_rot['keypoints'], matches, matches, H

# ============================================================================
# Main
# ============================================================================
def main():
    image_path = project_root / 'Images' / 'HEstain.png'


    print("=" * 80)
    print("Five‑way Registration Comparison (inkl. SIFT + SuperGlue)")
    print("=" * 80)

    # Create output directory
    out_dir = project_root / 'results' / 'python' / 'fiveway_comparison' / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # Load original image
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        raise FileNotFoundError(image_path)
    print(f"\nOriginal image: {img_orig.shape}")

    # Grid points (for evaluation)
    grid_points = select_grid_points(img_orig, grid_rows=4, grid_cols=5, margin=50)

    # Rotate image (ground truth)
    img_rot, gt_points, _ = rotate_image_and_points(img_orig, grid_points, ROTATION_ANGLE)

    # ------------------------------------------------------------------------
    # Pipeline 1: SIFT + FLANN + RANSAC
    # ------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Pipeline 1: SIFT + FLANN + RANSAC")
    print("-"*70)
    kp1, kp2, matches, inliers, H = pipeline_sift(img_orig, img_rot)
    errors1, trans1 = evaluate_grid_points(grid_points, gt_points, H)
    draw_matches(img_orig, img_rot, kp1, kp2, matches, inliers,
                 "SIFT + FLANN + RANSAC", out_dir / "sift_matches.png")
    draw_grid_evaluation(img_orig, img_rot, grid_points, gt_points, trans1, errors1,
                         "SIFT + FLANN + RANSAC", out_dir / "sift_grid.png")

    # ------------------------------------------------------------------------
    # Pipeline 2: rpautrat SuperPoint + FLANN + RANSAC
    # ------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Pipeline 2: rpautrat SuperPoint + FLANN + RANSAC")
    print("-"*70)
    sp_model = load_rpautrat_sp()
    kp1, kp2, matches, inliers, H = pipeline_rpautrat_sp(sp_model, img_orig, img_rot)
    errors2, trans2 = evaluate_grid_points(grid_points, gt_points, H)
    draw_matches(img_orig, img_rot, kp1, kp2, matches, inliers,
                 "rpautrat SP + FLANN + RANSAC", out_dir / "rpautrat_sp_matches.png")
    draw_grid_evaluation(img_orig, img_rot, grid_points, gt_points, trans2, errors2,
                         "rpautrat SP + FLANN + RANSAC", out_dir / "rpautrat_sp_grid.png")

    # ------------------------------------------------------------------------
    # Pipeline 3: MagicLeap SuperPoint + FLANN + RANSAC
    # ------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Pipeline 3: MagicLeap SuperPoint + FLANN + RANSAC")
    print("-"*70)
    ml_sp = load_magicleap_sp(max_kp=MAX_KP_SP_FLANN)
    kp1, kp2, matches, inliers, H = pipeline_magicleap_flann(ml_sp, img_orig, img_rot)
    errors3, trans3 = evaluate_grid_points(grid_points, gt_points, H)
    draw_matches(img_orig, img_rot, kp1, kp2, matches, inliers,
                 "MagicLeap SP + FLANN + RANSAC", out_dir / "ml_sp_flann_matches.png")
    draw_grid_evaluation(img_orig, img_rot, grid_points, gt_points, trans3, errors3,
                         "MagicLeap SP + FLANN + RANSAC", out_dir / "ml_sp_flann_grid.png")

    # ------------------------------------------------------------------------
    # Pipeline 4: MagicLeap SuperPoint + SuperGlue (official)
    # ------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Pipeline 4: MagicLeap SuperPoint + SuperGlue")
    print("-"*70)
    ml_sp_sg = load_magicleap_sp(max_kp=MAX_KP_SP_SG)   # reuse same SP model
    for sg_type in ['outdoor', 'indoor']:
        print(f"\n   SuperGlue model: {sg_type}")
        sg_model = load_magicleap_superglue(sg_type)
        kp1, kp2, matches, inliers, H = pipeline_magicleap_superglue(ml_sp_sg, sg_model, img_orig, img_rot)
        errors, trans = evaluate_grid_points(grid_points, gt_points, H)
        if sg_type == 'outdoor':
            errors4_out, trans4_out = errors, trans
            matches4_out = matches
            inliers4_out = inliers
            kp1_out, kp2_out = kp1, kp2
        else:
            errors4_in, trans4_in = errors, trans
            matches4_in = matches
            inliers4_in = inliers
            kp1_in, kp2_in = kp1, kp2

        draw_matches(img_orig, img_rot, kp1, kp2, matches, inliers,
                     f"ML SP + SuperGlue ({sg_type})", out_dir / f"ml_sp_sg_{sg_type}_matches.png")
        draw_grid_evaluation(img_orig, img_rot, grid_points, gt_points, trans, errors,
                             f"ML SP + SuperGlue ({sg_type})", out_dir / f"ml_sp_sg_{sg_type}_grid.png")

    # ------------------------------------------------------------------------
    # Pipeline 5: SIFT + SuperGlue (neu)
    # ------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Pipeline 5: SIFT + SuperGlue")
    print("-"*70)
    for sg_type in ['outdoor', 'indoor']:
        print(f"\n   SuperGlue model: {sg_type}")
        sg_model = load_magicleap_superglue(sg_type)   # gleiche Ladefunktion
        kp1, kp2, matches, inliers, H = pipeline_sift_superglue(sg_model, img_orig, img_rot)
        if H is None:
            print(f"   SIFT + SuperGlue ({sg_type}) – keine Homographie berechnet.")
            continue
        errors, trans = evaluate_grid_points(grid_points, gt_points, H)
        draw_matches(img_orig, img_rot, kp1, kp2, matches, inliers,
                     f"SIFT + SuperGlue ({sg_type})", out_dir / f"sift_sg_{sg_type}_matches.png")
        draw_grid_evaluation(img_orig, img_rot, grid_points, gt_points, trans, errors,
                             f"SIFT + SuperGlue ({sg_type})", out_dir / f"sift_sg_{sg_type}_grid.png")
        if sg_type == 'outdoor':
            errors_sift_out, trans_sift_out = errors, trans
        else:
            errors_sift_in, trans_sift_in = errors, trans

    # ------------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY OF GRID POINT ERRORS (pixels)")
    print("=" * 80)
    print(f"\n{'Method':<40} {'Mean':>8} {'Std':>8} {'Max':>8} {'<10px':>10}")
    print("-" * 80)
    def fmt(arr):
        return f"{arr.mean():6.2f} {arr.std():6.2f} {arr.max():6.2f} {np.sum(arr < 10):4d}/{len(arr)}"
    print(f"{'SIFT + FLANN + RANSAC':<40} {fmt(errors1)}")
    print(f"{'rpautrat SP + FLANN + RANSAC':<40} {fmt(errors2)}")
    print(f"{'MagicLeap SP + FLANN + RANSAC':<40} {fmt(errors3)}")
    print(f"{'MagicLeap SP + SuperGlue (outdoor)':<40} {fmt(errors4_out)}")
    print(f"{'MagicLeap SP + SuperGlue (indoor)':<40} {fmt(errors4_in)}")
    if 'errors_sift_out' in locals():
        print(f"{'SIFT + SuperGlue (outdoor)':<40} {fmt(errors_sift_out)}")
    if 'errors_sift_in' in locals():
        print(f"{'SIFT + SuperGlue (indoor)':<40} {fmt(errors_sift_in)}")

    # Save all results
    save_dict = {
        'grid_points': grid_points, 'ground_truth': gt_points,
        'sift_errors': errors1, 'sift_transformed': trans1,
        'rpautrat_errors': errors2, 'rpautrat_transformed': trans2,
        'ml_flann_errors': errors3, 'ml_flann_transformed': trans3,
        'ml_sg_out_errors': errors4_out, 'ml_sg_out_transformed': trans4_out,
        'ml_sg_in_errors': errors4_in, 'ml_sg_in_transformed': trans4_in,
    }
    if 'errors_sift_out' in locals():
        save_dict['sift_sg_out_errors'] = errors_sift_out
        save_dict['sift_sg_out_transformed'] = trans_sift_out
    if 'errors_sift_in' in locals():
        save_dict['sift_sg_in_errors'] = errors_sift_in
        save_dict['sift_sg_in_transformed'] = trans_sift_in

    np.savez(out_dir / "results.npz", **save_dict)

    print(f"\n✅ All results saved to: {out_dir}")
    print("\n✨ Done!")

if __name__ == "__main__":
    main()