"""
camera_calibration.py

Provides the forward pinhole camera models for each view, relying on 
assumed room geometry and a self-calibrated least-squares optimization 
over annotated floor points to resolve the true Look-At targets and FOVs.

Outputs the calibrated camera intrinsics (K), extrinsics (R, t), and 
computes the View Quality (VQ) scores for every detection based on 
the 3D spatial relationship between the camera and the subject.

Usage:
    python -m src.camera_calibration
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Group 3 — local / project
from config import DATA_DIR

# --- ASSUMPTIONS & CALIBRATED CONSTANTS ---
# We assume a 7m x 7m room.
# We assume camera mounting height is 1.7m (approximate human height).
# Look-at targets and FOVs were derived via self-calibration (least_squares)
# to minimize cross-view projection error of ground-truth floor corners.
CAMERAS = {
    "view1": {
        "pos": np.array([0.0, 0.0, 1.7]),
        "target": np.array([4.26, 3.84, 0.0]),
        "fov_deg": 60.00
    },
    "view2": {
        "pos": np.array([0.0, 7.0, 1.7]),
        "target": np.array([5.61, 3.47, 0.0]),
        "fov_deg": 60.00
    },
    "view3": {
        "pos": np.array([7.0, 7.0, 1.7]),
        "target": np.array([3.52, 2.10, 0.0]),
        "fov_deg": 79.26
    }
}

FRAME_WIDTH = 360
FRAME_HEIGHT = 288
MANIFESTS_DIR = Path(__file__).resolve().parent.parent / "data_manifests"


def get_lookat_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])) -> tuple[np.ndarray, np.ndarray]:
    """Computes the rotation matrix R and translation t for a camera at `eye` looking at `target`."""
    forward = target - eye
    norm_f = np.linalg.norm(forward)
    if norm_f < 1e-6:
        forward = np.array([0, 0, -1])
    else:
        forward = forward / norm_f
        
    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right = right / norm_r
        
    down = np.cross(forward, right)
    
    R = np.vstack([right, down, forward])
    t = -R @ eye
    return R, t


def get_K(W: int, H: int, fov_deg: float) -> np.ndarray:
    """Computes the Intrinsic matrix K assuming the principal point is at the image center."""
    diag = np.sqrt(W**2 + H**2)
    f = (diag / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    return np.array([
        [f, 0, W/2.0],
        [0, f, H/2.0],
        [0, 0, 1.0]
    ])


def get_camera_model(view_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (K, R, t, pos) for the given view."""
    cam = CAMERAS[view_name]
    K = get_K(FRAME_WIDTH, FRAME_HEIGHT, cam["fov_deg"])
    R, t = get_lookat_matrix(cam["pos"], cam["target"])
    return K, R, t, cam["pos"]


def ray_intersect_z0(K: np.ndarray, R: np.ndarray, t: np.ndarray, u: float, v: float) -> np.ndarray | None:
    """Ray-casts a pixel (u,v) onto the Z=0 floor plane. Returns (X,Y) or None if looking away."""
    ray_c = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_w = np.linalg.inv(R) @ ray_c
    C = -np.linalg.inv(R) @ t
    
    if abs(ray_w[2]) < 1e-6:
        return None
        
    l = -C[2] / ray_w[2]
    if l < 0:
        return None
        
    P = C + l * ray_w
    return P[:2]


def compute_vq_scores() -> None:
    """
    Computes real VQ terms for every detection across all views.
    Q_i: Detection confidence + edge check (gates the expression).
    D_i: Euclidean distance from camera XY to subject floor XY.
    phi_i: angle between camera optical axis and ray to subject.
    theta_i: angle between subject movement direction and optical axis.
    
    INTENTIONAL DESIGN CHOICES & LIMITATIONS:
    1. Distance Term (D_i): We use a triangular/linear distance formula peaked at 
       D_Bi=2.5m: max(0, 1 - |D_i - 2.5| / 2.5). This is a deliberate deviation from 
       the base paper's literal linear form (1 - D_i/D_Bi) because a symmetric 
       "ideal distance" penalizing both too-close and too-far is more sensible for 
       surveillance quality (extreme close-ups aren't always better views).
    2. Theta Term (theta_i): Subject movement direction is used as a proxy for body 
       orientation. However, 60-80% of detections lack sufficient movement history 
       and fall back to theta_i=0 (facing camera). This means this term has weak 
       discriminative power for much of the dataset. This is an accepted limitation.
    3. Angle Scoring (phi_i, theta_i): We use a cosine mapping (math.cos) rather than 
       the base paper's literal linear penalty. This is a deliberate, confirmed 
       deviation chosen for its smooth falloff behavior, and has been reviewed 
       and kept as-is.
    """
    # 1. Load tracking data to build trajectory history for theta_i
    # We use the stitched tracklets if possible, or just the original detections' track_ids.
    # Since detections/viewX.csv has track_id from ByteTrack, we group by track_id.
    
    all_stats = {}
    for view in ["view1", "view2", "view3"]:
        print(f"[INFO] Computing VQ scores for {view}...")
        K, R, t, cam_pos = get_camera_model(view)
        
        # Optical axis is the 3rd row of R (the forward vector)
        optical_axis = R[2, :]
        cam_xy = cam_pos[:2]
        
        det_path = DATA_DIR / "detections" / f"{view}.csv"
        if not det_path.exists():
            continue
            
        # Group detections by track_id to compute velocities
        detections = []
        track_history = defaultdict(list)
        
        with open(det_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                detections.append(row)
                
        # First pass: map floor positions
        for det in detections:
            u = (float(det["x1"]) + float(det["x2"])) / 2.0
            v = float(det["y2"]) # bottom-center pixel
            floor_pt = ray_intersect_z0(K, R, t, u, v)
            det["floor_pt"] = floor_pt
            
            track_id = int(det.get("track_id", -1))
            f_idx = int(det["frame_idx"])
            if track_id != -1 and floor_pt is not None:
                track_history[track_id].append((f_idx, floor_pt))
                
        # Sort histories by frame
        for tid in track_history:
            track_history[tid].sort(key=lambda x: x[0])
            
        out_rows = []
        for det in detections:
            # Q_i: base proxy (confidence) penalized if touching edge
            conf = float(det["confidence"])
            x1, y1, x2, y2 = float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])
            edge_penalty = 0.5 if (x1 < 5 or y1 < 5 or x2 > FRAME_WIDTH - 5 or y2 > FRAME_HEIGHT - 5) else 1.0
            Q_i = conf * edge_penalty
            
            floor_pt = det["floor_pt"]
            if floor_pt is None:
                # Fallback if projection fails
                D_i, phi_i, theta_i = 10.0, 90.0, 0.0
                vq_score = 0.0
            else:
                # D_i: 2D distance on floor (ignoring height difference as a simplification)
                D_i = float(np.linalg.norm(floor_pt - cam_xy))
                
                # phi_i: angle to optical axis
                ray = np.array([floor_pt[0], floor_pt[1], 0.0]) - cam_pos
                ray = ray / np.linalg.norm(ray)
                cos_phi = np.dot(ray, optical_axis)
                phi_i = float(np.degrees(np.arccos(np.clip(cos_phi, -1.0, 1.0))))
                
                # theta_i: movement direction vs optical axis
                track_id = int(det.get("track_id", -1))
                f_idx = int(det["frame_idx"])
                theta_i = 0.0 # Default if stationary
                
                if track_id != -1:
                    # Find a previous frame to compute velocity (e.g. up to 10 frames ago)
                    hist = track_history[track_id]
                    prev_pt = None
                    for h_f, h_pt in reversed(hist):
                        if h_f < f_idx and (f_idx - h_f) <= 15:
                            prev_pt = h_pt
                            break
                            
                    if prev_pt is not None:
                        move_vec = floor_pt - prev_pt
                        move_norm = np.linalg.norm(move_vec)
                        if move_norm > 0.1: # moved at least 10cm
                            move_dir = np.array([move_vec[0], move_vec[1], 0.0]) / move_norm
                            # We assume the subject faces their direction of movement.
                            # We want the angle between the subject's face normal (move_dir) and the vector TO the camera.
                            vec_to_cam = cam_pos - np.array([floor_pt[0], floor_pt[1], 0.0])
                            vec_to_cam = vec_to_cam / np.linalg.norm(vec_to_cam)
                            cos_theta = np.dot(move_dir, vec_to_cam)
                            theta_i = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
                        else:
                            # Stationary/no clear movement -> fallback
                            theta_i = 0.0
                
                # VQ Formula Synthesis
                # Q_i GATES the expression: if occluded/cut-off, VQ collapses.
                # Equal weights assumption: 1/3 each for proxy terms.
                
                # For distance: The linear form peaking at D_B = 2.5m
                score_D = max(0.0, 1.0 - abs(D_i - 2.5) / 2.5)
                
                # For phi (camera centering): 0 deg is best. Cosine mapping.
                score_phi = max(0.0, math.cos(math.radians(phi_i)))
                
                # For theta (subject facing camera): 0 deg is best (looking directly at camera).
                score_theta = max(0.0, math.cos(math.radians(theta_i)))
                
                # Final VQ: Q_i * (omega_theta*theta + omega_phi*phi + omega_l*D)
                vq_score = Q_i * ((1/3) * score_theta + (1/3) * score_phi + (1/3) * score_D)
                
                
            out_rows.append({
                "frame_idx": det["frame_idx"],
                "track_id": det.get("track_id", -1),
                "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                "confidence": det["confidence"],
                "Q_i": round(Q_i, 4),
                "D_i": round(D_i, 4),
                "phi_i": round(phi_i, 4),
                "theta_i": round(theta_i, 4),
                "vq_score": round(vq_score, 4)
            })
            
        # Write out to data_manifests
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MANIFESTS_DIR / f"vq_scores_{view}.csv"
        
        fieldnames = ["frame_idx", "track_id", "x1", "y1", "x2", "y2", "confidence", "Q_i", "D_i", "phi_i", "theta_i", "vq_score"]
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
            
        print(f"[DONE] Wrote {len(out_rows)} VQ scores to {out_path.name}")
        
        # Stats collection
        fallback_count = sum(1 for r in out_rows if r["theta_i"] == 0.0)
        print(f"  -> theta_i fallback (stationary/no-history): {fallback_count} detections")
        all_stats[view] = out_rows
        
    # Aggregate and print global stats
    print("\n--- GLOBAL STATS ---")
    keys = ["theta_i", "phi_i", "D_i", "vq_score"]
    for k in keys:
        vals = [r[k] for view_rows in all_stats.values() for r in view_rows]
        if vals:
            print(f"  {k}: min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}")

if __name__ == "__main__":
    compute_vq_scores()
