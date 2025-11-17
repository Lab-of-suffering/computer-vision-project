"""
Self-calibration core implementation based on Structure-from-Motion.
Full implementation matching self_calibration.ipynb
Estimates camera intrinsics from a sequence of images without calibration patterns.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

try:
    from scipy.optimize import least_squares
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


@dataclass
class FrameData:
    """Container for storing keypoints and descriptors per frame"""
    kps: List[cv2.KeyPoint]
    desc: np.ndarray
    pts_px: np.ndarray


def preprocess_gray(g: np.ndarray) -> np.ndarray:
    """Apply CLAHE to enhance image contrast"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(g)


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def create_feature_detector(feature_type: str = "sift", nfeatures: int = 3000):
    """Create SIFT or ORB feature detector"""
    if feature_type.lower() == "sift":
        try:
            return cv2.SIFT_create(nfeatures=nfeatures)
        except Exception:
            pass
    return cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=10)


def match_descriptors(d1: np.ndarray, d2: np.ndarray, 
                     ratio: float = 0.80, min_matches: int = 50) -> List[cv2.DMatch]:
    """
    Perform two-way descriptor matching with Lowe's ratio test
    """
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return []

    is_bin = (d1.dtype == np.uint8 and d2.dtype == np.uint8)
    
    if is_bin:
        matcher_fwd = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matcher_rev = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        d1 = np.asarray(d1, dtype=np.float32)
        d2 = np.asarray(d2, dtype=np.float32)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=64)
        matcher_fwd = cv2.FlannBasedMatcher(index_params, search_params)
        matcher_rev = cv2.FlannBasedMatcher(index_params, search_params)

    # Forward KNN matching with Lowe's ratio test
    knn = matcher_fwd.knnMatch(d1, d2, k=2)
    ratio_pass = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            ratio_pass.append(m)

    if not ratio_pass:
        return []

    # Reverse matching for mutual consistency
    knn_rev = matcher_rev.knnMatch(d2, d1, k=1)
    rev_map = {}
    for pair in knn_rev:
        if not pair:
            continue
        r = pair[0]
        rev_map[r.queryIdx] = r.trainIdx

    # Keep only mutual matches
    mutual = []
    seen = set()
    for m in ratio_pass:
        if rev_map.get(m.trainIdx, -1) == m.queryIdx:
            key = (m.queryIdx, m.trainIdx)
            if key not in seen:
                seen.add(key)
                mutual.append(m)

    if len(mutual) < min_matches:
        return []

    return mutual


def ransac_F(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Robust estimation of Fundamental matrix using RANSAC"""
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                     ransacReprojThreshold=3.0, 
                                     confidence=0.995, maxIters=4000)
    if F is None or F.shape != (3, 3):
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC,
                                         ransacReprojThreshold=3.0, 
                                         confidence=0.995, maxIters=8000)
    mask = (mask.ravel().astype(bool) if mask is not None else np.zeros(len(pts1), bool))
    return F, mask


def estimate_focal_from_F_list(F_list: List[np.ndarray], W: int, H: int, 
                               verbose: bool = False) -> float:
    """
    Estimate focal length from multiple Fundamental matrices using SVD scoring
    """
    if len(F_list) < 3:
        f_heur = 1.2 * max(W, H)
        if verbose:
            print(f"[Focal] Not enough F-matrices ({len(F_list)}), fallback f≈{f_heur:.1f}")
        return float(f_heur)

    maxWH = float(max(W, H))

    def score_for_f(f):
        K = np.array([[f, 0.0, W * 0.5],
                     [0.0, f, H * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
        
        total = 0.0
        count = 0
        for F in F_list:
            E = K.T @ F @ K
            U, S, Vt = np.linalg.svd(E)
            s1, s2, s3 = S
            
            if s1 <= 0.0 or s2 <= 0.0:
                continue
            
            num = (s1 - s2) ** 2 + s3 ** 2
            den = s1 * s2
            total += num / den
            count += 1
        
        if count == 0:
            return np.inf
        return total / count

    # Coarse grid search
    f_min = 0.8 * maxWH
    f_max = 1.8 * maxWH
    n_samples = 50
    
    f_grid = np.linspace(f_min, f_max, n_samples)
    scores = [score_for_f(f) for f in f_grid]
    
    scores = np.asarray(scores, dtype=np.float64)
    best_idx = int(np.argmin(scores))
    f_best = float(f_grid[best_idx])

    if verbose:
        print(f"[Focal] coarse search [{f_min:.1f}, {f_max:.1f}] px, "
              f"best≈{f_best:.2f}, score={scores[best_idx]:.3e}")

    # Local refinement
    width = (f_max - f_min) / n_samples * 4.0
    f_ref_min = max(f_min, f_best - width)
    f_ref_max = min(f_max, f_best + width)
    
    n_ref = 30
    f_ref_grid = np.linspace(f_ref_min, f_ref_max, n_ref)
    ref_scores = [score_for_f(f) for f in f_ref_grid]
    
    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    best_idx2 = int(np.argmin(ref_scores))
    f_opt = float(f_ref_grid[best_idx2])

    if verbose:
        print(f"[Focal] refine [{f_ref_min:.1f}, {f_ref_max:.1f}] px, "
              f"f≈{f_opt:.2f}, score={ref_scores[best_idx2]:.3e}")

    return float(f_opt)


def triangulate_points(P0: np.ndarray, P1: np.ndarray, 
                      x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Triangulate 3D points from corresponding normalized image coordinates"""
    x0_h = np.vstack([x0.T, np.ones((1, x0.shape[0]))])
    x1_h = np.vstack([x1.T, np.ones((1, x1.shape[0]))])
    X_h = cv2.triangulatePoints(P0, P1, x0_h[:2, :], x1_h[:2, :])
    X = (X_h[:3, :] / X_h[3, :]).T.copy()
    return X


def project_points(X: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                  K: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using camera intrinsics and extrinsics"""
    img_pts, _ = cv2.projectPoints(X.astype(np.float32),
                                   rvec.reshape(3, 1).astype(np.float32),
                                   tvec.reshape(3, 1).astype(np.float32),
                                   K.astype(np.float32), None)
    return img_pts.reshape(-1, 2)


def pinhole_project_batch(f: float, cx: float, cy: float,
                          R: np.ndarray, t: np.ndarray,
                          X: np.ndarray) -> np.ndarray:
    """Fast vectorized pinhole projection"""
    X = np.asarray(X, np.float64)
    R = np.asarray(R, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3,)
    Xc = (R @ X.T).T + t
    z = np.clip(Xc[:, 2:3], 1e-9, None)
    u = Xc[:, :2] / z
    u = u * float(f) + np.array([cx, cy], dtype=np.float64)
    return u


def _proj_err_px(K, rvec, tvec, X, m):
    """Compute projection error in pixels, matching notebook implementation exactly"""
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
    Xc = (R @ X.T + np.asarray(tvec, np.float64).reshape(3, 1)).T
    z = Xc[:, 2:3]
    good = z[:, 0] > 1e-6
    u = Xc[:, :2] / np.clip(z, 1e-6, None)
    u = u * K[0, 0] + np.array([K[0, 2], K[1, 2]], dtype=np.float64)
    err = np.linalg.norm(u - m, axis=1)
    return err, good


def build_clean_subset(observations, poses, points3d, K,
                      per_frame_cap=800,
                      err_thresh=3.0):
    """Return filtered observations: list of (pid, fi, m_px) only for inliers."""
    kept = []
    for pid, obs in observations.items():
        if pid not in points3d:
            continue
        X = np.asarray(points3d[pid], np.float64).reshape(1, 3)
        # Group obs by frame
        byf = {}
        for (fi, m) in obs:
            if fi in poses:
                byf.setdefault(int(fi), []).append(np.asarray(m, np.float64).reshape(2,))
        for fi, ms in byf.items():
            rvec, tvec = poses[fi]
            M = np.stack(ms, axis=0)
            Xrep = np.repeat(X, len(ms), axis=0)
            err, good = _proj_err_px(K, rvec, tvec, Xrep, M)
            idx = np.where((good) & (err < err_thresh))[0]
            if idx.size:
                if idx.size > per_frame_cap:
                    idx = idx[:per_frame_cap]
                for j in idx:
                    kept.append((pid, fi, M[j]))
    return kept


def ba_refine_intrinsics_only(observations, poses, points3d, f0, cx0, cy0, W, H,
                               clean_obs=None, max_obs_total=12000, verbose=True):
    """Refine intrinsics only using Bundle Adjustment"""
    if not SCIPY_OK:
        if verbose:
            print("[BA intr] SciPy NA")
        return f0, cx0, cy0

    # Use pre-cleaned obs if available; else fallback to raw
    triplets = []
    seen = 0
    if clean_obs is not None:
        for (pid, fi, m_px) in clean_obs:
            if pid in points3d and fi in poses:
                triplets.append((pid, fi, m_px))
                seen += 1
                if seen >= max_obs_total:
                    break
    
    if not triplets:
        # fallback: raw, very small cap
        for pid, obs in observations.items():
            if pid not in points3d:
                continue
            for (fi, m_px) in obs:
                if fi not in poses:
                    continue
                triplets.append((pid, fi, np.asarray(m_px, np.float64)))
                if len(triplets) >= min(4000, max_obs_total):
                    break
            if len(triplets) >= min(4000, max_obs_total):
                break

    if len(triplets) < 200:
        if verbose:
            print(f"[BA intr] too few obs: {len(triplets)}")
        return f0, cx0, cy0

    # Group by camera for batch projection
    by_cam = {}
    for pid, fi, m in triplets:
        by_cam.setdefault(int(fi), {"X": [], "m": []}).update({})
        by_cam[int(fi)]["X"].append(points3d[pid])
        by_cam[int(fi)]["m"].append(m)

    cam_ids = sorted(by_cam.keys())
    R_list, t_list, X_batches, m_batches = [], [], [], []
    for fid in cam_ids:
        rvec, tvec = poses[fid]
        R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))
        R_list.append(R)
        t_list.append(np.asarray(tvec, np.float64).reshape(3,))
        X_batches.append(np.asarray(by_cam[fid]["X"], np.float64))
        m_batches.append(np.asarray(by_cam[fid]["m"], np.float64))

    f0 = float(f0)
    cx0 = float(cx0)
    cy0 = float(cy0)
    maxWH = float(max(W, H))

    def residuals(theta):
        f, cx, cy = theta
        res = []
        for R, t, Xb, Mb in zip(R_list, t_list, X_batches, m_batches):
            proj = pinhole_project_batch(f, cx, cy, R, t, Xb)
            res.append((proj - Mb).ravel())
        res = np.concatenate(res, axis=0)

        # Weak priors
        lam_f = 5e-6
        lam_c = 5e-5
        pri = np.array([
            lam_f * (f - f0) / maxWH,
            lam_c * (cx - cx0) / W,
            lam_c * (cy - cy0) / H
        ], dtype=np.float64)
        return np.concatenate([res, pri], axis=0)

    theta0 = np.array([f0, cx0, cy0], np.float64)
    lb = np.array([0.55 * maxWH, 0.25 * W, 0.15 * H], np.float64)
    ub = np.array([2.20 * maxWH, 0.75 * W, 0.85 * H], np.float64)

    result = least_squares(
        residuals, x0=theta0, bounds=(lb, ub),
        loss="soft_l1", f_scale=2.0,
        max_nfev=12, verbose=2 if verbose else 0
    )
    f_opt, cx_opt, cy_opt = result.x
    if verbose:
        print(f"[BA intr] f {f0:.2f}->{f_opt:.2f}, cx {cx0:.2f}->{cx_opt:.2f}, cy {cy0:.2f}->{cy_opt:.2f}")
    return float(f_opt), float(cx_opt), float(cy_opt)


def ba_refine_f_cx_cy_full(observations, poses, points3d, f0, cx0, cy0, W, H,
                           max_frames_ba=15, max_points_ba=800, max_obs_total=10000, verbose=True):
    """
    Mini full bundle-adjustment:

      - Optimizes intrinsics (f, cx, cy)
      - Optimizes poses (rvec, tvec) for a subset of cameras
      - Optimizes 3D points for a subset of tracks

    All residuals are reprojection errors in pixels computed via cv2.projectPoints.
    """
    if not SCIPY_OK:
        print("[BA] SciPy not available, skipping BA")
        return float(f0), float(cx0), float(cy0)

    if not poses or not points3d or not observations:
        print("[BA] Empty poses/points/observations, skipping BA")
        return float(f0), float(cx0), float(cy0)

    total_frames = len(poses)
    total_points = len(points3d)
    total_obs_raw = sum(len(v) for v in observations.values())
    print(f"[BA] total frames={total_frames}, total points={total_points}, total raw obs={total_obs_raw}")

    # Select subset of frames for BA
    frame_ids = sorted(int(fi) for fi in poses.keys())
    if not frame_ids:
        print("[BA] No frame ids, skipping BA")
        return float(f0), float(cx0), float(cy0)

    if len(frame_ids) > max_frames_ba:
        step = max(1, len(frame_ids) // max_frames_ba)
        frame_ids = frame_ids[::step]

    frame_set = set(frame_ids)
    base_fid = frame_ids[0]       # base camera: fixed pose
    opt_frame_ids = [fid for fid in frame_ids if fid != base_fid]
    cam_index = {fid: idx for idx, fid in enumerate(opt_frame_ids)}
    n_cams_opt = len(opt_frame_ids)

    print(f"[BA] frames selected for BA: {len(frame_ids)} (base={base_fid}, opt={n_cams_opt})")

    # Build subset of points and observations
    pt_ids = []
    X0_list = []
    obs_pt_idx = []
    obs_cam_idx = []
    obs_meas = []

    pts_used = 0
    obs_used = 0

    for pid, X in points3d.items():
        obs_full = observations.get(pid, [])
        if not obs_full:
            continue

        obs_f = [(int(fi), m) for (fi, m) in obs_full if int(fi) in frame_set]
        if len(obs_f) < 2:
            continue

        pt_idx = len(pt_ids)
        pt_ids.append(pid)
        X0_list.append(np.asarray(X, dtype=np.float64).reshape(3,))

        for (fi, m_px) in obs_f:
            obs_pt_idx.append(pt_idx)
            ci = -1 if fi == base_fid else cam_index.get(fi, -1)
            if ci == -1 and fi != base_fid:
                continue
            obs_cam_idx.append(ci)
            obs_meas.append(np.asarray(m_px, dtype=np.float64).reshape(2,))
            obs_used += 1
            if obs_used >= max_obs_total:
                break

        pts_used += 1
        if pts_used >= max_points_ba or obs_used >= max_obs_total:
            break

    if obs_used < 60 or pts_used < 30:
        if verbose:
            print(f"[BA] Not enough obs/points: pts={pts_used}, obs={obs_used}")
        return float(f0), float(cx0), float(cy0)

    X0 = np.vstack(X0_list).astype(np.float64)
    obs_meas = np.vstack(obs_meas).astype(np.float64)
    obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int32)
    obs_cam_idx = np.asarray(obs_cam_idx, dtype=np.int32)

    used_frames = sorted({base_fid} | {opt_frame_ids[ci] for ci in obs_cam_idx if ci >= 0})
    if verbose:
        print(f"[BA] frames actually used: {len(used_frames)}")

    # Initial extrinsics
    rvecs0 = np.zeros((n_cams_opt, 3), dtype=np.float64)
    tvecs0 = np.zeros((n_cams_opt, 3), dtype=np.float64)
    for fid in opt_frame_ids:
        if fid not in poses:
            continue
        rvec, tvec = poses[fid]
        rvecs0[cam_index[fid]] = np.asarray(rvec, dtype=np.float64).reshape(3,)
        tvecs0[cam_index[fid]] = np.asarray(tvec, dtype=np.float64).reshape(3,)

    f0 = float(f0)
    cx0 = float(cx0)
    cy0 = float(cy0)
    maxWH = float(max(W, H))

    if verbose:
        print(f"[BA] initial intrinsics: f={f0:.2f}, cx={cx0:.2f}, cy={cy0:.2f}")

    def pack_params(f, cx, cy, rvecs, tvecs, X):
        return np.concatenate([
            np.array([f, cx, cy], dtype=np.float64),
            rvecs.ravel(),
            tvecs.ravel(),
            X.ravel(),
        ])

    def unpack_params(theta):
        theta = np.asarray(theta, dtype=np.float64).ravel()
        f, cx, cy = theta[0], theta[1], theta[2]
        off = 3

        rvecs = rvecs0.copy()
        tvecs = tvecs0.copy()

        if n_cams_opt > 0:
            rvecs = theta[off: off + 3 * n_cams_opt].reshape(n_cams_opt, 3)
            off += 3 * n_cams_opt
            tvecs = theta[off: off + 3 * n_cams_opt].reshape(n_cams_opt, 3)
            off += 3 * n_cams_opt

        X = theta[off:].reshape(X0.shape[0], 3)
        return float(f), float(cx), float(cy), rvecs, tvecs, X

    theta0 = pack_params(f0, cx0, cy0, rvecs0, tvecs0, X0)

    lb = np.full_like(theta0, -np.inf, dtype=np.float64)
    ub = np.full_like(theta0, +np.inf, dtype=np.float64)

    f_lo, f_hi = 0.5 * maxWH, 2.5 * maxWH
    cx_lo, cx_hi = 0.1 * W, 0.9 * W
    cy_lo, cy_hi = 0.1 * H, 0.9 * H

    lb[0], ub[0] = f_lo, f_hi
    lb[1], ub[1] = cx_lo, cx_hi
    lb[2], ub[2] = cy_lo, cy_hi

    def compute_rms(f, cx, cy, rvecs, tvecs, X):
        K_loc = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        err2 = []
        for k in range(obs_meas.shape[0]):
            j = int(obs_pt_idx[k])
            ci = int(obs_cam_idx[k])

            if ci >= 0:
                r = rvecs[ci]
                t = tvecs[ci]
            else:
                r = np.zeros(3, dtype=np.float64)
                t = np.zeros(3, dtype=np.float64)

            Xj = X[j].reshape(1, 3).astype(np.float32)
            r_ = r.reshape(3, 1).astype(np.float32)
            t_ = t.reshape(3, 1).astype(np.float32)
            proj, _ = cv2.projectPoints(Xj, r_, t_, K_loc, None)
            proj = proj.reshape(2,)
            diff = proj - obs_meas[k]
            err2.append(float(diff.dot(diff)))
        return math.sqrt(sum(err2) / len(err2))

    rms0 = compute_rms(f0, cx0, cy0, rvecs0, tvecs0, X0)
    if verbose:
        print(f"[BA] initial RMS reprojection error: {rms0:.4f} px (subset)")

    # Group observations by camera for efficient residual computation
    obs_idx_by_cam = {}
    for k in range(obs_meas.shape[0]):
        ci = int(obs_cam_idx[k])
        obs_idx_by_cam.setdefault(ci, []).append(k)
    for ci in list(obs_idx_by_cam.keys()):
        obs_idx_by_cam[ci] = np.asarray(obs_idx_by_cam[ci], np.int32)

    def residuals(theta):
        f, cx, cy, rvecs, tvecs, X = unpack_params(theta)

        # Precompute rotation matrices
        Rs = np.zeros((n_cams_opt, 3, 3), np.float64)
        for i in range(n_cams_opt):
            R_i, _ = cv2.Rodrigues(rvecs[i].reshape(3, 1))
            Rs[i] = R_i

        res_chunks = []

        # Base camera (fixed)
        idx = obs_idx_by_cam.get(-1, None)
        if idx is not None and idx.size:
            j_idx = obs_pt_idx[idx]
            Xb = X[j_idx]
            Rb = np.eye(3, dtype=np.float64)
            tb = np.zeros(3, dtype=np.float64)
            proj = pinhole_project_batch(f, cx, cy, Rb, tb, Xb)
            res_chunks.append((proj - obs_meas[idx]).ravel())

        # Optimized cameras
        for ci in range(n_cams_opt):
            idx = obs_idx_by_cam.get(ci, None)
            if idx is None or not idx.size:
                continue
            j_idx = obs_pt_idx[idx]
            Xb = X[j_idx]
            proj = pinhole_project_batch(f, cx, cy, Rs[ci], tvecs[ci], Xb)
            res_chunks.append((proj - obs_meas[idx]).ravel())

        res = np.concatenate(res_chunks, axis=0) if res_chunks else np.zeros(0, np.float64)

        # Soft priors
        lam_f = 1e-5
        lam_c = 1e-4
        lam_pose = 5e-5
        lam_X = 5e-6

        pri = [
            lam_f * (f - f0) / maxWH,
            lam_c * (cx - cx0) / W,
            lam_c * (cy - cy0) / H,
        ]
        pri.extend((lam_pose * (rvecs.ravel() - rvecs0.ravel())).tolist())
        pri.extend((lam_pose * (tvecs.ravel() - tvecs0.ravel())).tolist())
        pri.extend((lam_X * (X.ravel() - X0.ravel())).tolist())

        return np.concatenate([res, np.asarray(pri, np.float64)], axis=0)

    try:
        if verbose:
            print("[BA] starting full mini-BA optimization...")
        result = least_squares(
            residuals,
            x0=theta0,
            bounds=(lb, ub),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=15,
            verbose=2 if verbose else 0,
        )
        f_opt, cx_opt, cy_opt, rvecs_opt, tvecs_opt, X_opt = unpack_params(result.x)
        rms1 = compute_rms(f_opt, cx_opt, cy_opt, rvecs_opt, tvecs_opt, X_opt)
        if verbose:
            print(f"[BA] optimization finished, status={result.status}, nfev={result.nfev}")
            print(f"[BA] intrinsics refined: f: {f0:.2f} -> {f_opt:.2f}, "
                  f"cx: {cx0:.2f} -> {cx_opt:.2f}, cy: {cy0:.2f} -> {cy_opt:.2f}")
            print(f"[BA] final RMS reprojection error: {rms1:.4f} px (subset)")

        # Update global dicts
        for fid in opt_frame_ids:
            if fid in poses:
                idx = cam_index[fid]
                poses[fid] = (rvecs_opt[idx].copy(), tvecs_opt[idx].copy())

        for local_idx, pid in enumerate(pt_ids):
            points3d[pid] = X_opt[local_idx].copy()

    except Exception as e:
        if verbose:
            print(f"[BA] least_squares failed: {e}")
        f_opt, cx_opt, cy_opt = f0, cx0, cy0

    return float(f_opt), float(cx_opt), float(cy_opt)


def self_calibrate(images: List[np.ndarray], 
                  target_width: int = 960,
                  feature_type: str = "sift",
                  stride: int = 1,
                  max_frames: int = 500,
                  verbose: bool = True) -> Dict:
    """
    Full self-calibration pipeline matching notebook implementation
    
    Args:
        images: List of BGR images
        target_width: Width to resize images for feature extraction
        feature_type: "sift" or "orb"
        stride: Frame sampling stride
        max_frames: Maximum number of frames to use
        verbose: Print progress
        
    Returns:
        Dictionary with calibration results
    """
    
    if len(images) < 10:
        raise ValueError(f"Need at least 10 images, got {len(images)}")
    
    # Sample frames
    images = images[::stride][:max_frames]
    H, W = images[0].shape[:2]
    cx, cy = W * 0.5, H * 0.5
    
    if verbose:
        print(f"[SelfCal] Processing {len(images)} frames, HxW={H}x{W}")
    
    # Extract features
    feat = create_feature_detector(feature_type)
    frames: List[FrameData] = []
    
    for img in images:
        scale = min(1.0, target_width / float(img.shape[1]))
        img_small = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                              interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        gray = preprocess_gray(to_gray(img_small))
        
        kps_small, desc = feat.detectAndCompute(gray, None)
        
        # Map keypoints back to original scale
        kps, pts_px = [], []
        inv = 1.0 / scale
        
        for kp in kps_small:
            x, y = kp.pt[0] * inv, kp.pt[1] * inv
            size = max(1.0, kp.size * inv)
            angle = float(kp.angle) if kp.angle is not None else -1.0
            new_kp = cv2.KeyPoint(float(x), float(y), float(size), angle, 
                                 float(kp.response), int(kp.octave), int(kp.class_id))
            kps.append(new_kp)
            pts_px.append([x, y])
        
        pts_px = np.array(pts_px, dtype=np.float32)
        frames.append(FrameData(kps=kps, desc=desc, pts_px=pts_px))
    
    if verbose:
        print(f"[SelfCal] Features extracted for {len(frames)} frames")
    
    # Initial K guess
    f0 = 1.2 * max(H, W)
    K = np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    
    # Find good frame pairs with valid epipolar geometry
    good_pairs = []
    F_candidates = []
    
    i = 0
    while i < len(frames) - 1 and len(good_pairs) < 60:
        for j in (i+1, min(i+2, len(frames)-1)):
            if j >= len(frames):
                continue
            
            f1, f2 = frames[i], frames[j]
            matches = match_descriptors(f1.desc, f2.desc, ratio=0.85)
            
            if len(matches) < 60:
                continue
            
            pts1 = np.array([f1.kps[m.queryIdx].pt for m in matches], np.float32)
            pts2 = np.array([f2.kps[m.trainIdx].pt for m in matches], np.float32)
            F, mask = ransac_F(pts1, pts2)
            
            inl = int(mask.sum())
            
            if F is not None and inl >= 30 and np.linalg.cond(F) < 1e20:
                good = [matches[k] for k in range(len(matches)) if mask[k]]
                
                # Check parallax
                pts1_u = cv2.undistortPoints(pts1[mask].reshape(-1, 1, 2), K, None).reshape(-1, 2)
                pts2_u = cv2.undistortPoints(pts2[mask].reshape(-1, 1, 2), K, None).reshape(-1, 2)
                
                E_tmp = K.T @ F @ K
                _, Rtmp, ttmp, _ = cv2.recoverPose(E_tmp, 
                                                    pts1_u.reshape(-1, 1, 2),
                                                    pts2_u.reshape(-1, 1, 2))
                
                b1 = np.column_stack([pts1_u, np.ones(len(pts1_u))])
                b2 = (Rtmp @ np.column_stack([pts2_u, np.ones(len(pts2_u))]).T).T
                b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
                b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
                
                ang = np.degrees(np.arccos(np.clip(np.sum(b1 * b2, axis=1), -1, 1)))
                if np.median(ang) < 1.5:
                    continue
                
                good_pairs.append((i, j, good, F))
                F_candidates.append(F.copy())
                i = j
                break
        else:
            i += 1
    
    if len(good_pairs) < 5:
        raise ValueError(f"Not enough good frame pairs found: {len(good_pairs)}. Try more/better images.")
    
    if verbose:
        print(f"[SelfCal] Found {len(good_pairs)} good frame pairs")
    
    # Estimate focal length from F-matrices
    f_init = estimate_focal_from_F_list(F_candidates, W, H, verbose=verbose)
    
    fx = fy = float(f_init)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    
    if verbose:
        print(f"[SelfCal] Initial intrinsics: f≈{fx:.1f} px, cx={cx:.1f}, cy={cy:.1f}")
    
    # === RECONSTRUCTION PHASE ===
    # Initialize seed pair with sufficient parallax
    i0, j0, matches0, F01 = good_pairs[0]
    pts1 = np.array([frames[i0].kps[m.queryIdx].pt for m in matches0], np.float32)
    pts2 = np.array([frames[j0].kps[m.trainIdx].pt for m in matches0], np.float32)
    E = K.T @ F01 @ K
    pts1_u = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    pts2_u = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)
    _, R01, t01, inl01 = cv2.recoverPose(E, pts1_u.reshape(-1, 1, 2), pts2_u.reshape(-1, 1, 2))
    inl01 = inl01.ravel().astype(bool)
    
    # Camera poses, 3D points and observations
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    points3d: Dict[int, np.ndarray] = {}
    observations: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    obs_by_frame: Dict[int, List[Tuple[np.ndarray, int]]] = defaultdict(list)
    
    def add_observation(pid: int, frame_id: int, m_px: np.ndarray):
        observations.setdefault(pid, []).append((frame_id, m_px))
        obs_by_frame[frame_id].append((m_px, pid))
    
    # Initialize first two camera poses
    poses[i0] = (np.zeros(3), np.zeros(3))
    rvec1, _ = cv2.Rodrigues(R01)
    poses[j0] = (rvec1.ravel(), t01.ravel())
    
    # Triangulate initial 3D points
    x1n = pts1_u[inl01]
    x2n = pts2_u[inl01]
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R01, t01])
    X01 = triangulate_points(P0, P1, x1n, x2n)
    
    # Filter and validate initial triangulated points
    SEED_REPROJ_THRESH = 5.0
    kept = 0
    for j, X in enumerate(X01):
        z0 = (np.eye(3) @ X + np.zeros(3))[2]
        z1 = (R01 @ X + t01.ravel())[2]
        if z0 <= 0 or z1 <= 0:
            continue
        
        pr0 = project_points(X[None, :], np.zeros(3), np.zeros(3), K)[0]
        pr1 = project_points(X[None, :], cv2.Rodrigues(R01)[0].ravel(), t01.ravel(), K)[0]
        if np.linalg.norm(pr0 - pts1[inl01][j]) < SEED_REPROJ_THRESH and \
           np.linalg.norm(pr1 - pts2[inl01][j]) < SEED_REPROJ_THRESH:
            
            pid = len(points3d)
            points3d[pid] = X
            add_observation(pid, i0, pts1[inl01][j])
            add_observation(pid, j0, pts2[inl01][j])
            kept += 1
    
    if verbose:
        print(f"[seed] 3D points kept after reproj filter: {kept}")
    
    # Incremental reconstruction
    for (ia, ib, matches, F_ab) in good_pairs[1:]:
        f_cur, f_nxt = frames[ia], frames[ib]
        pts2d, pts3d_list = [], []
        
        obs_list_ia = obs_by_frame.get(ia, [])
        for m in matches:
            pt_cur = np.array(f_cur.kps[m.queryIdx].pt, dtype=np.float32)
            found = False
            
            for meas, pid in obs_list_ia:
                if np.linalg.norm(meas - pt_cur) < 0.5:
                    pts3d_list.append(points3d[pid])
                    pts2d.append(f_nxt.kps[m.trainIdx].pt)
                    found = True
                    break
        
        if len(pts3d_list) >= 6:
            success, rvec, tvec, inl = cv2.solvePnPRansac(
                np.array(pts3d_list, np.float32), np.array(pts2d, np.float32),
                K, None, iterationsCount=2000, reprojectionError=2.5, confidence=0.999
            )
            
            if not success:
                p_a = np.array([f_cur.kps[m.queryIdx].pt for m in matches], np.float32)
                p_b = np.array([f_nxt.kps[m.trainIdx].pt for m in matches], np.float32)
                Eab = K.T @ F_ab @ K
                _, R, t, _ = cv2.recoverPose(
                    Eab,
                    cv2.undistortPoints(p_a.reshape(-1, 1, 2), K, None),
                    cv2.undistortPoints(p_b.reshape(-1, 1, 2), K, None)
                )
                rvec, _ = cv2.Rodrigues(R)
                tvec = t
        else:
            p_a = np.array([f_cur.kps[m.queryIdx].pt for m in matches], np.float32)
            p_b = np.array([f_nxt.kps[m.trainIdx].pt for m in matches], np.float32)
            Eab = K.T @ F_ab @ K
            _, R, t, _ = cv2.recoverPose(
                Eab,
                cv2.undistortPoints(p_a.reshape(-1, 1, 2), K, None),
                cv2.undistortPoints(p_b.reshape(-1, 1, 2), K, None)
            )
            rvec, _ = cv2.Rodrigues(R)
            tvec = t
        
        # Check parallax
        pts_i = np.array([f_cur.kps[m.queryIdx].pt for m in matches], np.float32)
        pts_j = np.array([f_nxt.kps[m.trainIdx].pt for m in matches], np.float32)
        
        pts_i_u = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts_j_u = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        
        R_est, _ = cv2.Rodrigues(rvec)
        b1 = np.column_stack([pts_i_u, np.ones(len(pts_i_u))])
        b2 = (R_est @ np.column_stack([pts_j_u, np.ones(len(pts_j_u))]).T).T
        b1 /= np.linalg.norm(b1, axis=1, keepdims=True)
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.sum(b1 * b2, axis=1), -1, 1)))
        if np.median(ang) < 3.0:
            continue
        
        # Pose assignment
        poses.setdefault(ia, (np.zeros(3), np.zeros(3)))
        poses[ib] = (rvec.ravel(), tvec.ravel())
        
        # Triangulate new points
        R_i, _ = cv2.Rodrigues(poses[ia][0])
        t_i = poses[ia][1].reshape(3, 1)
        R_j, _ = cv2.Rodrigues(poses[ib][0])
        t_j = poses[ib][1].reshape(3, 1)
        P_i = np.hstack([R_i, t_i])
        P_j = np.hstack([R_j, t_j])
        
        pts_i = np.array([f_cur.kps[m.queryIdx].pt for m in matches], np.float32)
        pts_j = np.array([f_nxt.kps[m.trainIdx].pt for m in matches], np.float32)
        
        pts_i_u = cv2.undistortPoints(pts_i.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        pts_j_u = cv2.undistortPoints(pts_j.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        
        X_ij = triangulate_points(P_i, P_j, pts_i_u, pts_j_u)
        
        # Validate and store newly triangulated points
        for k, Xk in enumerate(X_ij):
            z_i = (R_i @ Xk + t_i.ravel())[2]
            z_j = (R_j @ Xk + t_j.ravel())[2]
            if z_i <= 0 or z_j <= 0:
                continue
            
            pr_i = project_points(Xk[None, :], poses[ia][0], poses[ia][1], K)[0]
            pr_j = project_points(Xk[None, :], poses[ib][0], poses[ib][1], K)[0]
            
            if (np.linalg.norm(pr_i - pts_i[k]) < 2.5) and (np.linalg.norm(pr_j - pts_j[k]) < 2.5):
                matched_pid = None
                pt_i = pts_i[k]
                
                for pid, obs in observations.items():
                    for cam_id, pt in obs:
                        if cam_id == ia and np.linalg.norm(pt - pt_i) < 0.8:
                            matched_pid = pid
                            break
                    if matched_pid is not None:
                        break
                
                if matched_pid is not None:
                    add_observation(matched_pid, ib, pts_j[k])
                else:
                    pid = len(points3d)
                    points3d[pid] = Xk
                    add_observation(pid, ia, pts_i[k])
                    add_observation(pid, ib, pts_j[k])
    
    if verbose:
        print(f"[Reconstruction] poses={len(poses)}, points3d={len(points3d)}")
    
    # Outlier filtering
    all_errs = []
    track_err = {}
    
    for pid, obs in observations.items():
        errs = []
        for (fi, m_px) in obs:
            if fi not in poses:
                continue
            rvec, tvec = poses[fi]
            pr = project_points(points3d[pid][None, :], rvec, tvec, K)[0]
            errs.append(float(np.linalg.norm(pr - m_px)))
        if errs:
            track_err[pid] = errs
            all_errs.extend(errs)
    
    if all_errs:
        q1, q3 = np.percentile(all_errs, [25, 75])
        iqr = max(q3 - q1, 1e-6)
        thr = q3 + 2.0 * iqr
        
        for pid, errs in list(track_err.items()):
            keep_obs = []
            for (fi, m_px), e in zip(observations[pid], errs):
                if e < thr:
                    keep_obs.append((fi, m_px))
            if len(keep_obs) >= 2:
                observations[pid] = keep_obs
            else:
                observations.pop(pid, None)
                points3d.pop(pid, None)
    
    if verbose:
        print(f"[After outlier filtering] poses={len(poses)}, points3d={len(points3d)}")
    
    # === BUNDLE ADJUSTMENT ===
    if SCIPY_OK and len(points3d) > 50:
        # Build clean subset before BA (matching notebook)
        clean_obs = build_clean_subset(observations, poses, points3d, K, per_frame_cap=600, err_thresh=3.0)
        if verbose:
            print(f"[Pre-BA] kept obs after culling: {len(clean_obs)}")
        
        # Intrinsics-only BA
        f_ba, cx_ba, cy_ba = ba_refine_intrinsics_only(
            observations, poses, points3d,
            fx, cx, cy, W, H,
            clean_obs=clean_obs,
            max_obs_total=20000,
            verbose=verbose
        )
        
        if verbose:
            print(f"[After intr-BA] f≈{f_ba:.1f} px, cx={cx_ba:.1f}, cy={cy_ba:.1f}")
        
        # Full BA (intrinsics + poses + points)
        f_refined, cx_ref, cy_ref = ba_refine_f_cx_cy_full(
            observations, poses, points3d,
            f_ba, cx_ba, cy_ba, W, H,
            max_frames_ba=20,
            max_points_ba=1200,
            max_obs_total=15000,
            verbose=verbose
        )
        
        # Safety checks
        f_floor = 0.6 * max(W, H)
        if (not np.isfinite(f_refined)) or (f_refined < f_floor):
            if verbose:
                print(f"[BA] refined f={f_refined:.2f} too small, fallback to >= {f_floor:.2f}")
            f_refined = max(f_ba, f_floor)
        
        cx_ref = float(np.clip(cx_ref, 0.3 * W, 0.7 * W))
        cy_ref = float(np.clip(cy_ref, 0.3 * H, 0.7 * H))
        
        fx = fy = f_refined
        cx, cy = cx_ref, cy_ref
    else:
        if verbose:
            print("[BA] Skipped (SciPy unavailable or insufficient points)")
    
    K_final = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    
    if verbose:
        print(f"[Final intrinsics] f≈{fx:.1f} px, cx={cx:.1f}, cy={cy:.1f}")
    
    result = {
        "success": True,
        "camera_matrix": K_final.tolist(),
        "focal_length": float(fx),
        "principal_point": [float(cx), float(cy)],
        "image_size": [W, H],
        "num_images_used": len(images),
        "num_good_pairs": len(good_pairs),
        "num_poses": len(poses),
        "num_points3d": len(points3d),
        "method": "self_calibration"
    }
    
    return result
