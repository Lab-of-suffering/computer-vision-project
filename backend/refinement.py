import cv2
import numpy as np
from scipy.optimize import least_squares

def _pack_params(K: np.ndarray, dist: np.ndarray, rvecs: list[np.ndarray], tvecs: list[np.ndarray], optimize_skew: bool):
    """
    Pack intrinsics, distortion and extrinsics into a 1D parameter vector.
    If optimize_skew is False, skew is fixed to 0 and not included.
    """
    if optimize_skew:
        intr = np.array([K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]], dtype=np.float32)  # fx, fy, s, cx, cy
    else:
        intr = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float32)  # fx, fy, cx, cy (s implicitly 0)

    dist = np.asarray(dist, dtype=np.float32).reshape(-1)  # k1,k2,p1,p2,k3

    ex = []
    for rvec, tvec in zip(rvecs, tvecs):
        ex.append(rvec.reshape(3,))
        ex.append(tvec.reshape(3,))
    ex = np.hstack(ex) if ex else np.array([], dtype=np.float32)

    return np.hstack((intr, dist, ex))

def _unpack_params(x: np.ndarray, n_views: int, optimize_skew: bool):
    """
    Unpack parameter vector into K, dist, rvecs, tvecs
    """
    idx = 0
    if optimize_skew:
        fx, fy, s, cx, cy = x[idx:idx+5]; idx += 5
    else:
        fx, fy, cx, cy = x[idx:idx+4]; s = 0.0; idx += 4

    K = np.array([[fx, s, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    dist = x[idx:idx+5]; idx += 5

    rvecs = []
    tvecs = []
    for i in range(n_views):
        rvec = x[idx:idx+3]; idx += 3
        tvec = x[idx:idx+3]; idx += 3
        rvecs.append(rvec)
        tvecs.append(tvec)

    return K, dist, rvecs, tvecs

def _reprojection_residuals(x: np.ndarray,
                            n_views: int,
                            world_coords: np.ndarray,
                            img_points_list: list[np.ndarray],
                            optimize_skew: bool):
    """
    Compute reprojection residuals for least_squares.
    Returns 1D residual vector [u_err0, v_err0, u_err1, v_err1, ...].
    """
    K, dist, rvecs, tvecs = _unpack_params(x, n_views, optimize_skew)
    k1, k2, k3, p1, p2 = dist
    residuals = []

    # world_coords assumed (M,2) with Z=0
    for view_idx in range(n_views):
        img_pts = img_points_list[view_idx]
        rvec = rvecs[view_idx].reshape(3, 1)
        tvec = tvecs[view_idx].reshape(3, )

        R, _ = cv2.Rodrigues(rvec)
        for i, (u_obs, v_obs) in enumerate(img_pts):
            Xw, Yw, _ = world_coords[i]
            Pw = np.array([Xw, Yw, 0.0], dtype=np.float32)
            Pc = R @ Pw + tvec
            Xc, Yc, Zc = Pc
            if abs(Zc) < 1e-12:
                # extremely unlikely, skip to avoid NaNs
                residuals.extend([0.0, 0.0])
                continue
            x = Xc / Zc
            y = Yc / Zc
            r2 = x*x + y*y
            r4 = r2*r2
            r6 = r4*r2
            radial = 1.0 + k1*r2 + k2*r4 + k3*r6
            x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
            y_dist = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y

            u_proj = K[0,0]*x_dist + K[0,1]*y_dist + K[0,2]
            v_proj = K[1,1]*y_dist + K[1,2]

            residuals.append(u_obs - u_proj)
            residuals.append(v_obs - v_proj)

    return np.asarray(residuals, dtype=np.float32)

def refine_camera_parameters(img_points_list: list[np.ndarray],
                      world_coords: np.ndarray,
                      K_init: np.ndarray,
                      extrinsics_init: list[np.ndarray],
                      dist_init: np.ndarray = np.zeros(5, dtype=np.float32),
                      optimize_skew: bool = False,
                      verbose: int = 2,
                      max_nfev: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full nonlinear bundle adjustment optimizing intrinsics fx,fy,cx,cy (and s if optimize_skew=True),
    distortion (k1,k2,p1,p2,k3) and per-view extrinsics (rvec,tvec) using least squares.

    Returns:
    K_opt (3x3), dist_opt (5,), extrinsics_opt (list of 4x4 matrices), rvecs_opt, tvecs_opt
    """
    n_views = len(img_points_list)
    if n_views == 0:
        raise ValueError("No views provided for bundle adjustment")

    rvecs = []
    tvecs = []
    for ext in extrinsics_init:
        R = ext[:3, :3]
        t = ext[:3, 3].reshape(3,)
        rvec, _ = cv2.Rodrigues(R)
        rvecs.append(rvec.reshape(3,))
        tvecs.append(t.reshape(3,))


    x0 = _pack_params(K_init, dist_init, rvecs, tvecs, optimize_skew)

    # run least_squares
    fun = lambda x: _reprojection_residuals(x, n_views, world_coords, img_points_list, optimize_skew)
    res = least_squares(fun, x0, method='lm' if n_views * world_coords.shape[0] * 2 < 2000 else 'trf',
                        verbose=verbose, max_nfev=max_nfev)
    
    K_opt, dist_opt, rvecs_opt, tvecs_opt = _unpack_params(res.x, n_views, optimize_skew)

    # build extrinsics 4x4 list
    extrinsics_opt = []
    for rvec, tvec in zip(rvecs_opt, tvecs_opt):
        R_opt, _ = cv2.Rodrigues(rvec.reshape(3,1))
        E = np.eye(4, dtype=np.float32)
        E[:3,:3] = R_opt
        E[:3,3] = tvec.reshape(3,)
        extrinsics_opt.append(E)

    extrinsics_opt = np.asarray(extrinsics_opt, dtype=np.float32)
    rvecs_opt = np.asarray(rvecs_opt, dtype=np.float32)
    tvecs_opt = np.asarray(tvecs_opt, dtype=np.float32)

    return K_opt, dist_opt, extrinsics_opt, rvecs_opt, tvecs_opt

