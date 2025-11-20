import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

def get_img_paths(images_path: str) -> list[str]:
    """Helper function to get image path from a folder."""
    patterns = ['*.jpg', '*.jpeg', '*.png']
    images_list = []
    for pattern in patterns:
        images_list.extend(glob.glob(os.path.join(images_path, pattern)))
    return images_list

def get_world_coordinates(pattern_size: tuple[int, int], square_size: int) -> np.ndarray:
    """Function to get chessboard corners' coordinates in the real world."""
    n_corners = pattern_size[0] * pattern_size[1]
    world_X, world_Y = np.meshgrid(range(pattern_size[0]), range(pattern_size[1]))
    world_X = world_X.reshape(n_corners, 1)
    world_Y = world_Y.reshape(n_corners, 1)

    # Corners coordinates in the real world
    return np.array(np.hstack((world_X, world_Y, np.zeros((world_X.shape))), dtype=np.float32)) * square_size


def get_v(h: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Helper function to get v[i][j]. 
    
    h, i, j are all zero-indexed
    """
    v_ij = np.array([
        h[0, i] * h[0, j],
        h[0, i] * h[1, j] + h[1, i] * h[0, j],
        h[2, i] * h[0, j] + h[0, i] * h[2, j],
        h[1, i] * h[1, j],
        
        h[2, i] * h[1, j] + h[1, i] * h[2, j],
        h[2, i] * h[2, j]
    ])
    return v_ij

def get_B(b: np.ndarray) -> np.ndarray:
    """Helper function to get symmetric B matrix from 6 b's."""
    b_11 = b[0]
    b_12 = b[1]
    b_13 = b[2]
    b_22 = b[3]
    b_23 = b[4]
    b_33 = b[5]

    return np.array([
        [b_11, b_12, b_13],
        [b_12, b_22, b_23],
        [b_13, b_23, b_33]
    ], dtype=np.float32)

def compute_H(img_path: str, 
              pattern_size: tuple[int, int], 
              world_coordinates: np.ndarray, 
              show_images: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Function to compute H matrix for every image. 
    
    Returns list of H matrices and list of image points.
    """
    img_list = get_img_paths(img_path)
    
    term_criterion = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    H_matrices, img_points = [], []

    for image_name in img_list:
        img_rgb = cv2.imread(image_name)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        ok, corners = cv2.findChessboardCorners(img_grey, pattern_size, None)
        
        if not ok: continue
        
        refined_corners = cv2.cornerSubPix(img_grey, corners, pattern_size, (-1, -1), term_criterion)
        refined_corners = refined_corners.reshape(-1, 2)
        img_points.append(refined_corners)

        H, _ = cv2.findHomography(world_coordinates, refined_corners, cv2.RANSAC, 5.0)
        if H is None: continue

        H = H.astype(np.float32)
        if abs(H[2, 2]) > 0:
            H = H / H[2, 2]
        H_matrices.append(H)
        
        # Show picture of chessboard corners
        if show_images:
            cv2.drawChessboardCorners(img_rgb, pattern_size, refined_corners, True)
            cv2.imshow('img', img_rgb)
            cv2.waitKey(400)

    cv2.destroyAllWindows()

    return H_matrices, img_points

def find_K(H_matrices: list[np.ndarray]) -> np.ndarray:
    """Function to find matrix K from H matrices."""
    V = []

    for h in H_matrices:
        v_12 = get_v(h, 0, 1)
        v_11 = get_v(h, 0, 0)
        v_22 = get_v(h, 1, 1)

        V.append(v_12.T)
        V.append(v_11.T - v_22.T)
    
    V = np.asarray(V, dtype=np.float32)
    U_, sigma, V_ = svd(V)
    b = V_[-1, :] # vector corresponding the the lowest eigenvalue
    
    B = get_B(b)
    # B = 0.5 * (B + B.T)

    # Ensure overall sign so diagonal is positive (b is up to scale)
    if B[0, 0] < 0: B = -B

    # Try Cholesky on B (we want R upper s.t. R.T @ R = B -> K_inv = R -> K = inv(R))
    L = np.linalg.cholesky(B)  # L @ L.T = B
    K = np.linalg.inv(L.T)
    K = K / K[2, 2]
    if K[0, 0] < 0: K = -K

    return K


def find_E(H_matrices: list[np.ndarray], K: np.ndarray) -> list[np.ndarray]:
    """Function to find extrinsic matrix from H matrices and K matrix."""
    K_inv = np.linalg.inv(K)

    rotations = []
    translations = []
    extrinsics = []

    for H in H_matrices:
        H = H.astype(np.float32)
        # ensure homography normalized
        if abs(H[2, 2]) > 0:
            H = H / H[2, 2]

        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        kinv_h1 = K_inv @ h1
        kinv_h2 = K_inv @ h2
        kinv_h3 = K_inv @ h3

        # scale: average norm of first two columns for robustness
        norm1 = np.linalg.norm(kinv_h1)
        norm2 = np.linalg.norm(kinv_h2)
        if norm1 == 0 or norm2 == 0:
            raise ValueError("Degenerate homography: zero column after applying K^{-1}")

        lam = 1.0 / ((norm1 + norm2) / 2.0)

        r1 = lam * kinv_h1
        r2 = lam * kinv_h2
        r3 = np.cross(r1, r2)
        t = lam * kinv_h3

        # orthonormalize R using SVD to correct noise: R = U @ Vt
        R_approx = np.column_stack((r1, r2, r3))
        U, _, Vt = svd(R_approx)
        R = U @ Vt

        # enforce right-handed frame
        if np.linalg.det(R) < 0:
            R = -1*R
            t = -t

        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = R
        ext[:3, 3] = t
        rotations.append(R)
        translations.append(t)
        extrinsics.append(ext)

    return extrinsics

