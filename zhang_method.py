import cv2
import numpy as np
from init_estimation import get_img_paths, get_world_coordinates, compute_H, find_K, find_E
from refinement import refine_camera_parameters

def calibrate_camera(img_path: str, 
                     pattern_size: tuple[int, int], 
                     square_size: int,
                     show_progress: int = 0,
                     show_images: bool = False,
                     print_results: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    
    Returns K, distortion, rotation vectors, translation vectors.
    """
    n_corners = pattern_size[0] * pattern_size[1]

    # Corners coordinates in the real world
    world_coordinates = get_world_coordinates(pattern_size, square_size)
    H_matrices, img_points = compute_H(img_path, pattern_size, world_coordinates, show_images)
    
    K = find_K(H_matrices)
    E = find_E(H_matrices, K)
        
    K_opt, dist_opt, E_opt, rvecs, tvecs = refine_camera_parameters(img_points, world_coordinates, K, E, verbose=show_progress)

    # (k1, k2, k3, p1, p2) instead of (k1, k2, p1, p2, k3)
    dist_opt = np.array([dist_opt[0], dist_opt[1], dist_opt[3], dist_opt[4], dist_opt[2]])
    
    if print_results:
        print("Refined K:\n", K_opt)
        print("Refined E (for the first image):\n", E_opt[0])
        print("Estimated distortion (k1, k2, k3, p1, p2):", dist_opt)


    mean_error = 0
    img_list = get_img_paths(img_path)
        
    for i in range(len(img_list)):
        imgpoints2, _ = cv2.projectPoints(world_coordinates, rvecs[i], tvecs[i], K_opt, dist_opt)
        error = cv2.norm(img_points[i].reshape((n_corners,1,2)), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    if print_results:
        print("Mean reprojection error:\n", mean_error/len(img_list))
    
    return K_opt, dist_opt, rvecs, tvecs


def main():
    # 11x7 chessboard (counted by the corners of the squares) with 30 mm per square
    K, dist, rvecs, tvecs = calibrate_camera('data/imgs/leftcamera', (11, 7), 30, show_images=True, show_progress=2, print_results=True)

    #TODO: remove small black frames around the image after undistorion
    img_rgb = cv2.imread('data/imgs/leftcamera/Im_L_1.png')
    img_undist = cv2.undistort(img_rgb, K, dist, None)
    cv2.imshow('undistorted', img_undist)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cv2.imwrite('output/0.png', img_undist)

    print('\n\nUndistorted image is in output/0.png')
    

if __name__ == '__main__':
    main()