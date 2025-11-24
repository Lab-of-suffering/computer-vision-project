from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import tempfile
from typing import List
import base64

try:
    from .zhang_method import calibrate_camera
    from .init_estimation import get_img_paths, get_world_coordinates, compute_H
    from .self_calibration_core import self_calibrate
except ImportError:
    # Allow running as `python backend/main.py`
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from zhang_method import calibrate_camera
    from init_estimation import get_img_paths, get_world_coordinates, compute_H
    from self_calibration_core import self_calibrate

app = FastAPI(title="Camera Calibration API")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default chessboard parameters
DEFAULT_PATTERN_SIZE = (11, 7)  # corners in the chessboard
DEFAULT_SQUARE_SIZE = 30  # mm

@app.get("/")
async def root():
    return {"message": "Camera Calibration API", "status": "running"}

@app.post("/calibrate")
async def calibrate(
    files: List[UploadFile] = File(...),
    pattern_width: int = 11,
    pattern_height: int = 7,
    square_size: int = 30
):
    """
    Calibrate camera using uploaded chessboard images
    
    Parameters:
    - files: List of image files (minimum 5, recommended 15+)
    - pattern_width: Number of inner corners per chessboard row
    - pattern_height: Number of inner corners per chessboard column
    - square_size: Size of a square in your defined unit (e.g., mm)
    """
    
    if len(files) < 5:
        raise HTTPException(
            status_code=400, 
            detail=f"At least 5 images required, got {len(files)}"
        )
    
    # Create temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files
        for idx, file in enumerate(files):
            file_path = os.path.join(temp_dir, f"img_{idx:03d}.png")
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        
        # Run calibration
        pattern_size = (pattern_width, pattern_height)
        K, dist, rvecs, tvecs = calibrate_camera(
            temp_dir, 
            pattern_size, 
            square_size,
            show_progress=0,
            show_images=False,
            print_results=False
        )
        
        # Calculate mean reprojection error
        world_coordinates = get_world_coordinates(pattern_size, square_size)
        img_list = get_img_paths(temp_dir)
        H_matrices, img_points = compute_H(temp_dir, pattern_size, world_coordinates, False)
        
        n_corners = pattern_size[0] * pattern_size[1]
        mean_error = 0
        for i in range(len(img_list)):
            imgpoints2, _ = cv2.projectPoints(world_coordinates, rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_points[i].reshape((n_corners,1,2)), imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        mean_error = mean_error / len(img_list)
        
        # Prepare response
        result = {
            "success": True,
            "camera_matrix": K.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_reprojection_error": float(mean_error),
            "num_images_used": len(files),
            "pattern_size": pattern_size,
            "square_size": square_size
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        import traceback
        error_detail = f"Calibration failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Print to console for debugging
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/self-calibrate")
async def self_calibrate_endpoint(
    files: List[UploadFile] = File(...),
    stride: int = 4,
    max_frames: int = 150
):
    """
    Self-calibration without calibration pattern.
    Estimates camera intrinsics from image sequence alone.
    
    Parameters:
    - files: List of image files (minimum 10, recommended 15-100, maximum 500)
    - stride: Frame sampling stride (default: 1 = use all frames)
    - max_frames: Maximum frames to process (default: 500)
    """
    
    if len(files) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Self-calibration needs at least 10 images, got {len(files)}"
        )
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Load all images
        images = []
        for idx, file in enumerate(files):
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            images.append(img)
        
        if len(images) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Only {len(images)} valid images loaded, need at least 10"
            )
        
        # Run self-calibration
        result = self_calibrate(
            images=images,
            target_width=960,
            feature_type="sift",
            stride=stride,
            max_frames=max_frames,
            verbose=True
        )
        
        return JSONResponse(content=result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"Self-calibration failed: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Self-calibration failed: {str(e)}")
    
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/undistort")
async def undistort_image(
    file: UploadFile = File(...),
    camera_matrix: str = None,
    distortion_coeffs: str = None
):
    """
    Undistort an image using calibration parameters
    """
    try:
        # Parse camera matrix and distortion coefficients
        K = np.array(eval(camera_matrix))
        dist = np.array(eval(distortion_coeffs))
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Undistort
        undistorted = cv2.undistort(img, K, dist, None)
        
        # Encode back to image
        _, buffer = cv2.imencode('.png', undistorted)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return JSONResponse(content={
            "success": True,
            "undistorted_image": f"data:image/png;base64,{img_base64}"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Undistortion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

