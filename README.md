# 📷 Camera Intrinsic Parameter Estimation

### *Pattern-Based Calibration (Zhang’s Algorithm) and Self-Calibration from Monocular Sequences*

This project implements and compares two complementary approaches for estimating camera intrinsic parameters:

1. **Classical pattern-based calibration** using a planar checkerboard (Zhang’s method).
2. **Self-calibration** from arbitrary monocular image sequences using epipolar geometry and bundle adjustment.

Both pipelines are implemented in Python using OpenCV, NumPy, and supporting SfM utilities.

## ✔ Datasets

* Kaggle stereo chessboard images

    https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures
* ETH3D monocular SLAM sequences "Einstein 1"

    https://www.eth3d.net/slam_datasets


## 📂 Project Structure

```
computer-vision-project
├── [backend/](backend/)                # FastAPI app + calibration algorithms
│   ├── [main.py](backend/main.py)             # API entry point (FastAPI)
│   ├── [zhang_method.py](backend/zhang_method.py)     # Pattern-based calibration pipeline
│   ├── [init_estimation.py](backend/init_estimation.py)  # Homography & corner helpers
│   ├── [refinement.py](backend/refinement.py)       # Bundle-adjustment refinements
│   ├── [self_calibration_core.py](backend/self_calibration_core.py)
│   ├── [unpack_real_parameters.py](backend/unpack_real_parameters.py)
│   └── [requirements.txt](backend/requirements.txt)
├── [frontend/](frontend/)               # React + Tailwind UI (Vite)
│   ├── [src/App.jsx](frontend/src/App.jsx)      # Main workflow
│   └── ...
├── [data/](data/)                    # Sample input imagery + example parameters
│   ├── [imgs/](data/imgs/)              # Chessboard sets
│   └── [out/](data/out/)               # Reference K/dist outputs
├── [notebooks/](notebooks/)               # Exploration + parity with production code
│   ├── [zhang_method.ipynb](notebooks/zhang_method.ipynb)
│   └── [self_calibration.ipynb](notebooks/self_calibration.ipynb)
├── [output/](output/)                 # Undistorted images / logs
└── [README.md](README.md)
```

## 🔧 Installation

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000


## 📊 Outputs

- **Images Used** — number of uploaded frames that passed validation.
- **Quality Metric** — mean reprojection error (Zhang) or number of good feature pairs (self-cal).
- **Pattern/Image Size** — board corner grid for Zhang, or width×height for self-calibration.
- **Camera Matrix (K)** — rendered as a 3×3 matrix plus individual `fx`, `fy`, `cx`, `cy` values.
- **Distortion Coefficients** — `(k1, k2, p1, p2, k3)` with per-term explanations. For self-calibration this section clearly states “Assumed zero”.
- **Download JSON** — button to export the raw API response (K, dist, errors, etc.) for later use.

## 🔍 Evaluation Summary

| Method                  | RMSE (px) | Pros                                  | Cons                                       |
| ----------------------- | ---------- | -------------------------------------- | ------------------------------------------ |
| **Zhang’s calibration** | **0.19**   | Very accurate, stable, subpixel error  | Requires calibration pattern               |
| **Self-calibration**    | **8.06**   | No pattern needed, works on raw video  | Sensitive to parallax, texture, degeneracy |


## 👥 Team

* **[Anna Belyakova](https://github.com/belyakova-anna)** (team lead) — algorithms debugging, frontend, writing reports
* **[Sofia Pushkareva](https://github.com/mcpushka)** — self calibration algorithm implementation
* **[Ruslan Gatiatullin](https://github.com/Stillah)** — Zhang algorithm implementation
