# Camera Calibration Service

Two calibration methods available:
- **Chessboard Calibration** (Zhang's method): Traditional, more accurate
- **Self-Calibration** (SfM): No pattern required, just photos of any scene

## Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

## Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:3000