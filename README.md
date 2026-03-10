# 🔍 CIFAR-10 Image Classifier — Full-Stack Application

> **AcmeGrade Internship Project** | CNN + FastAPI + React

A production-ready, full-stack image recognition application that classifies images
into 10 categories using a Convolutional Neural Network trained on CIFAR-10.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Setup & Installation](#setup--installation)
6. [Training the Model](#training-the-model)
7. [Running the Backend](#running-the-backend)
8. [Running the Frontend](#running-the-frontend)
9. [API Documentation](#api-documentation)
10. [Model Details](#model-details)
11. [Example Interaction](#example-interaction)
12. [Future Improvements](#future-improvements)

---

## Project Overview

The application accepts any image uploaded by a user, resizes it to 32×32 pixels,
and passes it through a CNN to identify which of the 10 CIFAR-10 categories it
belongs to. Results are returned with a confidence score and a top-5 ranked list.

**10 Supported Classes**

| Class | Emoji | Class | Emoji |
|-----------|-------|-----------|-------|
| airplane | ✈️ | dog | 🐶 |
| automobile | 🚗 | frog | 🐸 |
| bird | 🐦 | horse | 🐴 |
| cat | 🐱 | ship | 🚢 |
| deer | 🦌 | truck | 🚛 |

---

## Architecture Diagram

```
User Browser
    │  (upload image)
    ▼
┌───────────────────┐
│   React Frontend  │  http://localhost:3000
│   (react-dropzone)│
└────────┬──────────┘
         │  POST /predict  (multipart/form-data)
         ▼
┌────────────────────┐
│  FastAPI Backend   │  http://localhost:8000
│  app.py            │
│  • Image decode    │
│  • Resize → 32×32  │
│  • Normalize       │
│  • model.predict() │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  TensorFlow CNN    │
│  cifar10_model.    │
│  keras             │
└────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------------|-------------------------------|
| ML Framework | TensorFlow / Keras |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + react-dropzone |
| HTTP Client | Axios |
| Data Science | NumPy, Scikit-learn, Matplotlib, Seaborn |
| Deployment | Docker (optional) |

---

## Repository Structure

```
cifar10-app/
├── model/
│   ├── data_loader.py       # Dataset loading, preprocessing, augmentation
│   ├── model.py             # CNN architecture definitions
│   ├── train.py             # Training script with callbacks
│   ├── evaluate.py          # Evaluation: metrics, confusion matrix, plots
│   ├── requirements.txt     # Python ML dependencies
│   └── saved_model/         # Created after training
│       ├── cifar10_model.keras
│       ├── best_model.keras
│       ├── norm_stats.json
│       ├── metrics.json
│       └── *.png            # Training curves, confusion matrix, etc.
│
├── backend/
│   ├── app.py               # FastAPI application
│   └── requirements.txt     # Backend Python dependencies
│
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Styles
│   │   ├── index.js         # React entry point
│   │   └── index.css        # Global styles
│   └── package.json
│
└── README.md
```

---

## Setup & Installation

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18 LTS |
| npm | ≥ 9 |
| git | any |

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cifar10-app.git
cd cifar10-app
```

### 2 — Python environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3 — Install ML dependencies

```bash
pip install -r model/requirements.txt
```

### 4 — Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 5 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Training the Model

> ⏱ Expected training time: ~20–40 min on a GPU, ~2–4 hours on CPU (100 epochs).

```bash
cd model

# Train the lightweight CNN (recommended for quick start)
python train.py --arch cnn --epochs 100 --batch 128

# OR train the deeper ResNet-style model (higher accuracy, slower)
python train.py --arch resnet --epochs 150 --batch 128
```

Training artifacts are saved to `model/saved_model/`:
- `cifar10_model.keras` — full saved model for serving
- `best_model.keras` — best checkpoint by validation accuracy
- `norm_stats.json` — per-channel mean & std used during training
- `training_curves.png` — loss/accuracy curves

### Evaluate

```bash
python evaluate.py
```

Outputs:
- Overall accuracy, precision, recall, F1
- Confusion matrix PNG
- Per-class accuracy bar chart
- 25-sample prediction grid
- `metrics.json`

---

## Running the Backend

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

Interactive Swagger docs: `http://localhost:8000/docs`

---

## Running the Frontend

```bash
cd frontend
npm start
```

The app opens at `http://localhost:3000` and proxies `/predict` calls to the
backend automatically (configured in `package.json`).

---

## API Documentation

### `GET /`
Health check.

**Response**
```json
{ "status": "ok", "model_loaded": true, "message": "CIFAR-10 Image Classifier API" }
```

---

### `GET /classes`
List all supported CIFAR-10 class names.

**Response**
```json
{
  "classes": ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"],
  "count": 10
}
```

---

### `POST /predict`
Classify an uploaded image.

**Request** — `multipart/form-data`
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | JPEG / PNG / WebP, max 10 MB |

**Response** `200 OK`
```json
{
  "prediction": "cat",
  "confidence": 0.9412,
  "top_k": [
    { "label": "cat",  "confidence": 0.9412, "class_index": 3 },
    { "label": "dog",  "confidence": 0.0387, "class_index": 5 },
    { "label": "deer", "confidence": 0.0091, "class_index": 4 },
    { "label": "frog", "confidence": 0.0063, "class_index": 6 },
    { "label": "bird", "confidence": 0.0047, "class_index": 2 }
  ],
  "inference_time_ms": 14.32
}
```

**Error Responses**
| Status | Meaning |
|--------|---------|
| 400 | Cannot decode image |
| 413 | File exceeds 10 MB |
| 415 | Unsupported media type |
| 503 | Model not loaded |

---

### `GET /model/info`
Returns model architecture info and test metrics.

**Response**
```json
{
  "name": "cifar10_cnn",
  "input_shape": [32, 32, 3],
  "output_classes": 10,
  "total_params": 3847690,
  "metrics": {
    "accuracy": 0.8831,
    "weighted_precision": 0.8849,
    "weighted_recall": 0.8831,
    "weighted_f1": 0.8831
  }
}
```

---

## Model Details

### Custom CNN (`--arch cnn`)

```
Input (32×32×3)
  └─ Block 1: Conv(64)→BN→ReLU × 2 → MaxPool → Dropout(0.4)
  └─ Block 2: Conv(128)→BN→ReLU × 2 → MaxPool → Dropout(0.4)
  └─ Block 3: Conv(256)→BN→ReLU × 2 → MaxPool → Dropout(0.4)
  └─ GlobalAvgPool
  └─ Dense(512) → BN → ReLU → Dropout(0.4)
  └─ Dense(10, softmax)
```

**Expected performance:** ~85–88 % test accuracy

### ResNet-style (`--arch resnet`)

Pre-activation residual blocks (He et al. 2016), 3 stages (64→128→256 filters),
global average pooling head.

**Expected performance:** ~91–93 % test accuracy

### Training configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam (lr=0.001) |
| LR schedule | ReduceLROnPlateau (×0.5, patience=10) |
| Early stopping | patience=20 |
| Batch size | 128 |
| Loss | Categorical cross-entropy |
| Regularization | L2 (1e-4), Batch Normalization, Dropout |
| Augmentation | Horizontal flip, translation, rotation, zoom |

---

## Example Interaction

1. User opens `http://localhost:3000`
2. Drags a cat photo onto the dropzone
3. Clicks **Classify Image**
4. Frontend POSTs to `http://localhost:8000/predict`
5. Backend preprocesses → CNN infers → returns JSON
6. UI displays:

```
🐱 Cat
94.1% confidence

Top-5 Predictions
🐱 cat   ████████████████████ 94.1%
🐶 dog   ██ 3.9%
🦌 deer  ▌ 0.9%
🐸 frog  ▌ 0.6%
🐦 bird  ▌ 0.5%

⚡ Inference time: 14.3 ms
```

---

## Future Improvements

- **Transfer learning** — fine-tune EfficientNet or MobileNetV3 for >95 % accuracy
- **Model quantization** — TFLite / ONNX export for faster inference
- **Batch prediction** — support multiple images in one API call
- **Docker Compose** — single-command deployment of backend + frontend
- **CI/CD** — GitHub Actions for automated testing and deployment
- **User accounts** — save prediction history per user (PostgreSQL + Auth)
- **Grad-CAM visualization** — overlay heat maps showing which pixels influenced the prediction
- **Dataset expansion** — fine-tune on custom classes beyond CIFAR-10

---

## License

MIT © AcmeGrade Internship Project
