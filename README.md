# Age and Gender Detection Using Deep Learning

A web-based computer vision application that detects faces in an uploaded image and predicts each detected face's **estimated age** and **gender**. The project combines trained CNN models with OpenCV face detection, a FastAPI backend, and a lightweight HTML/CSS/JavaScript frontend.

**Author:** Nived Tabib  
**GitHub Repository:** https://github.com/NT-0017/age-gender-detection  
**Live API:** https://age-gender-detection-hg6n.onrender.com/health
**API Documentation:** https://age-gender-detection-hg6n.onrender.com/docs  
**Frontend Demo:** https://age-gender-detection-1-ii33.onrender.com

> **Note:** The API and its prediction endpoint have been deployed and tested successfully. Replace the frontend placeholder above with the exact `.onrender.com` URL shown on your Render Static Site dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Live Demo and API](#live-demo-and-api)
- [Technology Stack](#technology-stack)
- [Model and Dataset](#model-and-dataset)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Run Locally](#run-locally)
- [API Reference](#api-reference)
- [Evaluation Results](#evaluation-results)
- [Known Limitations](#known-limitations)
- [Privacy and Responsible Use](#privacy-and-responsible-use)
- [Author](#author)

## Overview

The application accepts an image containing one or more faces, locates faces with an OpenCV DNN face detector, crops and preprocesses each detected face, and runs the trained age and gender models. The API returns JSON containing face bounding boxes, age estimates, gender predictions, and confidence values. The browser frontend uses that response to present the results visually.

The application uses existing trained model files; the deployment does not train models at request time.

## Features

- Upload an image containing one or multiple faces.
- Detect faces using OpenCV's DNN-based face detector.
- Estimate age for each detected face.
- Predict gender as **Male** or **Female**.
- Return bounding-box coordinates and confidence scores.
- View predictions through an interactive browser interface.
- Inspect and test API endpoints using FastAPI's Swagger UI.
- Deployable frontend and backend with public URLs.

## Live Demo and API

| Resource | Link |
|---|---|
| GitHub repository | https://github.com/NT-0017/age-gender-detection |
| Backend API | https://age-gender-detection-hg6n.onrender.com |
| API health check | https://age-gender-detection-hg6n.onrender.com/health |
| Interactive API documentation | https://age-gender-detection-hg6n.onrender.com/docs |
| Frontend demo |https://age-gender-detection-1-ii33.onrender.com |

The backend is hosted on Render. On Render's free web-service plan, the service may spin down after inactivity, so the first request after a period of inactivity can take longer.

## Technology Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Computer Vision:** OpenCV DNN
- **API:** FastAPI
- **ASGI Server:** Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** UTKFace
- **Deployment:** Render
- **Version Control:** Git and GitHub

## Model and Dataset

### Dataset

The models were developed using the **UTKFace** face-image dataset. The project workflow used **23,687 valid images**.

### Input preprocessing

- Face crops are resized to **64 × 64 pixels**.
- The model input uses **3 RGB channels**.
- Pixel values are converted to floating-point values and normalized to the **0–1** range.

### Prediction models

The project uses separate trained CNN models for the two prediction tasks:

- **Gender classification:** a sigmoid output is converted to a Male/Female label using a 0.5 threshold.
- **Age estimation:** a regression output is scaled to an age estimate and constrained to the configured 1–100 range.

### Face detection

The application uses OpenCV DNN with the included face-detector configuration and weights:

- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`

Face detections are filtered using a confidence threshold before age and gender inference.

## System Workflow

1. The user uploads an image through the frontend.
2. The frontend sends the image as multipart form data to `POST /api/detect`.
3. The backend decodes the image and runs the OpenCV DNN face detector.
4. Each accepted face is cropped, resized, converted to RGB, and normalized.
5. The age and gender models infer predictions for each face.
6. The API returns the image dimensions and a list of face results as JSON.
7. The frontend displays the detected faces, bounding boxes, and prediction cards.

## Project Structure

```text
age-gender-detection/
├── age_model.h5
├── gender_model.h5
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── Gender-Age-Detection-Using-DL.ipynb
├── backend/
│   ├── requirements.txt
│   ├── verify_models.py
│   └── app/
│       ├── __init__.py
│       ├── config.py
│       ├── inference.py
│       ├── main.py
│       └── schemas.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── .gitignore
└── README.md
```

## Run Locally

### Prerequisites

- Python 3.12 (the project was tested locally with Python 3.12.10)
- Git
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/NT-0017/age-gender-detection.git
cd age-gender-detection
```

### 2. Create and activate a virtual environment (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run the commands using the virtual-environment interpreter directly instead.

### 3. Install backend dependencies

```powershell
python -m pip install -r backend/requirements.txt
```

### 4. Start the backend

From the repository root:

```powershell
python -m uvicorn backend.app.main:app --reload
```

The API should be available at:

- API: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 5. Start the frontend

Open a **second terminal** in the repository root and run:

```powershell
python -m http.server 3000 --directory frontend
```

Open http://localhost:3000 in your browser.

**Local API setting:** For local development, set `API_DETECT` in `frontend/script.js` to:

```javascript
const API_DETECT = 'http://127.0.0.1:8000/api/detect';
```

For the deployed frontend, that setting should point to the deployed backend endpoint:

```javascript
const API_DETECT = 'https://age-gender-detection-hg6n.onrender.com/api/detect';
```

## API Reference

### `GET /health`

Checks whether the backend and model components have loaded.

**Example response**

```json
{
  "status": "ok",
  "gender_model": true,
  "age_model": true,
  "face_detector": true,
  "details": {
    "gender_model": "loaded",
    "age_model": "loaded",
    "face_detector": "loaded"
  }
}
```

### `POST /api/detect`

Accepts an image upload using `multipart/form-data`.

| Field | Type | Description |
|---|---|---|
| `file` | Image file | Image containing one or more faces |

Supported image formats are determined by the frontend/backend validation; the frontend currently accepts JPEG, PNG, BMP, and WebP images.

**Example response shape**

```json
{
  "image_width": 612,
  "image_height": 612,
  "faces": [
    {
      "box": {
        "x1": 54,
        "y1": 252,
        "x2": 131,
        "y2": 358
      },
      "age": 24,
      "gender": "Male",
      "gender_confidence": 0.9675,
      "face_confidence": 0.9973
    }
  ]
}
```

The values above illustrate the response structure from a test request; predictions will vary with the uploaded image.

### Test with Swagger UI

1. Open https://age-gender-detection-hg6n.onrender.com/docs.
2. Expand `POST /api/detect`.
3. Click **Try it out**.
4. Choose an image file.
5. Click **Execute** and inspect the JSON response.

## Evaluation Results

The recorded validation results from the project workflow were:

| Metric | Result |
|---|---:|
| Gender validation accuracy | 91.28% |
| Age validation MAE | 8.61 years |
| Valid images used in dataset workflow | 23,687 |

**MAE** (Mean Absolute Error) measures the average absolute difference between the predicted age and the ground-truth age. These are the recorded validation results from the model-development workflow, not a guarantee of performance on every real-world image.

## Known Limitations

- Age predictions are estimates and can differ substantially from a person's actual age.
- Performance may vary with lighting, pose, image quality, occlusion, face size, and demographic representation in the training data.
- The gender model produces a binary Male/Female prediction and does not represent the full range of gender identities.
- The models can make incorrect predictions; confidence scores should not be interpreted as proof that a prediction is correct.
- The free hosting service may sleep after inactivity, which can increase the initial response time.
- This application is intended for demonstration and learning, not as the sole basis for consequential decisions.

## Privacy and Responsible Use

Use images only when you have the right to process them and respect the privacy and consent of the people shown. Avoid using the predictions for high-impact decisions such as employment, education, healthcare, policing, or access to essential services. Age and gender predictions are uncertain model outputs and should not be treated as verified personal attributes.

## Author

**Nived Tabib**

- GitHub: https://github.com/NT-0017
- Project repository: https://github.com/NT-0017/age-gender-detection

---

If you find this project useful for learning, feel free to explore the code and open an issue with suggestions or bug reports.
