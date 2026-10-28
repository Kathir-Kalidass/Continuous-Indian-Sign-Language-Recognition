# Continuous-Indian-Sign-Language-Recognition

https://github.com/Swaroop-Srisailam/Continuous-Indian-Sign-Language-Recognition/assets/44131603/98543c84-06f2-47db-a440-208df85337d3

## Introduction

I have developed a pose-based deep learning model that is capable of recognizing six Indian dynamic signs using the Mediapipe framework for pose estimation. By using a sequential model, LSTM, I was able to achieve a high accuracy rate of 96.2%. The model has been tested in real-time and shows promising results for its ability to recognize dynamic signs.

### LSTM

LSTM networks are an extension of recurrent neural networks (RNNs) mainly introduced to handle situations where RNNs fail. Long Short-Term Memory (LSTM) was brought into the picture. It has been so designed that the vanishing gradient problem is almost completely removed, while the training model is left unaltered. LSTMs provide us with a large range of parameters such as learning rates, and input and output biases.

### Summary of LSTM model used

<img width="407" alt="summary" src="https://github.com/MDSALMANSHAMS/3D-Image-Matching-with-LoFTR/assets/68110323/68d6c332-35fd-4bdc-bc93-e9a0a4f942d3">

### MediaPipe Library

MediaPipe is a Framework for building machine learning pipelines for processing time-series data like video, audio, etc. This cross-platform Framework works on Desktop/Server, Android, iOS, and embedded devices like Raspberry Pi and Jetson Nano.

## Requirements

- OpenCV
- Mediapipe
- Matplotlib
- Numpy
- TensorFlow
- sklearn

## Dataset

The dataset consists of 5000+ data points. The format of data is .npy, which means numpy arrays.

## How to RUN

### Modelling

1. Creating Dataset
2. Importing the required Libraries
3. Specify the path for storing images
4. Data Processing
5. Designing Model structure
6. Training Model
7. Evaluation of model on test data
8. Download the `.h5` file from your Google Drive

### Testing

1. Importing the required Libraries
2. Implementation of Mediapipe
3. Calling the Trained Model
4. Implementation of Recoginition Model

## Live Webcam Demo

This repository now includes a standalone webcam demo script: `live_demo.py`.
It also includes a browser-based demo: `web_demo.py`.

### Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### Run

```bash
./.venv/bin/python live_demo.py
```

If your webcam is not on index `0`, try:

```bash
./.venv/bin/python live_demo.py --camera 1
```

### Recognized Words

The trained model in this repository recognizes these 6 signs:

- HI
- THANKS
- FISH
- COW
- PLANE
- CYCLING

The live demo shows:

- the webcam feed
- MediaPipe pose and hand landmarks
- per-word confidence bars
- a rolling sentence made from the recognized signs

Controls:

- press `q` to quit
- press `c` to clear the sentence

## Browser Demo

To use the camera directly in your web browser with a permission prompt:

```bash
./.venv/bin/python web_demo.py
```

Then open:

```text
http://127.0.0.1:8000
```

In the page:

- click `Start Camera`
- allow the browser camera permission
- perform one of the trained signs
- watch the sentence and confidence scores update live

Extra controls:

- `Stop` turns the browser camera off
- `Clear Sentence` resets the running sentence
