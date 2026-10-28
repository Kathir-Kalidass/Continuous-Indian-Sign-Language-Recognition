import argparse
import atexit

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from sign_recognizer import ACTIONS, SignRecognizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the browser-based Indian Sign Language demo."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the web server.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the web server.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a Keras .h5 model. Defaults to isl6.h5 or isl6_patched.h5 if present.",
    )
    parser.add_argument(
        "--threshold",
        default=0.70,
        type=float,
        help="Confidence threshold for adding a word to the sentence.",
    )
    return parser.parse_args()


def create_app(model_path=None, threshold=0.70):
    app = Flask(__name__)
    recognizer = SignRecognizer(model_path=model_path, threshold=threshold)
    atexit.register(recognizer.close)

    @app.get("/")
    def index():
        return render_template("index.html", words=ACTIONS.tolist(), threshold=threshold)

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "words": ACTIONS.tolist()})

    @app.post("/predict")
    def predict():
        frame_file = request.files.get("frame")
        if frame_file is None:
            return jsonify({"error": "Missing frame upload"}), 400

        frame_bytes = frame_file.read()
        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode uploaded frame"}), 400

        result = recognizer.process_frame(frame, draw_landmarks=False)
        probabilities = {
            word: round(float(score), 4)
            for word, score in zip(result["words"], result["probabilities"])
        }
        return jsonify(
            {
                "ready": result["ready"],
                "sentence": result["sentence"],
                "sentence_text": result["sentence_text"] or "Perform a sign",
                "predicted_word": result["predicted_word"] or "Waiting...",
                "confidence": round(float(result["confidence"] or 0.0), 4),
                "probabilities": probabilities,
                "words": result["words"],
            }
        )

    @app.post("/reset")
    def reset():
        recognizer.reset()
        return jsonify({"ok": True})

    return app


def main():
    args = parse_args()
    app = create_app(model_path=args.model, threshold=args.threshold)
    print("Starting browser demo at:")
    print(f"http://{args.host}:{args.port}")
    print("Open the page, click Start Camera, and allow camera access in the browser.")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
