import argparse

import cv2
from sign_recognizer import ACTIONS, SignRecognizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Continuous Indian Sign Language webcam demo."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a Keras .h5 model. Defaults to isl6.h5 or isl6_patched.h5 if present.",
    )
    parser.add_argument(
        "--camera",
        default=0,
        type=int,
        help="OpenCV camera index to open.",
    )
    parser.add_argument(
        "--threshold",
        default=0.70,
        type=float,
        help="Confidence threshold for adding a word to the sentence.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    recognizer = SignRecognizer(model_path=args.model, threshold=args.threshold)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. Try --camera 1 or another index."
        )

    print(f"Loaded model: {recognizer.model_path}")
    print("Recognized words:", ", ".join(ACTIONS.tolist()))
    print("Controls: press 'q' to quit, 'c' to clear the sentence.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result = recognizer.process_frame(frame, draw_landmarks=True)
        cv2.imshow("ISL Live Demo", result["frame"])

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            recognizer.reset()

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()


if __name__ == "__main__":
    main()
