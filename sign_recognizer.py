import atexit
import threading
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras


ACTIONS = np.array(["HI", "THANKS", "FISH", "COW", "PLANE", "CYCLING"])
COLORS = [
    (245, 23, 16),
    (245, 117, 16),
    (16, 245, 117),
    (16, 117, 245),
    (180, 16, 245),
    (245, 16, 180),
]
SEQUENCE_LENGTH = 20
MAX_SENTENCE_WORDS = 5

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def resolve_model_path(explicit_path):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return path

    for candidate in ("isl6.h5", "isl6_patched.h5"):
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError("No model file found. Expected isl6.h5 or isl6_patched.h5.")


def build_sign_model():
    return keras.Sequential(
        [
            keras.layers.Input(shape=(20, 258)),
            keras.layers.LSTM(64, return_sequences=True, activation="relu"),
            keras.layers.LSTM(128, return_sequences=True, activation="relu"),
            keras.layers.LSTM(64, return_sequences=False, activation="relu"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(len(ACTIONS), activation="softmax"),
        ]
    )


def load_sign_model(model_path):
    try:
        return keras.models.load_model(model_path, compile=False)
    except ValueError as exc:
        if "time_major" not in str(exc):
            raise

        model = build_sign_model()
        model.load_weights(model_path)
        return model


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    left_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, left_hand, right_hand])


def prob_viz(probabilities, actions, frame, colors):
    output_frame = frame.copy()
    for index, prob in enumerate(probabilities):
        top = 65 + index * 38
        cv2.rectangle(
            output_frame,
            (10, top),
            (10 + int(prob * 240), top + 24),
            colors[index],
            -1,
        )
        cv2.putText(
            output_frame,
            f"{actions[index]} {prob:.2f}",
            (16, top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output_frame


def draw_banner(frame, sentence, confidence):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 52), (18, 18, 18), -1)
    cv2.putText(
        frame,
        f"Sentence: {' '.join(sentence) if sentence else 'Perform a sign'}",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Words: HI, THANKS, FISH, COW, PLANE, CYCLING",
        (10, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 240, 180),
        1,
        cv2.LINE_AA,
    )
    if confidence is not None:
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (frame.shape[1] - 130, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


class SignRecognizer:
    def __init__(self, model_path=None, threshold=0.70, max_sentence_words=MAX_SENTENCE_WORDS):
        self.model_path = resolve_model_path(model_path)
        self.model = load_sign_model(self.model_path)
        self.threshold = threshold
        self.max_sentence_words = max_sentence_words
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.sentence = []
        self.last_confidence = None
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lock = threading.Lock()
        atexit.register(self.close)

    def reset(self):
        with self._lock:
            self.sequence.clear()
            self.sentence = []
            self.last_confidence = None

    def close(self):
        holistic = getattr(self, "holistic", None)
        if holistic is not None:
            holistic.close()
            self.holistic = None

    def process_frame(self, frame, draw_landmarks=True):
        with self._lock:
            image, results = mediapipe_detection(frame, self.holistic)
            if draw_landmarks:
                draw_styled_landmarks(image, results)

            self.sequence.appendleft(extract_keypoints(results))

            probabilities = np.zeros(len(ACTIONS), dtype=np.float32)
            predicted_word = ""
            ready = len(self.sequence) == SEQUENCE_LENGTH

            if ready:
                probabilities = self.model.predict(
                    np.expand_dims(list(self.sequence), axis=0),
                    verbose=0,
                )[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_word = ACTIONS[predicted_index]
                self.last_confidence = float(probabilities[predicted_index])

                if self.last_confidence >= self.threshold:
                    if not self.sentence or predicted_word != self.sentence[-1]:
                        self.sentence.append(predicted_word)
                        self.sentence = self.sentence[-self.max_sentence_words :]

                if draw_landmarks:
                    image = prob_viz(probabilities, ACTIONS, image, COLORS)

            if draw_landmarks:
                draw_banner(image, self.sentence, self.last_confidence)

            sentence_text = " ".join(self.sentence) if self.sentence else ""
            return {
                "frame": image,
                "ready": ready,
                "sentence": list(self.sentence),
                "sentence_text": sentence_text,
                "predicted_word": predicted_word,
                "confidence": self.last_confidence,
                "probabilities": probabilities.tolist(),
                "words": ACTIONS.tolist(),
            }
