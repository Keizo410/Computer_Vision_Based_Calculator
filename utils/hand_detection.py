import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return results.multi_hand_landmarks, image

    def get_landmark_positions(self, landmarks, image):
        h, w, _ = image.shape
        positions = {
            'index_tip': (int(landmarks[8].x * w), int(landmarks[8].y * h)),
            'middle_tip': (int(landmarks[12].x * w), int(landmarks[12].y * h)),
            'wrist': (int(landmarks[0].x * w), int(landmarks[0].y * h))
        }
        return positions
    