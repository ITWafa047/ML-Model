import numpy as np


class BlinkDetector:

    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.counter = {}
        self.blinked = {}

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_ear(self, eye):
        A = self.euclidean_distance(eye[1], eye[5])
        B = self.euclidean_distance(eye[2], eye[4])
        C = self.euclidean_distance(eye[0], eye[3])

        ear = (A + B) / (2 * C) if C > 0 else 0
        return ear

    def check(self, student_id, landmarks):

        # landmarks = 68 points
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        if student_id not in self.counter:
            self.counter[student_id] = 0
            self.blinked[student_id] = False

        if ear < self.ear_threshold:
            self.counter[student_id] += 1
        else:
            if self.counter[student_id] >= self.consecutive_frames:
                self.blinked[student_id] = True
                print(f"👁️ Blink detected for student ID: {student_id}")
            self.counter[student_id] = 0

        return self.blinked[student_id]
