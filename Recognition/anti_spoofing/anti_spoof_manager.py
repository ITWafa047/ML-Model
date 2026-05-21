from .blink_detector import BlinkDetector
from .head_pose_checker import HeadPoseChecker


class AntiSpoofManager:

    def __init__(self):
        self.blink = BlinkDetector()
        self.pose_checker = HeadPoseChecker()

    def _has_68_point_landmarks(self, landmarks):
        return isinstance(landmarks, (list, tuple)) and len(landmarks) >= 48

    def verify(self, student_id, face_data):
        landmarks = face_data.get("landmarks")

        if self._has_68_point_landmarks(landmarks):
            blink_ok = self.blink.check(student_id, landmarks)
        else:
            blink_ok = True

        yaw = face_data.get("yaw", 0.0)
        status = self.pose_checker.check(student_id, yaw)

        if not blink_ok:
            return False, "No blink detected"
        elif status == "verified":
            return True, "Live"
        elif status == "waiting":
            return False, "Turn Head"
        else:
            return False, "Spoof / Timeout"
