from .blink_detector import BlinkDetector
from .head_pose_checker import HeadPoseChecker


class AntiSpoofManager:

    def __init__(self):
        self.blink = BlinkDetector()
        self.pose_checker = HeadPoseChecker()

    def verify(self, student_id, face_data):
        landmarks = face_data["landmarks"]
        blink_ok = self.blink.check(student_id, landmarks)
        
        yaw = face_data["yaw"]
        status = self.pose_checker.check(student_id, yaw)
        
        if not blink_ok:
            return False, "No blink detected"
        elif status == "verified":
            return True, "Live"
        elif status == "waiting":
            return False, "Turn Head"
        else:
            return False, "Spoof / Timeout"
