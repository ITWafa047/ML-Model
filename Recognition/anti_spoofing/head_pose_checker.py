import time


class HeadPoseChecker:

    def __init__(self, required_turn=15, timeout=10):
        """
        require_turn: The number of levels required for this to be considered a valid point.\n
        timeout: The expiration date if you are not motivated.
        """

        self.required_turn = required_turn
        self.timeout = timeout
        self.challenges = {}

    def check(self, person_id, current_yaw):
        """
        It returns:

        - "waiting" if still waiting for action

        - "verified" if verified

        - "timeout" if failed
        """
        
        current_time = time.time()

        if person_id not in self.challenges:
            self.challenges[person_id] = {
                "start_yaw": current_yaw,
                "start_time": current_time,
                "verified": False
            }
            return "waiting"
        
        challenge = self.challenges[person_id]

        if challenge["verified"]:
            return "verified"
        
        yaw_diff = abs(current_yaw - challenge["start_yaw"])

        if yaw_diff >= self.required_turn:
            challenge["verified"] = True
            return "verified"
        
        if current_time - challenge["start_time"] > self.timeout:
            del self.challenges[person_id]
            return "timeout"
        
        return "waiting"
    
    def reset(self,person_id):
        if person_id in self.challenges:
            del self.challenges[person_id]
