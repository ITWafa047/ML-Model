from data.settings import final_attendance_db

def get_session_collection(session_id: str):
    return final_attendance_db[f"session_{session_id}"]

def get_final_attendance_collection():
    return final_attendance_db["final_attendance"]