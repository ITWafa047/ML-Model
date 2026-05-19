import logging
from typing import Dict, Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Thresholds for attendance processing
DISTANCE_THRESHOLD = 0.6  # Maximum L2 distance to consider a match
CONFIDENCE_THRESHOLD = 0.0  # Minimum confidence (0-1)

# Tracking to prevent duplicate attendance records
attendance_tracking = {}  # session_id -> {student_code -> last_timestamp}
DUPLICATE_CHECK_WINDOW = 5  # seconds


def process_attendance(
    student_code: Optional[str],
    distance: Optional[float],
    session_id: str,
    student_map: Dict[str, str]
) -> Dict:
    """
    Process attendance logic based on recognition result.
    
    Args:
        student_code: Recognized student code (None if not recognized)
        distance: Distance from FAISS search (None if not recognized)
        session_id: Current session ID
        student_map: Mapping of student_code to student_name
        
    Returns:
        Dictionary with attendance result
    """
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    
    # Case 1: No recognition (distance exceeds threshold or embedding extraction failed)
    if student_code is None:
        return {
            "status": "unknown",
            "message": "Face not recognized",
            "distance": distance if distance is not None else None,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S %p")
        }
    
    # Case 2: Distance threshold check (security)
    if distance is not None and distance > DISTANCE_THRESHOLD:
        logger.warning(
            f"Recognition rejected: distance {distance:.4f} > "
            f"threshold {DISTANCE_THRESHOLD}"
        )
        return {
            "status": "rejected",
            "message": f"Distance {distance:.4f} exceeds threshold",
            "student_code": student_code,
            "distance": distance,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S %p")
        }
    
    # Case 3: Check for duplicate attendance (prevent rapid re-entry)
    if not _check_duplicate_attendance(session_id, student_code, now):
        return {
            "status": "duplicate",
            "message": f"Student {student_code} already marked recently",
            "student_code": student_code,
            "student_name": student_map.get(student_code, "Unknown"),
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S %p")
        }
    
    # Case 4: Valid attendance - record it
    student_name = student_map.get(student_code, "Unknown")
    
    logger.info(
        f"✅ Attendance recorded: {student_code} ({student_name}) "
        f"at {now.strftime('%H:%M:%S')} (distance: {distance:.4f})"
    )
    
    return {
        "status": "success",
        "message": "Attendance recorded",
        "student_code": student_code,
        "student_name": student_name,
        "distance": distance,
        "confidence": 1.0 - min(distance / DISTANCE_THRESHOLD, 1.0),  # Confidence score
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S %p")
    }


def _check_duplicate_attendance(
    session_id: str,
    student_code: str,
    now: datetime
) -> bool:
    """
    Check if this attendance is a duplicate within the time window.
    
    Args:
        session_id: Session ID
        student_code: Student code
        now: Current datetime
        
    Returns:
        True if it's not a duplicate, False if it is
    """
    if session_id not in attendance_tracking:
        attendance_tracking[session_id] = {}
    
    session_tracking = attendance_tracking[session_id]
    
    if student_code in session_tracking:
        last_time = session_tracking[student_code]
        time_diff = (now - last_time).total_seconds()
        
        if time_diff < DUPLICATE_CHECK_WINDOW:
            logger.info(
                f"Duplicate check: {student_code} marked {time_diff:.1f}s ago"
            )
            return False
    
    # Update last attendance time
    session_tracking[student_code] = now
    return True


def get_attendance_summary(
    session_id: str,
    expected_students: Dict[str, str],
    minimum_attendance: int,
    maximum_attendance: int
) -> Dict:
    """
    Get a summary of attendance for the session.
    
    Args:
        session_id: Session ID
        expected_students: Dict of {student_code: student_name}
        minimum_attendance: Minimum required attendance count
        maximum_attendance: Maximum allowed attendance count
        
    Returns:
        Attendance summary dictionary
    """
    if session_id not in attendance_tracking:
        present_students = []
    else:
        present_students = list(attendance_tracking[session_id].keys())
    
    absent_students = [
        code for code in expected_students.keys()
        if code not in present_students
    ]
    
    attendance_count = len(present_students)
    is_minimum_met = attendance_count >= minimum_attendance
    is_maximum_exceeded = attendance_count > maximum_attendance
    
    return {
        "session_id": session_id,
        "present_count": attendance_count,
        "absent_count": len(absent_students),
        "total_expected": len(expected_students),
        "present_students": [
            {
                "student_code": code,
                "student_name": expected_students.get(code, "Unknown"),
                "time": attendance_tracking[session_id][code].strftime("%H:%M:%S %p")
            }
            for code in present_students
        ],
        "absent_students": [
            {
                "student_code": code,
                "student_name": expected_students.get(code, "Unknown")
            }
            for code in absent_students
        ],
        "minimum_met": is_minimum_met,
        "maximum_exceeded": is_maximum_exceeded,
        "status": (
            "complete" if is_minimum_met and not is_maximum_exceeded
            else "pending" if not is_minimum_met
            else "exceeded"
        )
    }


def clear_session_tracking(session_id: str) -> None:
    """Clear attendance tracking for a session."""
    if session_id in attendance_tracking:
        del attendance_tracking[session_id]
        logger.info(f"Cleared tracking for session {session_id}")


def update_distance_threshold(new_threshold: float) -> None:
    """
    Update the distance threshold for recognition.
    
    Args:
        new_threshold: New threshold value (0.0 - 1.0)
    """
    global DISTANCE_THRESHOLD
    
    if not 0.0 <= new_threshold <= 1.0:
        logger.error(f"Invalid threshold {new_threshold}, must be 0.0-1.0")
        return
    
    DISTANCE_THRESHOLD = new_threshold
    logger.info(f"Distance threshold updated to {new_threshold}")
