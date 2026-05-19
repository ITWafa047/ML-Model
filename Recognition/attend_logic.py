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


def classify_attendance(
    recorded_time: datetime,
    start_time: datetime,
    min_attend: int,
    max_attend: int
) -> str:
    """
    Classify attendance status based on time.
    
    Args:
        recorded_time: When the student was detected
        start_time: When the session started
        min_attend: Minute threshold for "present" (after start_time)
        max_attend: Minute threshold for "late" (after min_attend)
        
    Returns:
        Status: "present", "late", or "absent"
        
    Logic:
        - recorded_time <= start_time + min_attend: "present"
        - start_time + min_attend < recorded_time <= start_time + max_attend: "late"
        - recorded_time > start_time + max_attend: "absent" (shouldn't reach here)
    """
    diff_minutes = (recorded_time - start_time).total_seconds() / 60
    
    logger.info(f"Attendance classification: {diff_minutes:.1f} minutes after session start")
    logger.info(f"Thresholds: present={min_attend}min, late={max_attend}min")
    
    if diff_minutes <= min_attend:
        status = "present"
    elif diff_minutes <= max_attend:
        status = "late"
    else:
        # This shouldn't happen during session, but mark as absent if it does
        status = "absent"
    
    logger.info(f"Classification result: {status}")
    return status


def process_attendance(
    student_code: Optional[str],
    distance: Optional[float],
    session_id: str,
    student_map: Dict[str, str],
    session_data: Dict
) -> Dict:
    """
    Process attendance logic based on recognition result and time classification.
    
    Args:
        student_code: Recognized student code (None if not recognized)
        distance: Distance from FAISS search (None if not recognized)
        session_id: Current session ID
        student_map: Mapping of student_code to student_name
        session_data: Session metadata including start_time, min_attend, max_attend
        
    Returns:
        Dictionary with attendance result including status (present/late/absent)
    """
    now = datetime.now(ZoneInfo("Africa/Cairo"))
    
    # Case 1: No recognition (distance exceeds threshold or embedding extraction failed)
    if student_code is None:
        return {
            "status": "unknown",
            "message": "Face not recognized",
            "distance": distance if distance is not None else None,
            "recorded_at": now,
            "model_accuracy": None,
            "session_id": session_id
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
            "recorded_at": now,
            "model_accuracy": 1.0 - min(distance / DISTANCE_THRESHOLD, 1.0),
            "session_id": session_id
        }
    
    # Case 3: Check for duplicate attendance (prevent rapid re-entry)
    if not _check_duplicate_attendance(session_id, student_code, now):
        return {
            "status": "duplicate",
            "message": f"Student {student_code} already marked recently",
            "student_code": student_code,
            "student_name": student_map.get(student_code, "Unknown"),
            "recorded_at": now,
            "model_accuracy": None,
            "session_id": session_id
        }
    
    # Case 4: Classify attendance based on time
    student_name = student_map.get(student_code, "Unknown")
    
    # 🔥 NEW: Classify attendance status (present/late/absent)
    status = classify_attendance(
        recorded_time=now,
        start_time=session_data["start_time"],
        min_attend=session_data["min_attend"],
        max_attend=session_data["max_attend"]
    )
    
    model_accuracy = 1.0 - min(distance / DISTANCE_THRESHOLD, 1.0)
    
    logger.info(
        f"✅ Attendance recorded: {student_code} ({student_name}) "
        f"Status: {status} | Accuracy: {model_accuracy:.2%} | Distance: {distance:.4f}"
    )
    
    return {
        "status": status,  # 🔥 present / late / absent
        "message": f"Student marked as {status}",
        "student_code": student_code,
        "student_name": student_name,
        "recorded_at": now,
        "model_accuracy": model_accuracy,
        "session_id": session_id
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
    Get a summary of attendance for the session (from in-memory tracking).
    
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


async def get_attendance_summary_from_db(
    session_id: str,
    expected_students: Dict[str, str]
) -> Dict:
    """
    Get a comprehensive summary of attendance from MongoDB.
    
    🔥 NEW: Retrieves actual status (present/late/absent) from DB
    
    Args:
        session_id: Session ID
        expected_students: Dict of {student_code: student_name}
        
    Returns:
        Attendance summary with categorized students
    """
    from data.crud import get_session_collection
    
    session_collection = get_session_collection(session_id)
    
    # Fetch all attendance records
    records = await session_collection.find(
        {"session_id": session_id},
        {"_id": 0}
    ).to_list(None)
    
    present_students = []
    late_students = []
    absent_students = []
    recorded_codes = set()
    
    # Categorize recorded students
    for record in records:
        code = record.get("student_code")
        status = record.get("status")
        recorded_codes.add(code)
        
        student_info = {
            "student_code": code,
            "student_name": record.get("student_name", "Unknown"),
            "recorded_at": record.get("recorded_at"),
            "model_accuracy": record.get("model_accuracy")
        }
        
        if status == "present":
            present_students.append(student_info)
        elif status == "late":
            late_students.append(student_info)
        elif status == "absent":
            absent_students.append(student_info)
    
    # Find students who weren't recorded at all
    not_recorded = set(expected_students.keys()) - recorded_codes
    for code in not_recorded:
        absent_students.append({
            "student_code": code,
            "student_name": expected_students.get(code, "Unknown"),
            "recorded_at": None,
            "model_accuracy": None
        })
    
    total = len(expected_students)
    present_count = len(present_students)
    late_count = len(late_students)
    absent_count = len(absent_students)
    
    logger.info(
        f"Attendance Summary: {present_count} present, "
        f"{late_count} late, {absent_count} absent"
    )
    
    return {
        "session_id": session_id,
        "present_count": present_count,
        "late_count": late_count,
        "absent_count": absent_count,
        "total_expected": total,
        "present_students": present_students,
        "late_students": late_students,
        "absent_students": absent_students,
        "status": "complete" if (present_count + late_count) > 0 else "pending"
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
