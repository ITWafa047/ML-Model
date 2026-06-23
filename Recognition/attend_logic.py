import logging
from typing import Dict, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DISTANCE_THRESHOLD = 0.6
CONFIDENCE_THRESHOLD = 0.0

attendance_tracking = {}
DUPLICATE_CHECK_WINDOW = 40  # 40 ثواني بدل 30


def classify_attendance(recorded_time, start_time, min_attend, max_attend) -> str:
    diff_minutes = (
        recorded_time - start_time
    ).total_seconds() / 60  # convert to minutes

    if diff_minutes <= min_attend:
        return "present"
    elif diff_minutes > min_attend and diff_minutes <= max_attend:
        return "late"
    else:
        return "absent"


def process_attendance(
    student_code: Optional[str],
    distance: Optional[float],
    session_schedule_id: str,
    student_map: Dict[str, str],
    session_data: Dict,
) -> Dict:

    # ✅ FIX: local time بدون timezone خالص
    now = datetime.now()

    # 🔥 FIX: convert numpy → float
    distance = float(distance) if distance is not None else None

    # ❌ unknown
    if student_code is None:
        return {
            "status": "unknown",
            "message": "Face not recognized",
            "distance": distance,
            "recorded_at": now.isoformat(),
            "confidence_score": None,
            "session_schedule_id": session_schedule_id,
        }

    # ⚠️ duplicate
    if not _check_duplicate_attendance(session_schedule_id, student_code, now):
        return {
            "status": "duplicate",
            "message": f"Student {student_code} already marked recently",
            "student_code": str(student_code),
            "student_name": str(student_map.get(student_code, "Unknown")),
            "recorded_at": now.isoformat(),
            "confidence_score": None,
            "session_schedule_id": session_schedule_id,
        }

    # ✅ success
    student_name = student_map.get(student_code, "Unknown")

    # ✅ FIX: normalize start_time — شيل الـ timezone لو موجود أو حوّله من string
    start_time = session_data["start_time"]
    if isinstance(start_time, str):
        # لو اتخزن كـ string في MongoDB
        start_time = datetime.fromisoformat(start_time).replace(tzinfo=None)
    elif hasattr(start_time, 'tzinfo') and start_time.tzinfo is not None:
        # لو datetime بـ timezone
        start_time = start_time.replace(tzinfo=None)

    status = classify_attendance(
        recorded_time=now,
        start_time=start_time,
        min_attend=session_data["min_attend"],
        max_attend=session_data["max_attend"],
    )

    confidence_score = float(
        (1.0 - min(distance / DISTANCE_THRESHOLD, 1.0)) * 100
    )

    return {
        "status": str(status),
        "message": f"Student marked as {status}",
        "student_code": str(student_code),
        "student_name": str(student_name),
        "recorded_at": now.isoformat(),
        "confidence_score": confidence_score,
        "session_schedule_id": session_schedule_id,
    }


def _check_duplicate_attendance(
    session_schedule_id: str, student_code: str, now: datetime
) -> bool:
    if session_schedule_id not in attendance_tracking:
        attendance_tracking[session_schedule_id] = {}

    session_tracking = attendance_tracking[session_schedule_id]

    if student_code in session_tracking:
        last_time = session_tracking[student_code]
        time_diff = (now - last_time).total_seconds()

        if time_diff < DUPLICATE_CHECK_WINDOW:
            return False

    session_tracking[student_code] = now
    return True


def get_attendance_summary(
    session_schedule_id: str,
    expected_students: Dict[str, str],
    minimum_attendance: int,
    maximum_attendance: int,
) -> Dict:

    if session_schedule_id not in attendance_tracking:
        present_students = []
    else:
        present_students = list(attendance_tracking[session_schedule_id].keys())

    absent_students = [
        code for code in expected_students.keys() if code not in present_students
    ]

    attendance_count = len(present_students)

    return {
        "session_schedule_id": session_schedule_id,
        "present_count": attendance_count,
        "absent_count": len(absent_students),
        "total_expected": len(expected_students),
        "present_students": [
            {
                "student_code": str(code),
                "student_name": str(expected_students.get(code, "Unknown")),
                "time": attendance_tracking[session_schedule_id][code].strftime(
                    "%I:%M:%S %p"
                ),
            }
            for code in present_students
        ],
        "absent_students": [
            {
                "student_code": str(code),
                "student_name": str(expected_students.get(code, "Unknown")),
            }
            for code in absent_students
        ],
    }


async def get_attendance_summary_from_db(
    session_schedule_id: str, expected_students: Dict[str, str]
) -> Dict:
    from data.crud import get_session_collection

    session_collection = get_session_collection(session_schedule_id)

    records = await session_collection.find(
        {"session_schedule_id": session_schedule_id}, {"_id": 0}
    ).to_list(None)

    present_students = []
    late_students = []
    absent_students = []
    recorded_codes = set()

    for record in records:

        code = record.get("student_code")

        if code is None:
            continue
        
        code = str(code)

        status = record.get("status")
        recorded_codes.add(code)

        student_info = {
            "student_code": code,
            "student_name": str(record.get("student_name", "Unknown")),
            "recorded_at": str(record.get("recorded_at")),
            "confidence_score": (
                float(record.get("confidence_score"))
                if record.get("confidence_score")
                else None
            ),
        }

        if status == "present":
            present_students.append(student_info)
        elif status == "late":
            late_students.append(student_info)
        else:
            absent_students.append(student_info)

    not_recorded = set(expected_students.keys()) - recorded_codes

    for code in not_recorded:
        absent_students.append(
            {
                "student_code": str(code),
                "student_name": str(expected_students.get(code, "Unknown")),
                "recorded_at": None,
                "confidence_score": None,
            }
        )

    return {
        "session_schedule_id": session_schedule_id,
        "present_count": len(present_students),
        "late_count": len(late_students),
        "absent_count": len(absent_students),
        "total_expected": len(expected_students),
        "present_students": present_students,
        "late_students": late_students,
        "absent_students": absent_students,
    }


def clear_session_tracking(session_schedule_id: str) -> None:
    if session_schedule_id in attendance_tracking:
        del attendance_tracking[session_schedule_id]


def update_distance_threshold(new_threshold: float) -> None:
    global DISTANCE_THRESHOLD

    if not 0.0 <= new_threshold <= 1.0:
        return

    DISTANCE_THRESHOLD = new_threshold