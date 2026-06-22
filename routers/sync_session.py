import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Path

from Recognition.attend_logic import get_attendance_summary_from_db
from data.crud import get_final_attendance_collection, get_session_collection
from routers.start_session import get_session_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(tags=["Session Sync"])


@router.post("/sync/session/{session_schedule_id}")
async def sync_session_attendance(session_schedule_id: str = Path(...)):
    """
    Sync and finalize attendance for a session.
    Posts final attendance records to the final_attendance collection.
    """
    try:
        session_manager = get_session_manager()

        try:
            session_data = await session_manager.get_session_data(session_schedule_id)
        except Exception:
            session_collection = get_session_collection(session_schedule_id)
            session_doc = await session_collection.find_one({
                "session_schedule_id": session_schedule_id
            })

            if not session_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_schedule_id} not found"
                )

            student_map = {}
            for student in session_doc.get("students", []):
                student_map[student.get("student_code")] = student.get("student_name")

            session_data = {
                "student_map": student_map,
                "created_at": session_doc.get("created_at"),
                "start_time": session_doc.get("start_time"),
                "end_time": session_doc.get("end_time"),
                "min_attend": session_doc.get("min_attend"),
                "max_attend": session_doc.get("max_attend"),
            }

        session_collection = get_session_collection(session_schedule_id)

        attendance_records = await session_collection.find(
            {"session_schedule_id": session_schedule_id}, {"_id": 0}
        ).to_list(None)

        if not attendance_records:
            raise HTTPException(
                status_code=404,
                detail=f"No attendance records found for session {session_schedule_id}"
            )

        final_collection = get_final_attendance_collection()

        summary = await get_attendance_summary_from_db(
            session_schedule_id=session_schedule_id,
            expected_students=session_data["student_map"]
        )

        now_dt = datetime.now(ZoneInfo("Africa/Cairo"))

        final_attendance_doc = {
            "session_schedule_id": session_schedule_id,
            "synced_at": now_dt.isoformat(),
            "sync_timestamp": now_dt,
            "present_count": len(summary.get("present_students", [])),
            "late_count": len(summary.get("late_students", [])),
            "absent_count": len(summary.get("absent_students", [])),
            "total_expected": summary.get("total_expected", 0),
            "present_students": summary.get("present_students", []),
            "late_students": summary.get("late_students", []),
            "absent_students": summary.get("absent_students", []),
            "session_info": {
                "created_at": session_data.get("created_at"),
                "start_time": session_data.get("start_time"),
                "end_time": session_data.get("end_time"),
                "min_attend": session_data.get("min_attend"),
                "max_attend": session_data.get("max_attend"),
            }
        }

        result = await final_collection.insert_one(final_attendance_doc)

        logger.info(f"Session {session_schedule_id} synced and finalized")

        return {
            "status": "success",
            "message": f"Session {session_schedule_id} attendance synced successfully",
            "session_schedule_id": session_schedule_id,
            "synced_records": len(attendance_records),
            "summary": {
                "present_count": final_attendance_doc["present_count"],
                "late_count": final_attendance_doc["late_count"],
                "absent_count": final_attendance_doc["absent_count"],
                "total_expected": final_attendance_doc["total_expected"],
            },
            "inserted_id": str(result.inserted_id)
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error syncing session {session_schedule_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync session: {str(e)}"
        )


@router.post("/sync/post-attendance/{session_schedule_id}")
async def post_attendance_to_external(session_schedule_id: str = Path(...)):
    """
    Post finalized attendance to external system.
    This endpoint retrieves synced attendance and formats it for external posting.
    """
    try:
        final_collection = get_final_attendance_collection()

        final_attendance = await final_collection.find_one(
            {"session_schedule_id": session_schedule_id},
            sort=[("_id", -1)]
        )

        if not final_attendance:
            raise HTTPException(
                status_code=404,
                detail=f"No final attendance found for session {session_schedule_id}"
            )

        if "_id" in final_attendance:
            del final_attendance["_id"]

        post_payload = {
            "session_schedule_id": final_attendance["session_schedule_id"],
            "synced_at": final_attendance["synced_at"],
            "attendance_data": {
                "summary": {
                    "present": final_attendance["present_count"],
                    "late": final_attendance["late_count"],
                    "absent": final_attendance["absent_count"],
                    "total": final_attendance["total_expected"],
                },
                "present_students": final_attendance.get("present_students", []),
                "late_students": final_attendance.get("late_students", []),
                "absent_students": final_attendance.get("absent_students", []),
            },
            "session_info": final_attendance.get("session_info", {})
        }

        logger.info(f"Attendance posted for session {session_schedule_id}")

        return {
            "status": "success",
            "message": "Attendance posted successfully",
            "payload": post_payload
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Error posting attendance for session {session_schedule_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to post attendance: {str(e)}"
        )


@router.get("/sync/sessions")
async def get_all_synced_sessions():
    """
    Get all synced sessions from final_attendance collection.
    """
    try:
        final_collection = get_final_attendance_collection()

        synced_sessions = await final_collection.find(
            {}, {"_id": 0}
        ).to_list(None)

        if not synced_sessions:
            return {
                "status": "success",
                "message": "No synced sessions found",
                "total_sessions": 0,
                "sessions": []
            }

        return {
            "status": "success",
            "message": "Retrieved all synced sessions",
            "total_sessions": len(synced_sessions),
            "sessions": synced_sessions
        }

    except Exception as e:
        logger.error(f"Error retrieving synced sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )


@router.delete("/sync/session/{session_schedule_id}")
async def delete_session_data(session_schedule_id: str = Path(...)):
    """
    Delete all attendance records for a session after syncing.
    Use with caution - only after confirming successful sync.
    """
    try:
        session_collection = get_session_collection(session_schedule_id)

        session_delete_result = await session_collection.delete_many(
            {"session_schedule_id": session_schedule_id}
        )

        logger.info(
            f"Session {session_schedule_id} data deleted. "
            f"Deleted {session_delete_result.deleted_count} records"
        )

        return {
            "status": "success",
            "message": f"Session {session_schedule_id} data cleaned up",
            "deleted_records": session_delete_result.deleted_count,
            "session_schedule_id": session_schedule_id
        }

    except Exception as e:
        logger.error(f"Error deleting session {session_schedule_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session data: {str(e)}"
        )
