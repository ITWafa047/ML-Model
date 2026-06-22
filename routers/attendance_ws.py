import logging
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException

from Recognition.webcamRecognition import decode_frame, validate_frame
from Recognition.faceEngine import search_face
from Recognition.attend_logic import (
    process_attendance,
    DISTANCE_THRESHOLD
)
from Recognition.anti_spoofing.anti_spoof_manager import AntiSpoofManager
from upload.imageValidator import ImageValidator
from upload.faceProcessor import FaceProcessor
from data.crud import get_session_collection
from routers.start_session import get_session_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter(tags=["Attendance WebSocket"])

validator = ImageValidator()
processor = FaceProcessor()
anti_spoof_manager = AntiSpoofManager()


def estimate_yaw_from_landmarks(landmarks):
    left_eye = landmarks.get("left_eye") if isinstance(landmarks, dict) else None
    right_eye = landmarks.get("right_eye") if isinstance(landmarks, dict) else None
    nose = landmarks.get("nose") if isinstance(landmarks, dict) else None

    if not left_eye or not right_eye or not nose:
        return 0.0

    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    face_width = max(abs(right_eye[0] - left_eye[0]), 1.0)
    dx = nose[0] - eye_center_x
    yaw_radians = np.arctan2(dx, face_width)
    return float(np.degrees(yaw_radians))


# Live Session WebSocket Endpoint
@router.websocket("/ws/attendance")
async def attendance_websocket(websocket: WebSocket, session_schedule_id: str = Query(...)):
    await websocket.accept()
    logger.info(f"WebSocket connected for session: {session_schedule_id}")

    session_manager = get_session_manager()
    
    # 🔥 DEBUG: اطبع جميع السيشنات المتاحة في الذاكرة
    available_sessions = list(session_manager.session_data.keys())
    logger.info(f"Available sessions in memory: {available_sessions}")
    logger.info(f"Available FAISS indices in memory: {list(session_manager.faiss_indices.keys())}")

    try:
        # 🔥 استدعاء الدوال الـ async بـ fallback to DB
        session_data = await session_manager.get_session_data(session_schedule_id)
        faiss_index = await session_manager.get_faiss_index(session_schedule_id)

    except HTTPException as e:
        logger.error(f"Failed to get session data: {e.detail}")
        await websocket.send_json({
            "type": "error",
            "data": {"error": "Session not found", "details": str(e.detail)}
        })
        await websocket.close()
        return

    await websocket.send_json({
        "type": "session_info",
        "data": {
            "session_schedule_id": session_schedule_id,
            "student_count": len(session_data["student_codes"]),
            "status": "ready"
        }
    })

    # 🔥 DEBUG: Log student map
    logger.info(f"Student map for {session_schedule_id}: {session_data['student_map']}")

    while True:
        try:
            message = await websocket.receive_json()

            if "frame" not in message:
                await websocket.send_json({
                    "type": "error",
                    "data": {"error": "Invalid message format"}
                })
                continue

            frame = decode_frame(message["frame"])
            if not validate_frame(frame):
                continue

            faces_info = validator.faces_detection(frame)

            if not faces_info or not faces_info.get("faces"):
                await websocket.send_json({
                    "type": "attendance_result",
                    "data": {
                        "status": "unknown",
                        "message": "No face detected"
                    }
                })
                continue

            faces = faces_info["faces"]
            single_face = validator.single_face_validation(faces)

            validator.face_quality_checks(frame, single_face)

            face_data = {
                "landmarks": single_face.get("landmarks", {}),
                "yaw": estimate_yaw_from_landmarks(single_face.get("landmarks", {}))
            }

            live, anti_message = anti_spoof_manager.verify(session_schedule_id, face_data)
            if not live:
                if anti_message == "Turn Head":
                    await websocket.send_json({
                        "type": "attendance_result",
                        "data": {
                            "status": "unknown",
                            "message": "Turn head to verify liveness"
                        }
                    })
                    continue

                await websocket.send_json({
                    "type": "attendance_result",
                    "data": {
                        "status": "rejected",
                        "message": "Anti-spoofing failed",
                        "details": anti_message
                    }
                })
                continue

            aligned_face = validator.face_alignment(frame, single_face)

            validator.blur_validation(aligned_face)
            validator.brightness_validation(aligned_face)

            # ===============================
            # 🔥 EMBEDDING (FIX IMPORTANT)
            # ===============================
            mean_embedding, _ = processor.generate_embedding(aligned_face)

            # ✔️ لازم float32 مش float
            mean_embedding = mean_embedding.astype(np.float32)

            # ✔️ normalization مهم جدًا مع FAISS
            mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)

            # ===============================
            # 🔥 SEARCH FAISS
            # ===============================
            student_code, distance = search_face(
                query_embedding=mean_embedding,
                faiss_index=faiss_index,
                index_to_code=session_data["index_to_code"],
                k=1,
                threshold=DISTANCE_THRESHOLD
            )

            if distance is not None:
                distance = float(distance)

            # ===============================
            # 🔥 FIX 1: invalid match protection
            # ===============================
            if student_code is None or distance is None:
                await websocket.send_json({
                    "type": "attendance_result",
                    "data": {
                        "status": "unknown",
                        "message": "Face not recognized",
                        "session_schedule_id": session_schedule_id
                    }
                })
                continue

            # ===============================
            # 🔥 FIX 2: threshold check here (not in attend_logic)
            # ===============================
            if distance > DISTANCE_THRESHOLD:
                await websocket.send_json({
                    "type": "attendance_result",
                    "data": {
                        "status": "rejected",
                        "message": f"Low confidence match ({distance:.4f})",
                        "student_code": str(student_code),
                        "model_accuracy": float(1.0 - min(distance / DISTANCE_THRESHOLD, 1.0)),
                        "session_schedule_id": session_schedule_id
                    }
                })
                continue

            # ===============================
            # 🔥 CHECK: Session is active
            # ===============================
            now = datetime.now(ZoneInfo("Africa/Cairo"))

            start_time = session_data["start_time"]
            end_time = session_data["end_time"]

            # لو جايين string من Laravel
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except:
                    start_time = datetime.fromisoformat(start_time)

            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                except:
                    end_time = datetime.fromisoformat(end_time)

            # add timezone
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=ZoneInfo("Africa/Cairo"))
            else:
                start_time = start_time.astimezone(ZoneInfo("Africa/Cairo"))

            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=ZoneInfo("Africa/Cairo"))
            else:
                end_time = end_time.astimezone(ZoneInfo("Africa/Cairo"))

            # 🔥 أهم check
            if now < start_time or now > end_time:
                await websocket.send_json({
                    "type": "attendance_result",
                    "data": {
                        "status": "session_closed",
                        "message": "Session is not active",
                        "session_schedule_id": session_schedule_id
                    }
                })
                continue

            # ===============================
            # 🔥 ATTENDANCE LOGIC
            # ===============================
            attendance_result = process_attendance(
                student_code=student_code,
                distance=distance,
                session_schedule_id=session_schedule_id,
                student_map=session_data["student_map"],
                session_data=session_data
            )

            # ===============================
            # 🔥 SAVE TO DB
            # ===============================
            if attendance_result["status"] not in ["unknown", "rejected", "duplicate","session_closed"]:
                session_collection = get_session_collection(session_schedule_id)

                await session_collection.update_one(
                    {"student_code": attendance_result["student_code"]},
                    {
                        "$set": {
                            "student_code": attendance_result["student_code"],
                            "student_name": attendance_result["student_name"],
                            "status": attendance_result["status"],
                            "recorded_at": attendance_result["recorded_at"],
                            "model_accuracy": float(attendance_result["model_accuracy"])
                            if attendance_result["model_accuracy"] is not None else None,
                            "session_schedule_id": session_schedule_id
                        }
                    },
                    upsert=True
                )

            # ===============================
            # 🔥 SEND RESPONSE
            # ===============================
            await websocket.send_json({
                "type": "attendance_result",
                "data": attendance_result
            })

        except WebSocketDisconnect:
            logger.info(f"Disconnected session {session_schedule_id}")
            break

        except Exception as e:
            logger.error(f"Frame error: {str(e)}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "data": {
                        "error": "Processing error",
                        "details": str(e)
                    }
                })
            except:
                pass
    
    logger.info(f"Session {session_schedule_id} closed")