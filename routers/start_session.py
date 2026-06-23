import faiss
import numpy as np
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Tuple
from datetime import datetime,timezone

from schemas.sessionRequest import SessionStartRequest, StudentSession
from data.settings import collection
from data.crud import get_session_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Session Management"])


class SessionManager:

    def __init__(self):
        self.faiss_indices = {}
        self.session_data = {}
        logger.info("SessionManager initialized")

    # =========================
    # LOAD SESSION FROM DATABASE (FALLBACK)
    # =========================
    async def load_session_from_db(self, session_schedule_id: str):
        """
        استعادة بيانات السيشن من MongoDB إذا لم تكن موجودة في الذاكرة
        """
        try:
            session_collection = get_session_collection(session_schedule_id)
            session_doc = await session_collection.find_one(
                {"session_schedule_id": session_schedule_id}
            )

            if not session_doc:
                logger.error(f"Session {session_schedule_id} not found in MongoDB")
                return None

            logger.info(f"Loaded session {session_schedule_id} from MongoDB")

            # استخراج البيانات من MongoDB
            students = [StudentSession(**s) for s in session_doc.get("students", [])]

            # إعادة بناء البيانات
            embeddings_list, index_to_code, student_codes = await self.load_embeddings(
                students
            )
            faiss_index = self.build_faiss_index(embeddings_list)

            # 🔥 FIX: Use student names from the session document first, then fall back to collection
            student_map = {}

            # Create a map of students from the request
            students_by_code = {s.student_code: s for s in students}

            for student_code in student_codes:
                # First try to get from session document (has the actual names)
                if student_code in students_by_code:
                    student_map[student_code] = students_by_code[
                        student_code
                    ].student_name
                    logger.info(
                        f"✅ Using from session: {student_code} -> {students_by_code[student_code].student_name}"
                    )
                else:
                    # Fallback to collection
                    student_doc = await collection.find_one(
                        {"student_code": student_code}
                    )
                    if student_doc:
                        db_name = (
                            student_doc.get("student_name")
                            or student_doc.get("name")
                            or student_doc.get("full_name")
                            or student_code
                        )
                        student_map[student_code] = db_name
                        logger.info(
                            f"✅ Using from collection: {student_code} -> {db_name}"
                        )
                    else:
                        student_map[student_code] = student_code
                        logger.warning(f"⚠️ Student {student_code} not found anywhere")

            # تخزين البيانات في الذاكرة مرة أخرى
            self.faiss_indices[session_schedule_id] = faiss_index
            self.session_data[session_schedule_id] = {
                "index_to_code": index_to_code,
                "student_codes": student_codes,
                "student_map": student_map,
                "min_attend": session_doc.get("min_attend", 0),
                "max_attend": session_doc.get("max_attend", 100),
                "created_at": session_doc.get("created_at"),
                "start_time": session_doc.get("start_time"),
                "end_time": session_doc.get("end_time"),
            }

            logger.info(f"Session {session_schedule_id} restored to memory")
            return self.session_data[session_schedule_id]

        except Exception as e:
            logger.error(f"Error loading session from DB: {str(e)}")
            return None

    # =========================
    # LOAD STUDENTS
    # =========================
    async def load_students(self, students: List[StudentSession]) -> Dict[str, str]:
        student_map = {}

        for student in students:
            existing_student = await collection.find_one(
                {"student_code": student.student_code}
            )

            if not existing_student:
                logger.warning(
                    f"Student {student.student_code} not found in database, using request data"
                )
                student_map[student.student_code] = student.student_name
                continue

            # 🔥 FIX: Try multiple field names for student name
            db_student_name = (
                existing_student.get("student_name")
                or existing_student.get("name")
                or existing_student.get("full_name")
                or student.student_name
                or student.student_code
            )

            student_map[student.student_code] = db_student_name
            logger.info(
                f"✅ Loaded student: {student.student_code} -> {db_student_name} (keys: {list(existing_student.keys())})"
            )

        return student_map

    # =========================
    # LOAD EMBEDDINGS (FIXED)
    # =========================
    async def load_embeddings(
        self, students: List[StudentSession]
    ) -> Tuple[List[np.ndarray], Dict[int, str], List[str]]:

        embeddings_list = []
        index_to_code = {}
        student_codes = []

        for student in students:
            student_doc = await collection.find_one(
                {"student_code": student.student_code}
            )

            if not student_doc or "mean_embedding" not in student_doc:
                continue

            embedding = np.array(student_doc["mean_embedding"], dtype=np.float32)

            # ❌ skip invalid
            if embedding.shape != (512,):
                continue

            if np.linalg.norm(embedding) == 0:
                continue

            # 🔥 FIX 1: normalize EACH embedding
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            embeddings_list.append(embedding)
            index_to_code[len(embeddings_list) - 1] = student.student_code
            student_codes.append(student.student_code)

        if not embeddings_list:
            raise HTTPException(status_code=400, detail="No valid embeddings found")

        return embeddings_list, index_to_code, student_codes

    # =========================
    # BUILD FAISS (FIXED)
    # =========================
    def build_faiss_index(self, embeddings_list: List[np.ndarray]):
        embeddings_array = np.vstack(embeddings_list).astype(np.float32)

        # 🔥 FIX 2: normalize matrix
        faiss.normalize_L2(embeddings_array)

        # 🔥 FIX 3: cosine similarity (better for ArcFace)
        index = faiss.IndexFlatIP(embeddings_array.shape[1])

        index.add(embeddings_array)

        logger.info(f"FAISS built with {index.ntotal} embeddings")

        return index

    # =========================
    # CREATE SESSION
    # =========================
    async def create_session(self, request: SessionStartRequest):

        session_schedule_id = request.session_schedule_id

        try:
            student_map = await self.load_students(request.students)

            embeddings_list, index_to_code, student_codes = await self.load_embeddings(
                request.students
            )

            faiss_index = self.build_faiss_index(embeddings_list)

            now = datetime.now(timezone.utc).replace(tzinfo=None)

            self.faiss_indices[session_schedule_id] = faiss_index
            self.session_data[session_schedule_id] = {
                "index_to_code": index_to_code,
                "student_codes": student_codes,
                "student_map": student_map,
                "min_attend": request.min_attend,
                "max_attend": request.max_attend,
                "created_at": now.isoformat(),
                "start_time": request.start_time.isoformat(),
                "end_time": request.end_time.isoformat(),
            }

            # Mongo save
            session_collection = get_session_collection(session_schedule_id)

            await session_collection.insert_one(
                {
                    "session_schedule_id": session_schedule_id,
                    "students": [s.model_dump() for s in request.students],
                    "student_count": len(student_codes),
                    "min_attend": request.min_attend,
                    "max_attend": request.max_attend,
                    "status": "active",
                    "created_at": now.isoformat(),
                    "start_time": request.start_time.replace(tzinfo=None).isoformat(),
                    "end_time": request.end_time.replace(tzinfo=None).isoformat(),
                }
            )

            return {
                "status": "success",
                "session_schedule_id": session_schedule_id,
                "student_count": len(student_codes),
                "faiss_index_size": faiss_index.ntotal,
                "min_attend": request.min_attend,
                "max_attend": request.max_attend,
                "created_at": now.isoformat(),
            }

        except HTTPException:
            raise

        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================
    # GETTERS WITH FALLBACK TO DB
    # =========================
    async def get_faiss_index(self, session_schedule_id: str):
        """
        احصل على FAISS index، وإن لم يكن موجوداً في الذاكرة، اجلبه من MongoDB
        """
        # أولاً: تحقق من الذاكرة
        if session_schedule_id in self.faiss_indices:
            logger.info(f"FAISS index for {session_schedule_id} found in memory")
            return self.faiss_indices[session_schedule_id]

        # ثانياً: حاول تحميل من MongoDB
        logger.warning(
            f"FAISS index for {session_schedule_id} NOT in memory, loading from DB..."
        )
        await self.load_session_from_db(session_schedule_id)

        # ثالثاً: تحقق مرة أخرى
        if session_schedule_id in self.faiss_indices:
            logger.info(f"FAISS index for {session_schedule_id} loaded from DB")
            return self.faiss_indices[session_schedule_id]

        # لم نستطع إيجاد البيانات
        logger.error(f"FAISS index for {session_schedule_id} not found in memory or DB")
        raise HTTPException(
            status_code=404, detail="Session not found in memory or database"
        )

    async def get_session_data(self, session_schedule_id: str):
        """
        احصل على بيانات السيشن، وإن لم تكن موجودة في الذاكرة، اجلبها من MongoDB
        """
        # أولاً: تحقق من الذاكرة
        if session_schedule_id in self.session_data:
            logger.info(f"Session data for {session_schedule_id} found in memory")
            return self.session_data[session_schedule_id]

        # ثانياً: حاول تحميل من MongoDB
        logger.warning(
            f"Session data for {session_schedule_id} NOT in memory, loading from DB..."
        )
        session_data = await self.load_session_from_db(session_schedule_id)

        if session_data is not None:
            logger.info(f"Session data for {session_schedule_id} loaded from DB")
            return session_data

        # لم نستطع إيجاد البيانات
        logger.error(
            f"Session data for {session_schedule_id} not found in memory or DB"
        )
        raise HTTPException(
            status_code=404, detail="Session not found in memory or database"
        )

    def get_faiss_index_sync(self, session_schedule_id: str):
        """الإصدار السريع (بدون DB fallback) للاستخدام المتزامن"""
        if session_schedule_id not in self.faiss_indices:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.faiss_indices[session_schedule_id]

    def get_session_data_sync(self, session_schedule_id: str):
        """الإصدار السريع (بدون DB fallback) للاستخدام المتزامن"""
        if session_schedule_id not in self.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        return self.session_data[session_schedule_id]


SESSION_MANAGER = SessionManager()


def get_session_manager():
    return SESSION_MANAGER


@router.post("/start-session")
async def start_session(
    request: SessionStartRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    return await session_manager.create_session(request)


@router.get("/session/{session_schedule_id}")
async def get_session_info(
    session_schedule_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    # 🔥 استدعاء الدالة الـ async مع fallback to DB
    data = await session_manager.get_session_data(session_schedule_id)

    return {
        "session_schedule_id": session_schedule_id,
        "student_count": len(data["student_codes"]),
        "min_attend": data["min_attend"],
        "max_attend": data["max_attend"],
        "status": "active",
    }
