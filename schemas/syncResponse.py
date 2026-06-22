from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class StatusEnum(str, Enum):
    present = "present"
    absent = "absent"
    late = "late"

class SyncResponse(BaseModel):
    student_code: str
    session_schedule_id: str
    status: StatusEnum
    recorded_at: datetime
    model_accuracy: Optional[float] = None


class SyncListResponse(BaseModel):
    session_schedule_id: str
    students: List[SyncResponse]
    present_count: int
    late_count: int
    absent_count: int
    total: int
