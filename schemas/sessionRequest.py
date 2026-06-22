from datetime import datetime
from pydantic import BaseModel
from typing import List

class StudentSession(BaseModel):
    student_code: str
    student_name: str


class SessionStartRequest(BaseModel):
    session_schedule_id: str
    students: List[StudentSession]
    min_attend: int
    max_attend: int
    start_time: datetime
    end_time: datetime
