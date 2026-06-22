from fastapi import APIRouter, HTTPException

from data.settings import collection

router = APIRouter(tags=["Delete Image"])


@router.delete("/students/{student_code}")
async def delete_student(student_code: str):
    student_code = student_code.strip()

    if not student_code:
        raise HTTPException(status_code=400, detail="Student code is required.")

    result = await collection.delete_one({"student_code": student_code})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")

    return {
        "message": "Student deleted successfully",
        "student_code": student_code
    }