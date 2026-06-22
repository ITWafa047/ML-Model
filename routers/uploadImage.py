from functools import lru_cache

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.concurrency import run_in_threadpool

from upload.imageValidator import ImageValidator
from upload.faceProcessor import FaceProcessor
from schemas.uploadResponse import UploadResponse

from data.settings import collection

from datetime import datetime
from zoneinfo import ZoneInfo

router = APIRouter(tags=["Upload Image"])


@lru_cache
def get_validator() -> ImageValidator:
    return ImageValidator()


@lru_cache
def get_processor() -> FaceProcessor:
    return FaceProcessor()


def _process_and_store_embedding(
    student_code: str,
    image_rgb,
    validator: ImageValidator,
    processor: FaceProcessor,
):
    validator.size_validation(image_rgb)

    faces_info = validator.faces_detection(image_rgb)

    single_face = validator.single_face_validation(faces_info["faces"])

    validator.face_quality_checks(image_rgb, single_face)

    aligned_face = validator.face_alignment(image_rgb, single_face)

    validator.blur_validation(aligned_face)

    validator.brightness_validation(aligned_face)

    mean_embeddings, stack_embeddings = processor.generate_embedding(aligned_face)

    mean_embeddings = mean_embeddings.tolist()
    stack_embeddings = [emb.tolist() for emb in stack_embeddings]

    return {
        "student_code": student_code,
        "mean_embedding": mean_embeddings,
        "stack_embeddings": stack_embeddings,
    }


@router.post("/upload-image", response_model=UploadResponse)
async def upload_image(
    student_code: str = Form(...),
    file: UploadFile = File(...),
    validator: ImageValidator = Depends(get_validator),
    processor: FaceProcessor = Depends(get_processor),
):
    student_code = student_code.strip()
    if not student_code or not student_code.strip():
        raise HTTPException(status_code=400, detail="Student code is required.")

    await validator.validate_format(file)

    image_rgb = await validator.load_image(file)

    try:
        saved_student = await run_in_threadpool(
            _process_and_store_embedding,
            student_code,
            image_rgb,
            validator,
            processor,
        )

        now = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S %p")

        await collection.update_one(
            {"student_code": student_code},
            {
                "$set": {
                    "mean_embedding": saved_student["mean_embedding"],
                    "stack_embeddings": saved_student["stack_embeddings"],
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "student_code": student_code,
                    "created_at": now,
                },
            },
            upsert=True,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="The face embedding model is not available on the server.",
        ) from exc

    return UploadResponse(**saved_student)


# update existing student image endpoint
@router.post("/students/{student_code}/image")
async def update_image(
    student_code: str,
    file: UploadFile = File(...),
    validator: ImageValidator = Depends(get_validator),
    processor: FaceProcessor = Depends(get_processor),
):
    student_code = student_code.strip()
    if not student_code:
        raise HTTPException(status_code=400, detail="Student code is required.")

    existing = await collection.find_one({"student_code": student_code})
    if not existing:
        raise HTTPException(status_code=404, detail="Student not found")

    await validator.validate_format(file)

    image_rgb = await validator.load_image(file)

    try:
        saved_student = await run_in_threadpool(
            _process_and_store_embedding,
            student_code,
            image_rgb,
            validator,
            processor,
        )

        now = datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M:%S %p")

        await collection.update_one(
            {"student_code": student_code},
            {
                "$set": {
                    "mean_embedding": saved_student["mean_embedding"],
                    "stack_embeddings": saved_student["stack_embeddings"],
                    "updated_at": now,
                }
            }
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="The face embedding model is not available on the server.",
        ) from exc

    return UploadResponse(**saved_student)


@router.on_event("startup")
async def startup():
    await collection.create_index("student_code", unique=True)
