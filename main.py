from fastapi import FastAPI
import uvicorn
from routers.uploadImage import router as upload_router
from routers.start_session import router as session_router
from routers.attendance_ws import router as attendance_router
from core.middleware import AuthMiddleware
from routers.deleteImage import router as delete_router
app = FastAPI(
    title="ML Model API",
    version="1.0.0",
)


app.include_router(upload_router)
app.include_router(session_router)
app.include_router(attendance_router)
app.include_router(delete_router)

app.add_middleware(AuthMiddleware)

@app.get("/")
def home():
    return {"message": "ML Model API Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
