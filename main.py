from fastapi import FastAPI
import uvicorn
from routers.uploadImage import router as upload_router
from core.middleware import AuthMiddleware

app = FastAPI(
    title="Student Face Embedding API",
    version="1.0.0",
)

app.include_router(upload_router)


app.add_middleware(AuthMiddleware)

@app.get("/")
def home():
    return {"message": "FAISS Face Recognition API Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
