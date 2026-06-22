from motor.motor_asyncio import AsyncIOMotorClient


MONGO_URL = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_URL)
db = client["students_embeddings"]
collection = db["embeddings"]

FinalAttendance = AsyncIOMotorClient(MONGO_URL)
final_attendance_db = FinalAttendance["final_attendance"]

