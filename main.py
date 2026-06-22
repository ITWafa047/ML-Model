import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import uvicorn
from routers.uploadImage import router as upload_router
from routers.start_session import router as session_router
from routers.attendance_ws import router as attendance_router
from routers.sync_session import router as sync_router
from core.middleware import AuthMiddleware
from routers.deleteImage import router as delete_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model API",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(
        "Validation error for %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(upload_router)
app.include_router(session_router)
app.include_router(attendance_router)
app.include_router(sync_router)
app.include_router(delete_router)

app.add_middleware(AuthMiddleware)

@app.get("/")
def home():
    return {"message": "ML Model API Running"}



@app.get("/ws/attendance", tags=["Live Session"])
def get_live_session_code(session_schedule_id: str = "SEC_2026_02"):
    """
    Get Live Session Connection Code Template for Frontend
    
    Provides copy-paste ready code that frontend can use with their own session_schedule_id.
    
    Query Parameters:
    - session_schedule_id: The session ID to include in the code template (default: SEC_2026_02)
    """
    return {
        "session_schedule_id": session_schedule_id,
        "base_url": "wss://your-api-url",  # Frontend should replace with actual URL
        "javascript_code": f"""
// Live Session - Real-time Face Recognition
const session_schedule_id = "{session_schedule_id}";
const BASE_URL = "wss://your-api-url";  // Replace with actual API URL
const WS_URL = `${{BASE_URL}}/ws/attendance?session_schedule_id=${{session_schedule_id}}`;

const ws = new WebSocket(WS_URL);

ws.addEventListener('open', () => {{
    console.log('✅ Live Session Connected');
    console.log(`Session ID: ${{session_schedule_id}}`);
}});

ws.addEventListener('message', (event) => {{
    const response = JSON.parse(event.data);
    
    if (response.type === 'session_info') {{
        console.log(`📊 Session ready - Students: ${{response.data.student_count}}`);
    }} else if (response.type === 'attendance_result') {{
        const data = response.data;
        if (data.status === 'recognized') {{
            console.log(`✅ ${{data.student_name}} - Confidence: ${{(data.confidence * 100).toFixed(2)}}%`);
        }} else if (data.status === 'unknown') {{
            console.log('⚠️ Face not recognized');
        }} else if (data.status === 'anti_spoof_failed') {{
            console.log('🚫 Spoofing detected');
        }}
    }} else if (response.type === 'error') {{
        console.error('❌ Error:', response.data.error);
    }}
}});

ws.addEventListener('error', (error) => {{
    console.error('❌ WebSocket error:', error);
}});

ws.addEventListener('close', () => {{
    console.log('❌ Live Session Disconnected');
}});

// Function to send frame
function sendFrameFromCanvas(canvas) {{
    const imageData = canvas.toDataURL('image/jpeg');
    const base64 = imageData.split(',')[1];
    ws.send(JSON.stringify({{ frame: base64 }}));
}}

// Alternative: Send frame from video element
function sendFrameFromVideo(video) {{
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    sendFrameFromCanvas(canvas);
}}
        """,
        "python_code": f"""
import asyncio
import websockets
import base64
import json
import cv2

session_schedule_id = "{session_schedule_id}"
BASE_URL = "wss://your-api-url"  # Replace with actual API URL
WS_URL = f"{{BASE_URL}}/ws/attendance?session_schedule_id={{session_schedule_id}}"

def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

async def live_session():
    async with websockets.connect(WS_URL) as websocket:
        print("✅ Live Session Connected")
        print(f"Session ID: {{session_schedule_id}}")
        
        cap = cv2.VideoCapture(0)  # 0 = default webcam, 1 = external webcam
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        print("🎥 Webcam started...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame (optional)
            frame = cv2.flip(frame, 1)
            
            # Encode and send frame
            encoded = encode_frame(frame)
            await websocket.send(json.dumps({{"frame": encoded}}))
            
            # Receive response
            response = json.loads(await websocket.recv())
            
            if response['type'] == 'session_info':
                data = response['data']
                print(f"📊 Session ready - Students: {{data['student_count']}}")
            
            elif response['type'] == 'attendance_result':
                data = response['data']
                if data['status'] == 'recognized':
                    print(f"✅ {{data['student_name']}} - Confidence: {{data['confidence']*100:.2f}}%")
                elif data['status'] == 'unknown':
                    print("⚠️ Face not recognized")
                elif data['status'] == 'anti_spoof_failed':
                    print("🚫 Spoofing detected")
            
            elif response['type'] == 'error':
                print(f"❌ Error: {{response['data']['error']}}")
            
            # Display frame
            cv2.imshow("Live Session", frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("❌ Live Session Disconnected")

asyncio.run(live_session())
        """,
        "setup_instructions": {
            "step_1": "Copy the code above (JavaScript or Python)",
            "step_2": "Replace 'your-api-url' with your actual API URL",
            "step_3": "Replace session_schedule_id if needed (default: SEC_2026_02)",
            "step_4": "Run the code and monitor the console/terminal for results",
            "step_5": "Press 'q' to exit (or close the browser/Python window)"
        },
        "notes": {
            "javascript": "Requires HTML canvas or video element and modern browser",
            "python": "Requires: cv2 (opencv-python), websockets, asyncio libraries",
            "frame_format": "All frames must be JPEG encoded in base64 format",
            "frequency": "Recommended 30 fps for best performance",
            "api_url_example": "http://localhost:8000 (local) or wss://your-domain.com (production)"
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
