import cv2
import base64
import json
import asyncio
import websockets
from datetime import datetime, timedelta

BASE_URL = "ws://geo-significantly-often-cliff.trycloudflare.com"
SESSION_SCHEDULE_ID = "2"

WS_URL = f"{BASE_URL}/ws/attendance?session_schedule_id={SESSION_SCHEDULE_ID}"

# 🔥 Client-side duplicate tracking
last_marked_student = None
last_marked_time = None
DUPLICATE_COOLDOWN = 35  # ثواني

# 🔥 Filter repetitive messages
last_turn_head_time = None
TURN_HEAD_COOLDOWN = 5  # Don't print "Turn head" more than every 5 seconds


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


async def send_webcam():
    try:
        async with websockets.connect(WS_URL, max_size=10 * 1024 * 1024) as ws:

            print("✅ Connected")

            first_msg = await ws.recv()
            print("SESSION:", first_msg)

            # الكاميرا الخارجية
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                print("❌ Cannot open external webcam")
                return

            while True:
                ret, frame = cap.read()

                if not ret:
                    continue

                frame = cv2.resize(frame, (640, 480))

                frame_b64 = encode_frame(frame)

                await ws.send(json.dumps({"frame": frame_b64}))

                await asyncio.sleep(0.3)  # 3 frames per second

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=3)

                    data = json.loads(response)
                    
                    # 🔥 اطبع الرد بشكل كامل
                    if data.get("type") == "attendance_result":
                        status = data.get("data", {}).get("status")
                        
                        # 🔥 Filter out repetitive "Turn head" messages
                        if status == "unknown" and data.get("data", {}).get("message") == "Turn head to verify liveness":
                            global last_turn_head_time
                            now = datetime.now()
                            
                            if last_turn_head_time is None or (now - last_turn_head_time).total_seconds() > TURN_HEAD_COOLDOWN:
                                print("⚠️ LIVENESS CHECK: Please turn your head to verify you are alive")
                                last_turn_head_time = now
                            # Skip printing repetitive turn head messages
                            continue
                        
                        # Print full response for other statuses
                        print(json.dumps(data, indent=2, ensure_ascii=False))
                        
                        # 🔥 Update tracking for successful marks
                        global last_marked_student, last_marked_time
                        
                        student_code = data.get("data", {}).get("student_code")
                        student_name = data.get("data", {}).get("student_name")
                        
                        # إذا كان الحضور ناجح (present أو late)
                        if status in ["present", "late"] and student_code:
                            last_marked_student = student_code
                            last_marked_time = datetime.now()
                            print(f"✅ {student_name} ({student_code}) marked as {status}")
                        
                        # لو جاء رد duplicate، طبع رسالة إنذار بس
                        elif status == "duplicate":
                            print(f"⚠️ DUPLICATE: {data.get('data', {}).get('message')}")
                    else:
                        # Print other message types
                        print(json.dumps(data, indent=2, ensure_ascii=False))

                except asyncio.TimeoutError:
                    pass

                cv2.imshow("External Webcam", frame)

                key = cv2.waitKey(1)

                if key == 27:  # ESC
                    break

            cap.release()
            cv2.destroyAllWindows()
    except Exception as e:
        print("CLIENT ERROR:", e)


if __name__ == "__main__":
    asyncio.run(send_webcam())
