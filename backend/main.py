from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
import sys
import json
import cv2
import mediapipe as mp
import threading
import asyncio
import math
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from the root .env file
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path)

# Mount internal paths tightly
sys.path.append(os.path.dirname(__file__))
from models.plume_physics import gaussian_concentration

# --- Gesture Tracker Logic ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Global State for gestures
gesture_state = {
    "detected": False,
    "isFist": False,
    "targetX": 0,
    "targetY": 0,
    "dist": 0,
    "wind": {
        "speed": 5.0, 
        "direction": 210,
        "temp": 28.0,
        "rain": 0.0,
        "hum": 80.0
    }
}

WIND_CACHE_FILE = os.path.join(os.path.dirname(__file__), "wind_cache.json")

def load_wind_cache():
    if os.path.exists(WIND_CACHE_FILE):
        try:
            with open(WIND_CACHE_FILE, "r") as f:
                data = json.load(f)
                gesture_state["wind"] = data
                print(f"[WindCache] Loaded saved telemetry: {data}")
        except Exception as e:
            print(f"[WindCache] Load failed: {e}")
    else:
        # Initialize with defaults if missing
        print("[WindCache] Initializing new cache file with defaults.")
        save_wind_cache(gesture_state["wind"])

def save_wind_cache(wind_data):
    try:
        with open(WIND_CACHE_FILE, "w") as f:
            json.dump(wind_data, f)
    except Exception as e:
        print(f"[WindCache] Save failed: {e}")

async def fetch_wind_loop():
    """Background task to fetch NEA wind data every minute."""
    import urllib.request
    
    urls = {
        "speed": os.getenv("NEA_WIND_SPEED_URL"),
        "direction": os.getenv("NEA_WIND_DIRECTION_URL"),
        "temp": os.getenv("NEA_AIR_TEMP_URL"),
        "rain": os.getenv("NEA_RAINFALL_URL"),
        "hum": os.getenv("NEA_HUMIDITY_URL")
    }
    
    STATION_CONFIG = {
        'S50': 3,   # Clementi Road (Closest)
        'S60': 2,   # Sentosa
        'S117': 1   # Banyan Road
    }
    
    while True:
        try:
            new_wind = gesture_state["wind"].copy()
            updated = False
            
            headers = {"User-Agent": "Mozilla/5.0"}
            
            for key, url in urls.items():
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        data = json.loads(resp.read().decode())
                        readings = data.get("data", {}).get("readings", [])
                        if not readings: continue
                        
                        stations = readings[0].get("data", [])
                        if not stations: continue

                        if key == "direction":
                            # Vector Averaging for Direction
                            sum_sin, sum_cos, w_sum = 0, 0, 0
                            for s in stations:
                                sid = s["stationId"]
                                if sid in STATION_CONFIG:
                                    w = STATION_CONFIG[sid]
                                    rad = math.radians(float(s["value"]))
                                    sum_sin += math.sin(rad) * w
                                    sum_cos += math.cos(rad) * w
                                    w_sum += w
                            
                            if w_sum == 0: # Fallback
                                val = stations[0]["value"]
                                new_wind[key] = float(val)
                            else:
                                avg_rad = math.atan2(sum_sin, sum_cos)
                                avg_deg = math.degrees(avg_rad) % 360
                                if avg_deg < 0: avg_deg += 360
                                new_wind[key] = float(avg_deg)
                        else:
                            # Weighted Averaging for scalars
                            v_sum, w_sum = 0, 0
                            for s in stations:
                                sid = s["stationId"]
                                if sid in STATION_CONFIG:
                                    w = STATION_CONFIG[sid]
                                    v_sum += float(s["value"]) * w
                                    w_sum += w
                            
                            if w_sum == 0: # Fallback
                                val = stations[0]["value"]
                                new_wind[key] = float(val)
                            else:
                                new_wind[key] = float(v_sum / w_sum)
                        
                        updated = True
                except Exception as api_err:
                    print(f"[WindSync] API Error for {key}: {api_err}")
                    continue
            
            if updated:
                gesture_state["wind"] = new_wind
                save_wind_cache(new_wind)
                print(f"[WindSync] Weighted sync: {new_wind.get('speed',0):.1f}km/h {new_wind.get('direction',0):.0f}°")
                # print(f"[WindSync] Updated: {new_wind}")
                
        except Exception as e:
            print(f"[WindSync] Global Error: {e}")
            
        await asyncio.sleep(30) # 30 second refresh

def process_gestures():
    """Background thread for OpenCV and MediaPipe processing."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[GestureTracker] Camera access denied or unavailable.")
        return
        
    print("[GestureTracker] Webcam streaming active.")
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
            
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                
                # Fist Detection (v2)
                isFist = (
                    hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and
                    hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
                    hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
                    hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
                )
                
                dxDist = index.x - thumb.x
                dyDist = index.y - thumb.y
                raw_dist = math.sqrt(dxDist**2 + dyDist**2)
                
                # EMA Smoothing
                ALPHA = 0.3
                if not gesture_state.get("detected"):
                    smooth_x = index.x * 1000.0
                    smooth_y = index.y * 1000.0
                    smooth_dist = raw_dist
                else:
                    smooth_x = gesture_state["targetX"] + ALPHA * (index.x * 1000.0 - gesture_state["targetX"])
                    smooth_y = gesture_state["targetY"] + ALPHA * (index.y * 1000.0 - gesture_state["targetY"])
                    smooth_dist = gesture_state["dist"] + ALPHA * (raw_dist - gesture_state["dist"])

                gesture_state.update({
                    "detected": True,
                    "isFist": isFist,
                    "targetX": smooth_x,
                    "targetY": smooth_y,
                    "dist": smooth_dist
                })
        else:
            gesture_state["detected"] = False
            
        cv2.waitKey(33) # ~30 FPS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load cache and start background tasks
    load_wind_cache()
    tracker_thread = threading.Thread(target=process_gestures, daemon=True)
    tracker_thread.start()
    
    # Start wind fetcher as an asyncio task
    asyncio.create_task(fetch_wind_loop())
    
    yield
    # Shutdown logic (daemon threads clean up on exit)

app = FastAPI(title="CrowdShield Unified Backend", lifespan=lifespan)

# Allow React dev server to connect over WebSocket and HTTP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    mode: str = 'classic' # 'classic' (Mathematical Generator) vs 'ai' (PINN)
    x: float
    y: float
    u: float
    Q: float = 447000.0

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'plume_surrogate.onnx')
ort_session = None

if os.path.exists(MODEL_PATH):
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("PINN ONNX Surrogate is actively hosted running on port 8000.")
else:
    print("WARNING: ONNX tensor payload not found. Simulation API will securely fall back to the Teacher Ground-Truth calculations.")


def run_inference(mode: str, x: float, y: float, u: float, Q: float) -> dict:
    """Shared inference logic used by both HTTP and WebSocket endpoints."""
    if mode == 'ai' and ort_session is not None:
        sy = 0.128 * ((x / 1000) ** 0.90) * 1000
        D = (sy ** 2 * u) / (2 * max(x, 10))
        inputs = np.array([[x, y, u, D]], dtype=np.float32)
        ans = ort_session.run(None, {'input': inputs})
        return {"mode": "ai", "concentration": float(ans[0][0][0])}

    c = gaussian_concentration(x, y, u, Q, 'D')
    return {"mode": "classic", "concentration": float(c)}


# ── HTTP endpoint (preserved for backward compatibility) ──────────────────────
@app.post("/infer_plume")
def infer_plume(req: InferenceRequest):
    return run_inference(req.mode, req.x, req.y, req.u, req.Q)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws/plume")
async def ws_plume(websocket: WebSocket):
    await websocket.accept()
    print("React frontend connected to plume inference WebSocket.")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                result = run_inference(
                    mode=payload.get("mode", "classic"),
                    x=float(payload.get("x", 0)),
                    y=float(payload.get("y", 0)),
                    u=float(payload.get("u", 1)),
                    Q=float(payload.get("Q", 447000.0))
                )
                await websocket.send_text(json.dumps(result))
            except Exception as parse_err:
                await websocket.send_text(json.dumps({"error": str(parse_err)}))
    except WebSocketDisconnect:
        print("React frontend disconnected from plume WebSocket.")


# ── Gesture WebSocket endpoint ──────────────────────────────────────────────
@app.websocket("/ws/gestures")
async def ws_gestures(websocket: WebSocket):
    await websocket.accept()
    print("[GestureTracker] React Localhost connected!")
    try:
        while True:
            await websocket.send_json(gesture_state)
            await asyncio.sleep(1/30) # 30 FPS broadcast
    except WebSocketDisconnect:
        print("[GestureTracker] React Localhost disconnected.")


# ── Wind Status Endpoint ────────────────────────────────────────────────────
@app.get("/api/wind")
async def get_wind():
    return gesture_state["wind"]


# Reload Trigger
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
