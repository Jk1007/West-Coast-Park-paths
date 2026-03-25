# CrowdShield — Operational Dashboard

## 1. System Overview
CrowdShield is an integrated AI-driven operational dashboard for managing public safety incidents, specifically focused on chemical/gas plume modeling and real-time hand-gesture controls.

## 2. Installation & Setup

### Python Backend
1. Navigate to the `backend/` directory.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the unified server:
   ```
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   *Note: This server handles both Plume Inference (port 8000/ws/plume) and AI Hand Tracking (port 8000/ws/gestures).*

### React Frontend
1. Navigate to the root directory.
2. Install dependencies:
   ```
   npm install
   ```
3. Run the development server:
   ```
   npm run dev
   ```

## 3. Key Features
- **AI Plume Modeling**: Real-time Gaussian and PINN-based spread prediction via WebSockets.
- **Hand Gesture Navigation**: Control the map hands-free using your webcam.
  - **Pan**: Point with your index finger.
  - **Zoom**: Change the distance between your index finger and thumb.
  - **Stop**: Close your fist to freeze map movement.
- **Incident Dashboard**: Live reporting and database synchronization via Supabase.

## 4. Troubleshooting
- **Webcam Issues**: Ensure no other application is using your webcam before starting `main.py`.
- **WebSocket Disconnect**: If the frontend shows "Offline", ensure `uvicorn` is running on port 8000.
- **MediaPipe Errors**: Use `mediapipe==0.10.11` specifically for best compatibility with the current tracker.
