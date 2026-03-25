import React, { useEffect, useRef, useState } from 'react';
import Map from 'react-map-gl/maplibre';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Loader2 } from 'lucide-react';

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function GestureMap() {
    // Refs for DOM elements
    const videoRef = useRef(null);
    const mapRef = useRef(null);
    
    // State
    const [isModelLoading, setIsModelLoading] = useState(true);

    // AI & Loop Refs
    const landmarkerRef = useRef(null);
    const requestRef = useRef(null);
    
    // Tracking state using refs to avoid re-renders during 60FPS loop
    const stateRef = useRef({
        prevIndex: null,
        smoothedIndex: { x: 0, y: 0 },
        prevDistance: null,
        smoothedDistance: 0,
        lastVideoTime: -1
    });

    useEffect(() => {
        let isCancelled = false;

        const initMediaPipe = async () => {
            try {
                // 1. Load the WebAssembly files from CDN
                const filesetResolver = await FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.12/wasm"
                );
                
                // 2. Initialize HandLandmarker model
                const landmarker = await HandLandmarker.createFromOptions(filesetResolver, {
                    baseOptions: {
                        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                        delegate: "GPU"
                    },
                    runningMode: "VIDEO",
                    numHands: 1
                });

                if (isCancelled) return;
                landmarkerRef.current = landmarker;

                // 3. Start user webcam
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });

                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                    videoRef.current.addEventListener("loadeddata", () => {
                        setIsModelLoading(false);
                        predictWebcam();
                    });
                }
            } catch (err) {
                console.error("Error initializing MediaPipe or Webcam:", err);
            }
        };

        initMediaPipe();

        // Cleanup on unmount
        return () => {
            isCancelled = true;
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
            if (landmarkerRef.current) landmarkerRef.current.close();
            if (videoRef.current && videoRef.current.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    const predictWebcam = () => {
        const video = videoRef.current;
        const landmarker = landmarkerRef.current;
        const map = mapRef.current?.getMap(); // Get the underlying MapLibre instance

        // Keep looping unless elements have unmounted
        if (!video || !landmarker || !map) {
             requestRef.current = requestAnimationFrame(predictWebcam);
             return;
        }

        let startTimeMs = performance.now();
        
        // Detect hands only if the video frame has advanced
        if (stateRef.current.lastVideoTime !== video.currentTime) {
            stateRef.current.lastVideoTime = video.currentTime;
            const results = landmarker.detectForVideo(video, startTimeMs);

            if (results.landmarks && results.landmarks.length > 0) {
                const landmarks = results.landmarks[0];
                
                // Landmark 8: Index Finger Tip
                // Landmark 4: Thumb Tip
                const index = landmarks[8];
                const thumb = landmarks[4];
                
                // --- 1. MIRROR & SCALE INPUT ---
                // We flip the X-axis (1.0 - index.x) so moving the hand to the physical right 
                // intuitively maps to a 'right' movement on the screen.
                // Multiplier scales the normalized [0, 1] into a workable resolution.
                const targetX = (1.0 - index.x) * 1000; 
                const targetY = index.y * 1000;

                // --- 2. EMA SMOOTHING (Low-Pass Filter) ---
                const ALPHA = 0.2; // 0.2 provides a good balance between responsiveness and smoothness
                stateRef.current.smoothedIndex.x += ALPHA * (targetX - stateRef.current.smoothedIndex.x);
                stateRef.current.smoothedIndex.y += ALPHA * (targetY - stateRef.current.smoothedIndex.y);

                const sIndex = stateRef.current.smoothedIndex;

                // --- 3. PANNING LOGIC ---
                if (stateRef.current.prevIndex) {
                    const dx = sIndex.x - stateRef.current.prevIndex.x;
                    const dy = sIndex.y - stateRef.current.prevIndex.y;

                    // map.panBy() takes pixel offset. Positive X moves the camera viewport to the right 
                    // (which results in the map visually moving LEFT). 
                    // To make the map move right when dx is positive, we use a negative offset.
                    const PAN_SENSITIVITY = 1.5;
                    const offsetX = -dx * PAN_SENSITIVITY;
                    const offsetY = -dy * PAN_SENSITIVITY;

                    // Deadzone to avoid micro-jitters
                    if (Math.abs(offsetX) > 0.5 || Math.abs(offsetY) > 0.5) {
                        map.panBy([offsetX, offsetY], { animate: false });
                    }
                }
                
                stateRef.current.prevIndex = { x: sIndex.x, y: sIndex.y };

                // --- 4. ZOOMING LOGIC ---
                // Calculate Euclidean distance between thumb and index in normalized space [0-1]
                const dxDist = (1.0 - index.x) - (1.0 - thumb.x);
                const dyDist = index.y - thumb.y;
                const dist = Math.sqrt(dxDist * dxDist + dyDist * dyDist);

                const ZOOM_ALPHA = 0.15;
                stateRef.current.smoothedDistance = stateRef.current.smoothedDistance === 0 
                    ? dist 
                    : stateRef.current.smoothedDistance + ZOOM_ALPHA * (dist - stateRef.current.smoothedDistance);

                const sDist = stateRef.current.smoothedDistance;

                if (stateRef.current.prevDistance !== null) {
                    const dDist = sDist - stateRef.current.prevDistance;
                    const ZOOM_SENSITIVITY = 15; // Maps finger pinch distance directly to zoom levels
                    
                    // Deadzone threshold for zoom
                    if (Math.abs(dDist) > 0.002) { 
                        const currentZoom = map.getZoom();
                        map.jumpTo({ zoom: currentZoom + dDist * ZOOM_SENSITIVITY });
                    }
                }
                stateRef.current.prevDistance = sDist;

            } else {
                // Reset tracking vectors if the hand is lost from frame
                stateRef.current.prevIndex = null;
                stateRef.current.prevDistance = null;
            }
        }

        // Loop next frame
        requestRef.current = requestAnimationFrame(predictWebcam);
    };

    return (
        <div className="relative w-screen h-screen bg-black overflow-hidden font-sans">
            {/* MapLibre GL JS Container */}
            <Map
                ref={mapRef}
                initialViewState={{
                    longitude: 103.763,
                    latitude: 1.296,
                    zoom: 14
                }}
                mapStyle={MAP_STYLE}
                interactive={false} // Disable mouse interactions to enforce gesture-only control
                style={{ width: '100%', height: '100%' }}
            />

            {/* Picture-in-Picture Webcam Overlay */}
            <div className="absolute top-4 right-4 w-64 h-48 bg-gray-900 border-2 border-slate-700/80 rounded-xl overflow-hidden shadow-2xl z-50">
                {isModelLoading && (
                    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-gray-900/80 text-white backdrop-blur-sm">
                        <Loader2 className="w-8 h-8 animate-spin mb-3 text-blue-500" />
                        <span className="text-sm font-medium tracking-wide">Loading AI Models...</span>
                    </div>
                )}
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    // Tailwind class `-scale-x-100` mirrors the video input visually
                    className="w-full h-full object-cover transform -scale-x-100" 
                />
            </div>

            {/* User Instructions / Helper Text */}
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-gray-900/85 backdrop-blur-md px-6 py-3 rounded-full border border-gray-700/50 text-white text-sm shadow-xl z-50 pointer-events-none">
                <span className="font-semibold text-blue-400 mr-2">Hand Gestures Active:</span> 
                Track your index finger to pan, pinch your thumb and index together to zoom.
            </div>
        </div>
    );
}
