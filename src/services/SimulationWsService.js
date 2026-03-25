/**
 * SimulationWsService.js
 *
 * Singleton WebSocket client that maintains a persistent connection to the
 * CrowdShield Python backend at ws://localhost:8000/ws/plume.
 *
 * Usage:
 *   import SimulationWsService from '../services/SimulationWsService';
 *
 *   SimulationWsService.connect(onStatusChange);   // Mount
 *   SimulationWsService.inferPlume(params);         // Returns Promise<result|null>
 *   SimulationWsService.disconnect();               // Unmount
 */

const WS_URL = 'ws://localhost:8000/ws/plume';
const RECONNECT_DELAY_MS = 3000;

let ws = null;
let pendingResolvers = []; // Queue of { resolve, reject } for in-flight requests
let statusListeners = new Set();
let reconnectTimer = null;
let intentionalClose = false;

function notifyStatus(status) {
    statusListeners.forEach(fn => fn(status));
}

function openConnection() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;

    intentionalClose = false;
    notifyStatus('connecting');

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log('[SimulationWsService] Connected to Python plume backend.');
        notifyStatus('connected');

        // Flush any queued messages that arrived before connection was ready
        // (none expected, but defensive)
    };

    ws.onmessage = (event) => {
        try {
            const result = JSON.parse(event.data);
            const resolver = pendingResolvers.shift();
            if (resolver) resolver.resolve(result);
        } catch (err) {
            const resolver = pendingResolvers.shift();
            if (resolver) resolver.reject(err);
        }
    };

    ws.onerror = (err) => {
        console.warn('[SimulationWsService] WebSocket error:', err);
    };

    ws.onclose = () => {
        console.log('[SimulationWsService] Connection closed.');
        notifyStatus('disconnected');

        // Reject all pending promises
        pendingResolvers.forEach(({ reject }) => reject(new Error('WebSocket closed')));
        pendingResolvers = [];

        // Auto-reconnect unless deliberately closed
        if (!intentionalClose) {
            reconnectTimer = setTimeout(() => {
                console.log('[SimulationWsService] Attempting reconnect...');
                openConnection();
            }, RECONNECT_DELAY_MS);
        }
    };
}

const SimulationWsService = {
    /**
     * Establish connection and register a status callback.
     * @param {(status: 'connecting'|'connected'|'disconnected') => void} onStatusChange
     */
    connect(onStatusChange) {
        if (onStatusChange) statusListeners.add(onStatusChange);
        openConnection();
    },

    /**
     * Remove status listener and close the socket if no listeners remain.
     * @param {Function} onStatusChange - same reference passed to connect()
     */
    disconnect(onStatusChange) {
        if (onStatusChange) statusListeners.delete(onStatusChange);

        if (statusListeners.size === 0) {
            intentionalClose = true;
            clearTimeout(reconnectTimer);
            if (ws) {
                ws.close();
                ws = null;
            }
            pendingResolvers = [];
        }
    },

    /**
     * Send a plume inference request and await the result.
     * Returns null if the socket is not connected (graceful fallback).
     * @param {{ mode?: string, x: number, y: number, u: number, Q?: number }} params
     * @returns {Promise<{ mode: string, concentration: number } | null>}
     */
    inferPlume(params) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            return Promise.resolve(null); // Graceful fallback — caller can use client-side physics
        }

        return new Promise((resolve, reject) => {
            pendingResolvers.push({ resolve, reject });
            ws.send(JSON.stringify({
                mode: params.mode || 'classic',
                x: params.x,
                y: params.y,
                u: params.u,
                Q: params.Q || 447000.0,
            }));
        });
    },

    /** Returns true if the socket is currently open. */
    isConnected() {
        return ws?.readyState === WebSocket.OPEN;
    },
};

export default SimulationWsService;
