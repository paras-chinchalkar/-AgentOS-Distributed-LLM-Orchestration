"""
WebSocket telemetry endpoint — streams real-time agent events to the dashboard.
"""
import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.message_bus import MessageBus

logger = logging.getLogger(__name__)
ws_router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info("Dashboard client connected. Total: %d", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        logger.info("Dashboard client disconnected. Total: %d", len(self.active))

    async def broadcast(self, data: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


@ws_router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    await manager.connect(websocket)
    from src.api.main import message_bus
    try:
        async for event in message_bus.subscribe_events("agent_events"):
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except asyncio.CancelledError:
        manager.disconnect(websocket)
