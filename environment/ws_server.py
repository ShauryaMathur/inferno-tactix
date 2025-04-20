# ws_server.py - Updated to forward messages directly

import asyncio
import websockets
import json

client_websocket = None

async def _reader_task(websocket, msg_queue):
    try:
        async for message in websocket:
            print(f"📨 Received message: {message}")
            # Forward all messages to the queue
            await msg_queue.put(message)
            print(f"📨 Message added to queue. Queue size now: {msg_queue.qsize()}")
    except Exception as e:
        print(f"Error in _reader_task: {e}")

async def handler(websocket, path, msg_queue):
    global client_websocket
    if client_websocket is not None:
        print("🚫 Another connection attempted — closing previous one.")
        await client_websocket.close()

    print("🔥 React connected!")
    client_websocket = websocket

    try:
        await _reader_task(websocket, msg_queue)  # Handle receiving messages
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        print("React disconnected.")
        client_websocket = None

async def start_ws_server(msg_queue):
    print("🟢 Starting WebSocket server...")
    return await websockets.serve(lambda ws, path: handler(ws, path, msg_queue), "localhost", 8765)

async def get_client_ws(msg_queue, timeout=60):
    global client_websocket
    print("🧠 Waiting for React to connect...")

    for _ in range(timeout * 2):
        if client_websocket is not None and client_websocket.open:
            print(f"✅ React client is ready. [id: {id(client_websocket)}]")
            return client_websocket
        await asyncio.sleep(0.5)

    raise TimeoutError("React client didn't connect in time.")