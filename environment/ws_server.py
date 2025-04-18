import asyncio
import websockets
import json

client_websocket = None

# Function that handles receiving messages asynchronously and puts them in the queue
async def _reader_task(websocket, msg_queue):
    try:
        async for message in websocket:
            print(f"📨 Received message: {message}")
            # Put the received message into the msg_queue for processing by other parts of the program
            await msg_queue.put(message)
            print(f"📨 Message added to queue. Queue size now: {msg_queue.qsize()}")
    except Exception as e:
        print(f"Error in _reader_task: {e}")

# This is the main handler that sets up the WebSocket connection
async def handler(websocket, path, msg_queue):
    global client_websocket
    if client_websocket is not None:
        print("🚫 Another connection attempted — closing previous one.")
        await client_websocket.close()

    print("🔥 React connected!")
    client_websocket = websocket

    # Once the connection is established, we delegate the message reading to the _reader_task
    try:
        await _reader_task(websocket, msg_queue)  # This function will handle receiving messages
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        print("React disconnected.")
        client_websocket = None

# Function to start the WebSocket server
async def start_ws_server(msg_queue):
    print("🟢 Starting WebSocket server...")
    return await websockets.serve(lambda ws, path: handler(ws, path, msg_queue), "localhost", 8765)

# This function waits for the React client to connect
async def get_client_ws(msg_queue, timeout=60):
    global client_websocket
    print("🧠 Waiting for React to connect...")

    for _ in range(timeout * 2):
        if client_websocket is not None and client_websocket.open:
            print(f"✅ React client is ready. [id: {id(client_websocket)}]")
            return client_websocket
        await asyncio.sleep(0.5)

    raise TimeoutError("React client didn't connect in time.")
