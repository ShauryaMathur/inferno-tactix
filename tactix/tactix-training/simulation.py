import json
import numpy as np
import torch
import time
import asyncio
import websockets
from stable_baselines3 import PPO
import gymnasium as gym
import threading
import queue

from train import FireEnvSync

# Global WebSocket variables
client_websocket = None
message_queue = queue.Queue(maxsize=100)  # Limit queue size
stop_event = threading.Event()

# WebSocket server thread to handle communication
def websocket_server_thread():
    """Run the WebSocket server in a separate thread"""
    async def handler(websocket, path):
        global client_websocket
        
        if client_websocket is not None:
            await client_websocket.close()
        
        print("🔥 React connected!")
        client_websocket = websocket
        
        try:
            # Process incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        print("🏓 Received ping, sending pong")
                        await websocket.send(json.dumps({"type": "pong"}))
                    else:
                        # Put the message in the queue for processing by the environment
                        try:
                            message_queue.put(data, block=False)
                        except queue.Full:
                            message_queue.get_nowait()
                            message_queue.put(data)
                except Exception as e:
                    print(f"Error handling message: {e}")
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            print("React disconnected.")
            if client_websocket == websocket:
                client_websocket = None
    
    async def run_server():
        host = '0.0.0.0'  # Listen on all available interfaces
        port = 8765  # WebSocket port
        server = await websockets.serve(handler, host, port)
        print(f"🟢 WebSocket server started on {host}:{port}")
        
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        
        server.close()
        await server.wait_closed()
        print("🔴 WebSocket server stopped")
    
    # Set up event loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the server
    try:
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"Error in WebSocket server: {e}")
    finally:
        loop.close()

# Async function to send message through WebSocket
async def send_action_to_client(action_message):
    """Send action message to the React client via WebSocket"""
    global client_websocket
    if client_websocket is not None:
        try:
            await client_websocket.send(action_message)
            print("📤 Action message sent to React client")
        except Exception as e:
            print(f"❌ Error sending message: {e}")
    else:
        print("⚠️ No active WebSocket connection")

def simulate(model_path="ppo_firefighter.zip", total_episodes=5):
    """
    Simulates the PPO model for a set number of episodes with WebSocket communication.

    Args:
        model_path (str): Path to the trained PPO model.
        total_episodes (int): Number of episodes to simulate.
    """
    # Load the trained PPO model
    print(f"🔄 Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create the environment (replace with your environment)
    env = FireEnvSync()  # Make sure this is your custom environment

    # Start WebSocket server in a separate thread
    server_thread = threading.Thread(target=websocket_server_thread, daemon=True)
    server_thread.start()

    # Run for the specified number of episodes
    for episode in range(total_episodes):
        print(f"🚀 Episode {episode + 1}/{total_episodes} starting...")

        # Reset environment to initial state
        obs, _ = env.reset()  # Add _ to ignore the optional info
        done = False
        episode_reward = 0
        
        while not done:
            # Select an action using the trained PPO model
            action, _states = model.predict(obs, deterministic=True)

            # Convert NumPy arrays to lists before sending
            action_message = json.dumps({
                "action": action,
                "helicopter_coord": obs["helicopter_coord"].tolist()  # Convert ndarray to list
            })

            # Use asyncio to send the action message asynchronously
            asyncio.create_task(send_action_to_client(action_message))

            # Step the environment with the selected action
            obs, reward, done, _, _ = env.step
