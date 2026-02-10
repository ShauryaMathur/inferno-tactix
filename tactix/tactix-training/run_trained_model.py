import signal
import sys
import traceback
import os
import torch
import asyncio
import threading
import websockets
import queue
import time
import json

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from FireEnv import FireEnvSync

from training_utils import clear_gpu_memory, get_optimal_device

# Global variables for WebSocket connection
client_websocket = None
cells_data_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

# WebSocket server thread function
def websocket_server_thread():
    """Run the WebSocket server in a separate thread"""
    async def handler(websocket):
        global client_websocket
        
        if client_websocket is not None:
            await client_websocket.close()
        
        print("🔥 React frontend connected!")
        client_websocket = websocket
        
        try:
            # Handle ping messages
            async def handle_ping():
                try:
                    await websocket.send(json.dumps({"type": "pong"}))
                    print("📤 Sent pong response")
                except Exception as e:
                    print(f"Error sending pong: {e}")
            
            # Process incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        print("🏓 Received ping, sending pong")
                        await handle_ping()
                        continue
                except Exception:
                    pass
                
                # Put other messages in the queue for the environment
                try:
                    pass  # No message queue needed for this use case
                except queue.Full:
                    pass
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            print("React frontend disconnected.")
            if client_websocket == websocket:
                client_websocket = None
    
    async def run_server():
        host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
        port = int(os.environ.get("WEBSOCKET_PORT", "8765"))

        server = await websockets.serve(
            handler, 
            host, 
            port, 
            max_size=None, 
            compression='deflate'
        )
        print(f"🟢 WebSocket server started on {host}:{port}")
        
        # Keep server running until stop event is set
        while not stop_event.is_set():
            # Check for cell data to send
            try:
                cell_data = cells_data_queue.get_nowait()
                if client_websocket is not None:
                    await client_websocket.send(json.dumps(cell_data))
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error sending cell data: {e}")
            
            await asyncio.sleep(0.05)
        
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

print('EXECUTING TRAINED MODEL RUN...')

# Configuration
logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)

save_path = os.environ.get("MODEL_DIR", ".")
SAVED_AGENT_NAME = "ppo_firefighter_improved"
MODEL_FILE = os.path.join(save_path, SAVED_AGENT_NAME + ".zip")

NUM_EPISODES = 5

model = None

def send_cells_data(env, step_num):
    """Send cell data via WebSocket"""
    try:
        cells = env.cells
        
        # Build cell data structure
        cells_data = []
        for idx, cell in enumerate(cells):
            cell_info = {
                'idx': idx,
                'x': int(cell.x),
                'y': int(cell.y),
                'fireState': int(cell.fireState.value) if hasattr(cell.fireState, 'value') else int(cell.fireState),
                'ignitionTime': float(cell.ignitionTime) if cell.ignitionTime != float('inf') else -1,
                'zoneIdx': int(cell.zoneIdx),
                'isRiver': bool(cell.isRiver),
                'helitackDropCount': int(cell.helitackDropCount)
            }
            cells_data.append(cell_info)
        
        # Get helicopter position
        heli_coord = env.state.get('helicopter_coord', [0, 0])
        
        # Get fire statistics
        cells_burning, cells_burnt = env._get_current_fire_stats()
        
        # Prepare message
        message = {
            'type': 'step_update',
            'step': step_num,
            'cells': cells_data,
            'helicopter': {'x': int(heli_coord[0]), 'y': int(heli_coord[1])},
            'stats': {
                'cellsBurning': int(cells_burning),
                'cellsBurnt': int(cells_burnt)
            }
        }
        
        # Put message in queue
        try:
            cells_data_queue.put(message, block=False)
        except queue.Full:
            try:
                cells_data_queue.get_nowait()
                cells_data_queue.put(message, block=False)
            except queue.Empty:
                pass
                
    except Exception as e:
        print(f"Error building cells data: {e}")
        traceback.print_exc()

def make_env(rank=0):
    def _init():
        env = FireEnvSync(env_id=rank)
        env = Monitor(env, logdir)
        print(f"[ENV {rank}]")
        return env
    return _init

def run_episode(model, env, episode_num):
    """Run a single episode with the trained model"""
    obs, _ = env.reset()
    done = False
    episode_steps = 0
    
    print(f"🎮 Starting Episode {episode_num + 1}...")
    
    while not done:
        # Get action from the model
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Send cell data via WebSocket every step
        send_cells_data(env, episode_steps)
        
        episode_steps += 1
        
        # Small delay to allow WebSocket to send data
        time.sleep(0.01)
        
        if done:
            cells_burning, cells_burnt = env._get_current_fire_stats()
            print(f"✅ Episode {episode_num + 1} completed in {episode_steps} steps")
            print(f"   Burning cells: {cells_burning}, Burnt cells: {cells_burnt}")
    
    return episode_steps

def main():
    global model
    
    # Start WebSocket server
    print("Starting WebSocket server...")
    server_thread = threading.Thread(target=websocket_server_thread, daemon=True)
    server_thread.start()
    
    # Wait for WebSocket server to start
    print("Waiting for WebSocket server to start...")
    time.sleep(2)
    
    # Wait for React to connect
    print("Waiting for React to connect...")
    timeout = 500
    start_time = time.time()
    
    while client_websocket is None:
        time.sleep(0.5)
        if time.time() - start_time > timeout:
            print("Timeout waiting for React to connect")
            return
    
    print(f"✅ React client connected: {id(client_websocket)}")
    
    # Load the trained model
    if not os.path.isfile(MODEL_FILE):
        print(f"❌ Model file not found: {MODEL_FILE}")
        print("Please train a model first using train_with_render.py")
        return
    
    print("🔍 Loading trained model...")
    env = make_env(0)()
    device, is_gpu = get_optimal_device()
    
    try:
        model = PPO.load(MODEL_FILE, device=device, env=env)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return
    
    # Run episodes
    print(f"🚀 Running {NUM_EPISODES} episodes...")
    
    for episode in range(NUM_EPISODES):
        try:
            run_episode(model, env, episode)
            # Small pause between episodes
            time.sleep(1)
        except Exception as e:
            print(f"❌ Error in episode {episode + 1}: {e}")
            traceback.print_exc()
    
    print(f"✅ Completed {NUM_EPISODES} episodes!")
    
    if client_websocket is not None:
        try:
            await_message = {
                'type': 'done',
                'message': f'Completed {NUM_EPISODES} episodes'
            }
            # Send done message via queue
            cells_data_queue.put(await_message, block=False)
        except Exception as e:
            print(f"Error sending completion message: {e}")
    
    print("👋 Run complete. Exiting.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        traceback.print_exc()
    finally:
        stop_event.set()
        print("🔴 WebSocket server stopped")

