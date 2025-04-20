# train.py - Completely redesigned to avoid event loop issues

import asyncio
import threading
import queue
import json
import websockets
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.policies import ActorCriticPolicy

from infernoenv import FireEnvSyncWrapper
from FeatureExtractor import FireEnvCNN


MAX_TIMESTEPS = 2000
# Global variables for the WebSocket connection
client_websocket = None
message_queue = queue.Queue()
stop_event = threading.Event()

def websocket_server_thread():
    """Run the WebSocket server in a separate thread"""
    async def handler(websocket, path):
        global client_websocket
        
        if client_websocket is not None:
            await client_websocket.close()
        
        print("🔥 React connected!")
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
                # print(f"📨 Received message: {message}")
                
                # Handle ping specially
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        print("🏓 Received ping, sending pong")
                        await handle_ping()
                        continue
                except Exception:
                    pass
                
                # Put other messages in the queue for the environment
                message_queue.put(message)
                print(f"📨 Message added to queue. Size: {message_queue.qsize()}")
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            print("React disconnected.")
            if client_websocket == websocket:
                client_websocket = None
    
    async def run_server():
        server = await websockets.serve(handler, "localhost", 8765)
        print("🟢 WebSocket server started on localhost:8765")
        
        # Keep server running until stop event is set
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        
        server.close()
        await server.wait_closed()
        print("🔴 WebSocket server stopped")
    
    # Set up event loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the server
    loop.run_until_complete(run_server())
    loop.close()

# Synchronous environment implementation
class FireEnvSync(gym.Env):
    def __init__(self):
        super().__init__()
        global client_websocket, message_queue
        
        # Store references to global communication channels
        self.websocket = None  # Will be set when needed
        self.msg_queue = message_queue
        
        # Initialize spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0, high=8, shape=(160, 240), dtype=np.int32),
            'on_fire': spaces.Discrete(2)
        })
        
        # Initialize state
        self.step_count = 0
        self.state = self._default_state()
    
    def _default_state(self):
        return {
            'helicopter_coord': np.array([10, 10], dtype=np.int32),
            'cells': np.zeros((160, 240), dtype=np.int32),
            'on_fire': 0,
            'prevBurntCells': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenchedCells': 0
        }
    
    def _send_and_wait(self, message, timeout=5.0):
        """Send a message to React and wait for a response"""
        global client_websocket
        
        # Get the current websocket
        self.websocket = client_websocket
        if not self.websocket:
            print("⚠️ No active WebSocket connection")
            return None
        
        # Clear any old messages
        while not self.msg_queue.empty():
            try:
                self.msg_queue.get_nowait()
            except queue.Empty:
                break
        
        # print(f"📤 Sending message: {message}")
        asyncio.run(self._send_message(message))
        
        # Wait for a response with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to get a response (with a short timeout)
                response = self.msg_queue.get(timeout=0.1)
                # print(f"📥 Received response: {response}")
                
                # Parse the response
                try:
                    data = json.loads(response)
                    return data
                except Exception as e:
                    print(f"❌ Error parsing response: {e}")
                    return None
            except queue.Empty:
                # No response yet, check if WebSocket is still connected
                if client_websocket != self.websocket:
                    print("⚠️ WebSocket connection changed")
                    return None
        
        # Timeout reached
        print(f"⚠️ Timeout waiting for response after {timeout}s")
        return None
    
    async def _send_message(self, message):
        """Send a message to the WebSocket"""
        if self.websocket:
            try:
                await self.websocket.send(message)
            except Exception as e:
                print(f"❌ Error sending message: {e}")
    
    def reset(self, *, seed=None, options=None):
        print(f"🧼 Resetting environment with seed={seed}")
        super().reset(seed=seed)
        self.step_count = 0
        
        # Create reset message
        reset_message = json.dumps({"action": "reset"})
        
        # Send and wait for response
        response = self._send_and_wait(reset_message)

        self.state = {
            **self.state,
            ** response
        }
        
        # Create observation
        final_observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8),
            'on_fire': int(self.state['on_fire'])
        }
        
        return final_observation, {}
    
    def step(self, action):
        self.step_count += 1
        print(f"\n[Step {self.step_count}] Action taken: {action}")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0: heli_y += 5     
        elif action == 1: heli_y -= 5   
        elif action == 2: heli_x -= 5   
        elif action == 3: heli_x += 5   
        elif action == 4: print(f"Helitack performed at ({heli_x}, {heli_y})")
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 10, 239))
        heli_y = int(np.clip(heli_y, 10, 159))
        
        # Create step message
        step_message = json.dumps({
            "action": str(action),
            "helicopter_coord": [heli_x, heli_y]
        })
        
        # Send and wait for response
        response = self._send_and_wait(step_message, timeout=1.0)

        self.state = {
            **self.state,
            ** response,
            'helicopter_coord': [heli_x, heli_y]
        }
        
        # Process response or use default values
        reward = self.calculate_reward(
            self.state['prevBurntCells'],
            self.state['cellsBurnt'],
            self.state['cellsBurning'],
            self.state['quenchedCells']
        )
        done = False
        
        if self.state['cellsBurning'] == 0:
            done = True
        
        # Check for max steps
        if self.step_count >= MAX_TIMESTEPS:
            done = True
        
        # Create observation
        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8),
            'on_fire': int(self.state['on_fire'])
        }
        
        return observation, reward, done, False, {}
    
    def calculate_reward(self,prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        """
        Compute reward for current step based on fire status.

        :param prev_burnt: int - Number of burnt cells in previous step
        :param curr_burnt: int - Number of burnt cells in current step
        :param curr_burning: int - Number of currently burning cells
        :param extinguished_by_helitack: int - Number of cells extinguished this step
        :return: float - Calculated reward
        """
        newly_burnt = curr_burnt - prev_burnt

        reward = 0

        reward += extinguished_by_helitack * 10        # reward for extinguishing
        reward -= newly_burnt * 5                      # penalty for fire spread
        reward -= curr_burning * 1                     # penalty for ongoing fires
        reward -= 0.1                                  # step penalty

        return reward
    
    def close(self):
        print("Closing environment...")
        
        # Send close message
        close_message = json.dumps({"action": "close"})
        asyncio.run(self._send_message(close_message))

class FireEnvPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, 
                            features_extractor_class=FireEnvCNN,
                            features_extractor_kwargs=dict(features_dim=256),
                            **kwargs)
# Main function
def main():
    # Start WebSocket server in a separate thread
    server_thread = threading.Thread(target=websocket_server_thread, daemon=True)
    server_thread.start()
    
    # Wait for WebSocket server to start
    print("Waiting for WebSocket server to start...")
    time.sleep(2)
    
    # Wait for React to connect
    print("Waiting for React to connect...")
    timeout = 60
    start_time = time.time()
    while client_websocket is None:
        time.sleep(0.5)
        if time.time() - start_time > timeout:
            print("Timeout waiting for React to connect")
            return
    
    print(f"✅ React client connected: {id(client_websocket)}")
    
    # Create environment
    env = FireEnvSync()
    
    # Check environment
    try:
        check_env(env)
        print("✅ Environment check completed successfully!")
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        print("⚠️ Attempting to continue anyway")
    
    # Train model
    print("🚀 Starting training...")
    # await asyncio.sleep(5)
    callback = StopTrainingOnMaxEpisodes(max_episodes=100)

    model = PPO(
    FireEnvPolicy,
    env,
    verbose=1,
    n_steps=10,        # Collect exactly 10 steps before updating
    batch_size=10,     # Use all collected steps in a single update
    n_epochs=1,        # Perform only 1 optimization epoch per update
    learning_rate=0.0003,
    clip_range=0.2,
    gamma=0.99,        # Discount factor
    gae_lambda=0.95,   # GAE parameter
)
    model.learn(total_timesteps=100000, callback=callback)
    model.save("ppo_firefighter")
    print("✅ Training complete and model saved.")
    
    # Stop WebSocket server
    stop_event.set()
    server_thread.join(timeout=5)

if __name__ == "__main__":
    main()