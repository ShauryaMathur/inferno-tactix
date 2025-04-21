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
import signal
import sys
import traceback
import os
import pdb

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.policies import ActorCriticPolicy

from FeatureExtractor import FireEnvCNN

class FireState:
    Unburnt = 0
    Burning = 1
    Burnt = 2

class BurnIndex:
    Low = 0
    Medium = 1
    High = 2

MAX_TIMESTEPS = 2000
HELICOPTER_SPEED = 5
USE_TRAINED_AGENT = False
# Global variables for the WebSocket connection
client_websocket = None
message_queue = queue.Queue()
stop_event = threading.Event()
model = None  # Global model reference for signal handler

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\n⚠️ Received termination signal. Cleaning up...")
    if model is not None:
        try:
            print("💾 Saving model...")
            model.save("ppo_firefighter_interrupted")
            print("✅ Model saved as ppo_firefighter_interrupted")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    stop_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

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
                # print(f"📨 Message added to queue. Size: {message_queue.qsize()}")
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
    try:
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"Error in WebSocket server: {e}")
    finally:
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
        self.episode_count = 0
        self.state = self._default_state()
    
    def _default_state(self):
        return {
            'helicopter_coord': np.array([70, 115], dtype=np.int32),
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
        try:
            asyncio.run(self._send_message(message))
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return None
        
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
                raise
    
    def reset(self, *, seed=None, options=None):
        print(f"🧼 Resetting environment with seed={seed}")
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        
        # Create reset message
        reset_message = json.dumps({"action": "reset","episode": self.episode_count})
        
        # Send and wait for response
        response = self._send_and_wait(reset_message)
        if response is None:
            print("⚠️ No response from React, using default state")
            response = {}

        self.state = self._default_state()
        self.state = {
            **self.state,
            **response
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
        self.state['last_action'] = action

        print(f"\n[Step {self.step_count}] Action taken: {action}")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0: heli_y += HELICOPTER_SPEED
        elif action == 1: heli_y -= HELICOPTER_SPEED  
        elif action == 2: heli_x -= HELICOPTER_SPEED 
        elif action == 3: heli_x += HELICOPTER_SPEED  
        elif action == 4: print(f"Helitack performed at ({heli_x}, {heli_y})")
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        # Create step message
        step_message = json.dumps({
            "action": str(action),
            "helicopter_coord": [heli_x, heli_y]
        })
        
        # Send and wait for response
        response = self._send_and_wait(step_message, timeout=1.0)
        if response is None:
            print("⚠️ No response from React, maintaining current state")
            response = {}

        # self.state['prevBurntCells'] = self.state['cellsBurnt']
        self.state = {
            **self.state,
            'prevBurntCells': self.state['cellsBurnt'],
            **response,
            'helicopter_coord': [heli_x, heli_y]
        }
        
        # Process response or use default values
        reward = self.calculate_reward(
            self.state.get('prevBurntCells', 0),
            self.state.get('cellsBurnt', 0),
            self.state.get('cellsBurning', 0),
            self.state.get('quenchedCells', 0)
        )
        

        done = False
        
        if self.state.get('cellsBurning', 1) == 0:
            done = True
        
        # Check for max steps
        if self.step_count >= MAX_TIMESTEPS:
            done = True
        
        # Create observation
        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8),
            'on_fire': int(self.state.get('on_fire', 0))
        }
        
        return observation, reward, done, False, {}
    
    def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0
        newly_burnt = curr_burnt - prev_burnt
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        
        # Base rewards/penalties
        reward += extinguished_by_helitack * 100  # Reward for extinguishing fires
        reward -= newly_burnt * 10  # Penalty for new burnt cells
        reward -= curr_burning * 2   # Penalty for ongoing fires
        reward -= 0.1               # Small step penalty
        
        # Extract fire states from cells
        burning_mask = (cells // 3) == FireState.Burning
        burnt_mask = (cells // 3) == FireState.Burnt
        
        # Proximity reward to active fires
        if np.any(burning_mask):
            burning_coords = np.argwhere(burning_mask)
            min_distance = float('inf')
            
            for by, bx in burning_coords:
                distance = np.sqrt((heli_y - by)**2 + (heli_x - bx)**2)
                min_distance = min(min_distance, distance)
            
            # Exponential decay proximity reward
            proximity_reward = 10 * np.exp(-min_distance / 15)
            reward += proximity_reward
        
        # Helitack usage rewards/penalties
        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            fire_state = cells[heli_y, heli_x] // 3
            burn_index = cells[heli_y, heli_x] % 3
            
            if fire_state == FireState.Burning:
                # Major reward for extinguishing burning cells
                intensity_bonus = (burn_index + 1) * 2  # 2, 4, or 6 based on intensity
                reward += 50 * intensity_bonus
                
                # Bonus for targeting high spread potential areas
                adjacent_unburnt = 0
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = heli_y + dy, heli_x + dx
                    if 0 <= ny < cells.shape[0] and 0 <= nx < cells.shape[1]:
                        if (cells[ny, nx] // 3) == FireState.Unburnt:
                            adjacent_unburnt += 1
                
                if adjacent_unburnt > 0:
                    # Extra reward for fires near unburnt areas (high spread potential)
                    reward += 20 * (adjacent_unburnt / 4)
            
            elif fire_state == FireState.Burnt:
                # Significant penalty for wasting helitack on burnt cells
                reward -= 50
            
            elif fire_state == FireState.Unburnt:
                # Check if creating strategic firebreak
                nearby_burning = 0
                fire_proximity_score = 0
                
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        ny, nx = heli_y + dy, heli_x + dx
                        if 0 <= ny < cells.shape[0] and 0 <= nx < cells.shape[1]:
                            if (cells[ny, nx] // 3) == FireState.Burning:
                                distance = np.sqrt(dy**2 + dx**2)
                                nearby_burning += 1
                                fire_proximity_score += 1 / (distance + 1)
                
                if nearby_burning > 0:
                    # Reward for creating firebreaks (scaled by proximity to fires)
                    reward += 10 * fire_proximity_score
                else:
                    # Penalty for using helitack with no fires nearby
                    reward -= 20
        
        # Strategic positioning reward (even when not using helitack)
        if np.any(burning_mask):
            # Calculate fire density heat map
            heat_map = np.zeros_like(cells, dtype=float)
            for by, bx in np.argwhere(burning_mask):
                for dy in range(-10, 11):
                    for dx in range(-10, 11):
                        ny, nx = by + dy, bx + dx
                        if 0 <= ny < cells.shape[0] and 0 <= nx < cells.shape[1]:
                            distance = np.sqrt(dy**2 + dx**2)
                            heat_map[ny, nx] += 1 / (distance + 1)
            
            # Reward for being in high-density fire areas
            if 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
                position_value = heat_map[heli_y, heli_x]
                reward += position_value * 2
        
        # Edge penalty (encourage staying away from map boundaries)
        edge_distance = min(heli_x, heli_y, 239 - heli_x, 159 - heli_y)
        if edge_distance < 10:
            reward -= (10 - edge_distance) * 0.5
        
        print(f"Reward: {reward:.2f}")
        return reward
    
    def close(self):
        print("Closing environment...")
        
        try:
            # Send close message
            close_message = json.dumps({"action": "close"})
            asyncio.run(self._send_message(close_message))
        except Exception as e:
            print(f"❌ Error closing environment: {e}")

class FireEnvPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, 
                        features_extractor_class=FireEnvCNN,
                        features_extractor_kwargs=dict(features_dim=256),
                        **kwargs)

# Main function
def main():
    global model
    
    env = None
    server_thread = None
    
    try:
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
        env.episode_count = 0
        # Try to load existing model for resuming training
        print("🔍 Checking for existing model...")
        try:
            # Check if model file exists before attempting to load
            if USE_TRAINED_AGENT and os.path.exists("ppo_firefighter_interrupted.zip"):
                print("🔄 Found existing model, loading for continued training...")
                model = PPO.load("ppo_firefighter_interrupted")
                model.set_env(env)
                print("✅ Model loaded successfully!")
            else:
                # No existing model, create a new one
                print("🆕 No existing model found, initializing new model...")
                model = PPO(
                    FireEnvPolicy,
                    env,
                    n_steps=10,        # Collect exactly 10 steps before updating
                    batch_size=10,     # Use all collected steps in a single update
                    n_epochs=1,        # Perform only 1 optimization epoch per update
                    learning_rate=0.0003,
                    clip_range=0.2,
                    gamma=0.99,        # Discount factor
                    gae_lambda=0.95,   # GAE parameter
                    ent_coef=0.01,     # Entropy coefficient
                    vf_coef=0.5,       # Value function coefficient
                    max_grad_norm=0.5, # Maximum gradient norm
                    target_kl=0.01,    # Target KL divergence
                    # verbose=1
                    device='mps'
                )
                print("✅ New model initialized successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🆕 Initializing new model instead...")
            model = PPO(
                FireEnvPolicy,
                env,
                n_steps=10,
                batch_size=10,
                n_epochs=1,
                learning_rate=0.0003,
                clip_range=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.01,
                device='mps'
                # verbose=1
            )
            print("✅ New model initialized successfully!")
            

        # Train model
        print("🚀 Starting training...")
        model.learn(
            total_timesteps=200000,
            reset_num_timesteps=False  # This ensures continued training
        )
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        traceback.print_exc()
    finally:
        # This code will always run, even if an exception occurs
        
        # Save model if it exists
        if model is not None:
            try:
                print("💾 Saving model...")
                model.save("inferno_tactix_ppo")
                print("✅ Model saved.")
            except Exception as e:
                print(f"❌ Error saving model: {e}")
        
        # Close environment if it exists
        if env is not None:
            try:
                env.close()
                print("🧹 Environment closed.")
            except Exception as e:
                print(f"❌ Error closing environment: {e}")
        
        # Stop WebSocket server
        print("🛑 Stopping WebSocket server...")
        stop_event.set()
        
        # Wait for server thread to finish
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=5)
            if server_thread.is_alive():
                print("⚠️ WebSocket server thread did not terminate properly")
            else:
                print("✅ WebSocket server stopped.")
        
        print("👋 Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()