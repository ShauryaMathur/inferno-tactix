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
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


from FeatureExtractor import FireEnvLSTMPolicy

logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)

class LearningRateScheduleCallback(BaseCallback):
    def __init__(self, lr_schedule, verbose=0):
        super().__init__(verbose)
        self.lr_schedule = lr_schedule
        
    def _on_step(self):
        progress = self.num_timesteps / 200000  # Normalize by total timesteps
        new_lr = self.lr_schedule(progress)
        self.model.learning_rate = new_lr
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"Timestep {self.num_timesteps}/{200000}: Learning rate = {new_lr}")
        return True

# Define the schedule function
def lr_schedule(progress):
    return 0.0003 * (1.0 - progress)

class RewardLoggingCallback(BaseCallback):
    """
    Custom callback for logging episode rewards to TensorBoard in real-time
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.cumulative_reward = 0
        
    def _on_step(self):
        # Get reward from the last step
        reward = self.locals['rewards'][0]
        self.cumulative_reward += reward
        
        # If episode is done, log the total episode reward
        done = self.locals['dones'][0]
        if done:
            # Log cumulative reward for the episode
            self.logger.record('episode/reward', self.cumulative_reward)
            # Log episode length
            self.logger.record('episode/length', self.model.num_timesteps - sum(self.locals['dones']))
            # Reset for next episode
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0
            
            # Log mean of last 100 episodes 
            if len(self.episode_rewards) > 0:
                self.logger.record('episode/mean_reward_100', np.mean(self.episode_rewards[-100:]))
        
        return True

class FireState:
    Unburnt = 0
    Burning = 1
    Burnt = 2

class BurnIndex:
    Low = 0
    Medium = 1
    High = 2

MAX_TIMESTEPS = 2000
HELICOPTER_SPEED = 3
USE_TRAINED_AGENT = True
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
            model.save("ppo_firefighter")
            print("✅ Model saved as ppo_firefighter")
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
        server = await websockets.serve(handler, "localhost", 8765,max_size=None,compression='deflate')
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
            'cells': spaces.Box(low=0, high=8, shape=(4, 160, 240), dtype=np.int32),  # 4 stacked frames
            'on_fire': spaces.Discrete(2)
        })
        
        # Initialize state
        self.step_count = 0
        self.episode_count = 0
        self.state = self._default_state()
        
        # Frame stacking
        self.frame_history = deque(maxlen=4)
        
        # Initialize frame history with zeros
        self.frame_history.clear()
        for _ in range(4):
            self.frame_history.append(np.zeros((160, 240), dtype=np.int32))

    
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
    
    def _send_and_wait(self, message, timeout=500.0):
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
                response = self.msg_queue.get(timeout=500)
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
        if seed is not None:
            super().reset(seed=seed)
        
        if hasattr(self, 'model') and hasattr(self.model.policy, 'features_extractor'):
            self.model.policy.features_extractor.hidden = None
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
        
        # Initialize frame history
        self.frame_history.clear()
        initial_cells = np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8)
        for _ in range(4):
            self.frame_history.append(initial_cells)
        
        # Create observation with stacked frames
        stacked_cells = np.stack(list(self.frame_history), axis=0)
        
        # Ensure the shape is correct
        assert stacked_cells.shape == (4, 160, 240), f"Expected shape (4, 160, 240), got {stacked_cells.shape}"
        
        final_observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': stacked_cells,
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
        response = self._send_and_wait(step_message, timeout=500.0)
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
        
        # Update frame history
        current_cells = np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8)
        self.frame_history.append(current_cells)
        
        # Create observation with stacked frames
        stacked_cells = np.stack(list(self.frame_history), axis=0)
        
        # Ensure the shape is correct
        assert stacked_cells.shape == (4, 160, 240), f"Expected shape (4, 160, 240), got {stacked_cells.shape}"
        
        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': stacked_cells,
            'on_fire': int(self.state.get('on_fire', 0))
        }
        
        return observation, reward, done, False, {}
    
    # def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
    #     reward = 0
    #     # newly_burnt = curr_burnt - prev_burnt
    #     heli_x, heli_y = self.state['helicopter_coord']
    #     last_action = self.state.get('last_action', None)
    #     cells = np.array(self.state['cells'])
        
    #     # # Track effectiveness metrics
    #     # if not hasattr(self, 'prev_burning'):
    #     #     self.prev_burning = 0
        
    #     # # Calculate fire reduction
    #     # burning_reduction = self.prev_burning - curr_burning
        
    #     # # Base rewards/penalties
    #     reward += extinguished_by_helitack * 10  # Large reward for direct extinguishing
    #     # reward -= newly_burnt * 20  # Significant penalty for fire spread
    #     # reward -= curr_burning * 1  # Ongoing penalty proportional to fire size
        
    #     # Time penalty that increases with fire size
    #     # This encourages quick action rather than waiting
    #     # time_penalty = 0.1 * (1 + curr_burning / 100)
    #     reward -= 1
        
    #     # # Reward for ANY reduction in burning cells, regardless of cause
    #     # # But give more reward if it's due to helitack action
    #     # # if burning_reduction > 0:
    #     # #     if last_action == 4:  # Helitack was used
    #     # #         reward += burning_reduction * 10  # Higher reward for active suppression
    #     # #     else:
    #     # #         reward += burning_reduction * 2   # Lower reward for passive reduction
        
    #     # # Extract fire states
    #     fire_states = cells // 3
    #     # burning_mask = fire_states == FireState.Burning
        
    #     # # Proximity reward to active fires
    #     # # if np.any(burning_mask):
    #     # #     burning_coords = np.argwhere(burning_mask)
    #     # #     distances = np.sqrt(
    #     # #         (burning_coords[:, 0] - heli_y) ** 2 + 
    #     # #         (burning_coords[:, 1] - heli_x) ** 2
    #     # #     )
    #     # #     min_distance = np.min(distances)
            
    #     # #     # Stronger proximity reward
    #     # #     proximity_reward = 20 * np.exp(-min_distance / 10)
    #     # #     reward += proximity_reward
        
    #     # # Helitack effectiveness evaluation
    #     if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
    #         fire_state = fire_states[heli_y, heli_x]
    #         if fire_state == FireState.Burnt:
    #             reward -= 5
    #     #         # Significant penalty for wasted helitack
    #     #         reward -= 100
    #     #     burn_index = cells[heli_y, heli_x] % 3
            
    #     #     if fire_state == FireState.Burning:
    #     #         # Major reward for direct hits
    #     #         intensity_bonus = (burn_index + 1) * 3
    #     #         reward += 100 * intensity_bonus
            
    #     #     elif fire_state == FireState.Burnt:
    #     #         # Significant penalty for wasted helitack
    #     #         reward -= 100
            
    #     #     elif fire_state == FireState.Unburnt:
    #     #         # Check for strategic firebreak creation
    #     #         nearby_burning = np.sum(burning_mask[
    #     #             max(0, heli_y-5):min(160, heli_y+6),
    #     #             max(0, heli_x-5):min(240, heli_x+6)
    #     #         ])
                
    #     #         if nearby_burning > 0:
    #     #             # Reward for creating firebreaks near active fires
    #     #             reward += 30 * min(nearby_burning, 10)
    #     #         else:
    #     #             # Heavy penalty for using helitack far from fires
    #     #             reward -= 50
        
    #     # # Reward for aggressive behavior when fire is large
    #     # # if curr_burning > 50:  # Threshold for "large" fire
    #     # #     if last_action == 4:
    #     # #         reward += 20  # Bonus for being aggressive with large fires
    #     # #     else:
    #     # #         reward -= 5   # Penalty for not being aggressive
        
    #     # # Edge penalty (reduced)
    #     # # edge_distance = min(heli_x, heli_y, 239 - heli_x, 159 - heli_y)
    #     # # if edge_distance < 5:
    #     # #     reward -= (5 - edge_distance) * 0.2
        
    #     # # Update tracking variables
    #     # self.prev_burning = curr_burning
        
    #     # print(f"Reward: {reward:.2f} | Burning: {curr_burning} | Reduction: {burning_reduction}")
    #     print(f"Reward: {reward:.2f} | Burning: {curr_burning}")
    #     return reward

    def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0
        
        # Track fire progress
        if not hasattr(self, 'prev_burning'):
            self.prev_burning = curr_burning
        newly_burnt = curr_burnt - prev_burnt
        burning_reduction = self.prev_burning - curr_burning
        
        # Core objectives with clear signals
        reward += extinguished_by_helitack * 10       # Reward for direct fire suppression
        reward -= newly_burnt * 5                     # Penalty for new cells burning
        reward -= curr_burning * 0.1                  # Ongoing penalty proportional to fire size
        reward -= 0.1                                 # Small time penalty
        
        # Get positional information
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burning_mask = fire_states == FireState.Burning
        
        # Proximity guidance
        if np.any(burning_mask):
            burning_coords = np.argwhere(burning_mask)
            distances = np.sqrt(
                (burning_coords[:, 0] - heli_y) ** 2 + 
                (burning_coords[:, 1] - heli_x) ** 2
            )
            min_distance = np.min(distances)
            proximity_reward = 2 * np.exp(-min_distance / 20)
            reward += proximity_reward
        
        # Helitack evaluation
        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            fire_state = fire_states[heli_y, heli_x]
            burn_index = cells[heli_y, heli_x] % 3
            
            if fire_state == FireState.Burning:
                # Reward for direct hits based on intensity
                intensity_bonus = (burn_index + 1) * 2
                reward += 20 * intensity_bonus
            elif fire_state == FireState.Burnt:
                # Penalty for wasted helitack
                reward -= 5
            elif fire_state == FireState.Unburnt:
                # Strategic firebreak rewards
                nearby_burning = np.sum(burning_mask[
                    max(0, heli_y-5):min(160, heli_y+6),
                    max(0, heli_x-5):min(240, heli_x+6)
                ])
                if nearby_burning > 0:
                    reward += 5 * min(nearby_burning, 5)
                else:
                    reward -= 10
        
        # Update for next timestep
        self.prev_burning = curr_burning
        
        print(f"Reward: {reward:.2f} | Burning: {curr_burning} | New burnt: {newly_burnt} | Reduction: {burning_reduction}")
        return reward
    
    def close(self):
        print("Closing environment...")
        
        try:
            # Send close message
            close_message = json.dumps({"action": "close"})
            asyncio.run(self._send_message(close_message))
        except Exception as e:
            print(f"❌ Error closing environment: {e}")


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
        timeout = 500
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
        env = Monitor(env, filename=logdir)  
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_obs_keys=["helicopter_coord", "cells"],
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        # Try to load existing model for resuming training
        print("🔍 Checking for existing model...")
        try:
            # Check if model file exists before attempting to load
            if USE_TRAINED_AGENT and os.path.exists("ppo_firefighter.zip"):
                print("🔄 Found existing model, loading for continued training...")
                model = PPO.load("ppo_firefighter")
                model.set_env(vec_env)
                print("✅ Model loaded successfully!")
            else:
                # No existing model, create a new one
                print("🆕 No existing model found, initializing new model...")
                model = PPO(
                    FireEnvLSTMPolicy,
                    vec_env,
                    n_steps=128,        # Collect exactly 10 steps before updating
                    batch_size=32,     # Use all collected steps in a single update
                    n_epochs=4,        # Perform only 1 optimization epoch per update
                    learning_rate=0.0003,
                    clip_range=0.2,
                    gamma=0.99,        # Discount factor
                    gae_lambda=0.95,   # GAE parameter
                    ent_coef=0.05,     # Entropy coefficient
                    vf_coef=0.5,       # Value function coefficient
                    max_grad_norm=0.5, # Maximum gradient norm
                    target_kl=0.01,    # Target KL divergence
                    verbose=1,
                    device='mps',
                    tensorboard_log=logdir
                )
                print("✅ New model initialized successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🆕 Initializing new model instead...")
            model = PPO(
                FireEnvLSTMPolicy,
                vec_env,
                n_steps=128,
                batch_size=32,
                n_epochs=4,
                learning_rate=0.0003,
                clip_range=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.05,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.01,
                device='mps',
                verbose=1,
                tensorboard_log=logdir
            )
            print("✅ New model initialized successfully!")
        vec_env.model = model

        # Create the custom reward logging callback
        reward_callback = RewardLoggingCallback()
        lr_callback = LearningRateScheduleCallback(lr_schedule)

        # Checkpoint callback to save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,  # Save every 5000 steps
            save_path="./models/",
            name_prefix="ppo_firefighter",
            verbose=1
        )

        # Combine callbacks
        callbacks = CallbackList([reward_callback,lr_callback])
        os.makedirs(logdir, exist_ok=True)

        # Train model
        print("🚀 Starting training...")
        model.learn(
            total_timesteps=200000,
            reset_num_timesteps=False,  # This ensures continued training
            tb_log_name="run4",   # can be anything
            callback=callbacks  # Use our callback list

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
                model.save("ppo_firefighter")
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
            server_thread.join(timeout=500)
            if server_thread.is_alive():
                print("⚠️ WebSocket server thread did not terminate properly")
            else:
                print("✅ WebSocket server stopped.")
        
        print("👋 Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()