# simulation_fixed.py - Running a trained PPO firefighter agent with enhanced analytics
# Fixed JSON serialization issue with NumPy types

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
import gc
import torch
from collections import deque

print('🔥 Starting SIMULATION backend')

# Ensure the model path exists
# model_path = os.environ.get("MODEL_DIR", "/models")
model_path = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))

MODEL_FILE = os.path.join(model_path, "ppo_firefighter.zip")

# Create output directory for analytics
ANALYTICS_DIR = os.environ.get("ANALYTICS_DIR", "fire_analytics")
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# Environment settings
MAX_TIMESTEPS = 2000
HELICOPTER_SPEED = 2
NUM_EPISODES = 5  # Number of episodes to run

# Simulation settings
USE_DETERMINISTIC = False  # Set to False to allow exploration (stochastic policy)
FORCE_HELITACK_PROB = 0.05  # Force helitack with 5% probability
SIMULATION_SPEED = 0.1  # Delay between steps (seconds)
NORMALIZE_OBSERVATIONS = True  # Match training normalization

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def get_optimal_device():
    """
    Automatically selects the best available device for PyTorch
    """
    import torch
    import platform
    import os
    
    # Check for CUDA availability (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = 'cuda'
        is_gpu = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"🚀 Using CUDA device: {gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''} available)")
        
    # Check for MPS availability (Apple Silicon)
    elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        is_gpu = True
        try:
            import subprocess
            system_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
            print(f"🚀 Using MPS device for Apple Silicon: {system_info}")
        except:
            print(f"🚀 Using MPS device for Apple Silicon")
            
        # Enable MPS fallbacks for better compatibility
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
    # Fallback to CPU
    else:
        device = 'cpu'
        is_gpu = False
        cpu_count = os.cpu_count() or 1
        print(f"⚠️ No GPU detected, using CPU with {cpu_count} logical cores")
        
    return device, is_gpu

# Memory management function
def clear_gpu_memory():
    """Clear GPU memory to prevent memory leaks"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon
        torch.mps.empty_cache()

# Set thread limits to avoid oversubscription
torch.set_num_threads(4)  # Limit CPU threads used by PyTorch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # Limit GPU memory usage

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Import custom policy (required for model loading)
try:
    from FeatureExtractor import FireEnvLSTMPolicy
except ImportError:
    print("⚠️ Could not import FireEnvLSTMPolicy, attempting to register it dynamically")
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    import torch.nn as nn
    import gym
    
    # Recreate the custom policy - must match the training policy architecture
    class LSTMFeatureExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=128):
            super().__init__(observation_space, features_dim)
            
            # CNN for processing fire grid
            self.cnn = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # MLP for processing helicopter coordinates
            self.heli_mlp = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU()
            )
            
            # Determine flattened CNN output size
            with torch.no_grad():
                # Assuming 4x160x240 input (4 stacked frames)
                dummy_input = torch.zeros(1, 4, 160, 240)
                cnn_output_size = self.cnn(dummy_input).shape[1]
            
            # LSTM for temporal processing
            self.lstm = nn.LSTM(
                input_size=cnn_output_size + 16 + 1,  # CNN output + heli_mlp output + on_fire
                hidden_size=128,
                batch_first=True
            )
            
            # Hidden state for LSTM
            self.hidden = None
            
            # Final projection layer
            self.fc = nn.Linear(128, features_dim)
            
        def forward(self, observations):
            # Process cells through CNN
            cells = observations['cells'].float() / 8.0  # Normalize
            cnn_output = self.cnn(cells)
            
            # Process helicopter coordinates
            heli_coord = observations['helicopter_coord'].float() / torch.tensor([240.0, 160.0])  # Normalize
            heli_features = self.heli_mlp(heli_coord)
            
            # Process on_fire flag
            on_fire = observations['on_fire'].float().unsqueeze(1)
            
            # Concatenate all features
            combined = torch.cat([cnn_output, heli_features, on_fire], dim=1)
            
            # Reshape for LSTM (batch_size, sequence_length=1, features)
            combined = combined.unsqueeze(1)
            
            # Initialize hidden state if needed
            if self.hidden is None or self.hidden[0].shape[1] != combined.shape[0]:
                self.hidden = (
                    torch.zeros(1, combined.shape[0], 128).to(combined.device),
                    torch.zeros(1, combined.shape[0], 128).to(combined.device)
                )
            
            # Process through LSTM
            lstm_out, self.hidden = self.lstm(combined, self.hidden)
            
            # Final projection
            features = self.fc(lstm_out.squeeze(1))
            
            return features
    
    # Register the custom policy
    from stable_baselines3.common.policies import ActorCriticPolicy
    
    class FireEnvLSTMPolicy(ActorCriticPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                features_extractor_class=LSTMFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
                **kwargs
            )

# Fire state constants
class FireState:
    Unburnt = 0
    Burning = 1
    Burnt = 2

class BurnIndex:
    Low = 0
    Medium = 1
    High = 2

# Global variables for WebSocket connection
client_websocket = None
message_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()
model = None

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\n⚠️ Received termination signal. Cleaning up...")
    stop_event.set()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
                try:
                    message_queue.put(message, block=False)
                except queue.Full:
                    # If queue is full, remove oldest message and add new one
                    try:
                        message_queue.get_nowait()
                        message_queue.put(message)
                    except:
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

# Metrics tracking class
class MetricsTracker:
    def __init__(self):
        self.helitack_actions = []  # Format: [(step, x, y)]
        self.fire_spread = []       # Format: [(step, [(x1, y1), (x2, y2), ...])
        self.burnt_area_over_time = []  # Format: [(step, total_burnt)]
        self.burning_cells_over_time = []  # Format: [(step, total_burning)]
        self.episode_statistics = {}
    
    def record_helitack(self, step, x, y):
        """Record a helitack action"""
        # Convert any numpy types to Python native types
        self.helitack_actions.append((int(step), int(x), int(y)))
    
    def record_fire_spread(self, step, new_burning_cells):
        """Record newly burning cells"""
        # Convert any numpy types to Python native types
        native_new_burning = [(int(x), int(y)) for x, y in new_burning_cells]
        self.fire_spread.append((int(step), native_new_burning))
    
    def record_burnt_area(self, step, total_burnt):
        """Record total burnt area"""
        # Convert any numpy types to Python native types
        self.burnt_area_over_time.append((int(step), int(total_burnt)))
    
    def record_burning_cells(self, step, total_burning):
        """Record total burning cells"""
        # Convert any numpy types to Python native types
        self.burning_cells_over_time.append((int(step), int(total_burning)))
    
    def save_metrics(self, filename="fire_metrics.json"):
        """
        Write out metrics to disk with proper handling of NumPy types
        """
        # No need to manually convert - we'll use the NumpyEncoder

        data = {
            "helitack_actions": self.helitack_actions,
            "fire_spread": self.fire_spread,
            "burnt_area_over_time": self.burnt_area_over_time,
            "burning_cells_over_time": self.burning_cells_over_time,
            "episode_statistics": self.episode_statistics
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
            
        print(f"✅ Metrics saved to {filename}")
        return data
    
    def calculate_episode_statistics(self):
        """Calculate summary statistics for the episode"""
        # Calculate statistics
        total_helitacks = len(self.helitack_actions)
        final_burnt_area = self.burnt_area_over_time[-1][1] if self.burnt_area_over_time else 0
        max_burning = max([b[1] for b in self.burning_cells_over_time]) if self.burning_cells_over_time else 0
        
        # Calculate helitack effectiveness
        effective_helitacks = 0
        # Implementation depends on how you define effectiveness
        
        self.episode_statistics = {
            "total_helitacks": total_helitacks,
            "final_burnt_area": final_burnt_area,
            "max_burning_cells": max_burning,
            "effective_helitacks": effective_helitacks,
            "containment_time": self.calculate_containment_time()
        }
        
        return self.episode_statistics
        
    def calculate_containment_time(self):
        """Calculate when fire was effectively contained"""
        if not self.burning_cells_over_time:
            return None
            
        max_burning = 0
        max_step = 0
        
        for step, burning in self.burning_cells_over_time:
            if burning > max_burning:
                max_burning = burning
                max_step = step
                
        return max_step

# Synchronous environment implementation for simulation
class FireEnvSync(gym.Env):
    def __init__(self, lat=None, lon=None, date=None):
        super().__init__()
        global client_websocket, message_queue
        
        # Store references to global communication channels
        self.websocket = None
        self.msg_queue = message_queue
        
        # Initialize spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0, high=8, shape=(4, 160, 240), dtype=np.int32),
            'on_fire': spaces.Discrete(2)
        })
        self.lat = lat
        self.lon = lon
        self.date = date
        
        # Initialize state
        self.step_count = 0
        self.episode_count = 0
        self.state = self._default_state()
        
        # Frame stacking - optimized with numpy array
        self.frame_history = np.zeros((4, 160, 240), dtype=np.int32)
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        
        # Cached observation for reuse
        self.cached_obs = None
        
        # Running stats for observation normalization (if needed)
        self.normalize_obs = NORMALIZE_OBSERVATIONS
        if self.normalize_obs:
            # Mean and std for helicopter coordinates (estimated from training)
            self.heli_mean = np.array([120.0, 80.0])  # Midpoints of grid
            self.heli_std = np.array([70.0, 40.0])    # Rough estimates of std
    
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
    
    def _default_response(self):
        """Return a default response when websocket is unavailable"""
        return {
            'helicopter_coord': [70, 115],
            'cells': np.zeros((160, 240), dtype=np.int32).tolist(),
            'on_fire': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenchedCells': 0
        }
    
    def _send_and_wait(self, message, timeout=5.0):
        """Send a message to React and wait for a response"""
        global client_websocket
        
        # Early return if no active connection
        if client_websocket is None:
            print("⚠️ No active WebSocket connection")
            return self._default_response()
        
        # Get the current websocket
        self.websocket = client_websocket
        
        # Clear any old messages
        try:
            while True:
                self.msg_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Send message
        try:
            asyncio.run(self._send_message(message))
        except Exception as e:
            print(f"❌ Error sending message: {e}")
            return self._default_response()
        
        # Wait for a response with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to get a response (with a short timeout)
                response = self.msg_queue.get(timeout=0.5)
                
                # Parse the response
                try:
                    data = json.loads(response)
                    return data
                except Exception as e:
                    print(f"❌ Error parsing response: {e}")
                    return self._default_response()
            except queue.Empty:
                # No response yet, check if WebSocket is still connected
                if client_websocket != self.websocket:
                    print("⚠️ WebSocket connection changed")
                    return self._default_response()
        
        # Timeout reached
        print(f"⚠️ Timeout waiting for response after {timeout}s")
        return self._default_response()
    
    async def _send_message(self, message):
        """Send a message to the WebSocket"""
        if self.websocket:
            try:
                await self.websocket.send(message)
            except Exception as e:
                print(f"❌ Error sending message: {e}")
                raise
    
    def update_frame_history(self, new_frame):
        """Update frame history with new frame"""
        self.frame_history = np.roll(self.frame_history, -1, axis=0)
        self.frame_history[3] = new_frame
    
    def normalize_helicopter_coord(self, coord):
        """Normalize helicopter coordinates to match training"""
        if self.normalize_obs:
            return (coord - self.heli_mean) / self.heli_std
        return coord
    
    def preprocess_observation(self, obs):
        """Preprocess observation to match expected format"""
        obs['helicopter_coord'] = obs['helicopter_coord'].astype(np.int32)
        obs['cells'] = obs['cells'].astype(np.int32)
        obs['on_fire'] = np.array(obs['on_fire'], dtype=np.int32)
        return obs
    
    def reset(self, *, seed=None, options=None):
        print(f"🔄 Resetting environment - Episode {self.episode_count + 1}")
        if seed is not None:
            super().reset(seed=seed)
            
        # Reset LSTM hidden states if model has them
        if hasattr(self, 'model') and hasattr(self.model.policy, 'features_extractor'):
            self.model.policy.features_extractor.hidden = None
            
        # Reset metrics tracker
        self.metrics = MetricsTracker()
        
        self.step_count = 0
        self.episode_count += 1
        
        # Create reset message
        reset_message = json.dumps({"action": "reset", "episode": self.episode_count})
        
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
        initial_cells = np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8)
        self.frame_history.fill(0)
        for i in range(4):
            self.frame_history[i] = initial_cells.copy()
        
        # Create observation with stacked frames
        final_observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': self.frame_history.copy(),
            'on_fire': int(self.state['on_fire'])
        }
        
        # Preprocess observation
        final_observation = self.preprocess_observation(final_observation)
        
        # Cache observation
        self.cached_obs = final_observation
        
        return final_observation, {}
    
    def step(self, action):
        self.step_count += 1
        self.state['last_action'] = action

        if self.step_count % 10 == 0 or action == 4:
            print(f"[Step {self.step_count}] Action: {action}")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0: heli_y += HELICOPTER_SPEED
        elif action == 1: heli_y -= HELICOPTER_SPEED  
        elif action == 2: heli_x -= HELICOPTER_SPEED 
        elif action == 3: heli_x += HELICOPTER_SPEED  
        elif action == 4: 
            print(f"🚁 Helitack performed at ({heli_x}, {heli_y})")
            # Record helitack action for metrics
            self.metrics.record_helitack(self.step_count, heli_x, heli_y)
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        # Create step message
        step_message = json.dumps({
            "action": str(action),
            "helicopter_coord": [heli_x, heli_y]
        })
        
        # Send and wait for response
        response = self._send_and_wait(step_message)
        if response is None:
            print("⚠️ No response from React, maintaining current state")
            response = {}

        # Update state
        self.state = {
            **self.state,
            'prevBurntCells': self.state['cellsBurnt'],
            **response,
            'helicopter_coord': [heli_x, heli_y]
        }
        
        # Calculate reward (for logging, not used in simulation)
        reward = self.calculate_reward(
            self.state.get('prevBurntCells', 0),
            self.state.get('cellsBurnt', 0),
            self.state.get('cellsBurning', 0),
            self.state.get('quenchedCells', 0)
        )
        
        # Check if episode is done
        done = False
        if self.state.get('cellsBurning', 1) == 0 or self.step_count >= MAX_TIMESTEPS:
            done = True
            # Calculate episode statistics
            self.metrics.calculate_episode_statistics()
            # Save metrics
            metrics_file = os.path.join(ANALYTICS_DIR, f"metrics_episode_{self.episode_count}.json")
            self.metrics.save_metrics(metrics_file)
            
            print(f"✅ Episode {self.episode_count} completed in {self.step_count} steps")
            print(f"   Cells burnt: {self.state.get('cellsBurnt', 0)}")
            print(f"   Cells quenched: {self.state.get('quenchedCells', 0)}")
            print(f"   Helitack actions: {len(self.metrics.helitack_actions)}")
            
        # Track metrics
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burnt_cells = np.sum(fire_states == FireState.Burnt)
        burning_cells = np.sum(fire_states == FireState.Burning)
        
        # Record metrics
        self.metrics.record_burnt_area(self.step_count, burnt_cells)
        self.metrics.record_burning_cells(self.step_count, burning_cells)
        
        # Track fire spread (only every 5 steps to save computation)
        if self.step_count % 5 == 0 and hasattr(self, 'previous_fire_states'):
            new_burning = np.logical_and(
                fire_states == FireState.Burning,
                self.previous_fire_states != FireState.Burning
            )
            new_burning_coords = list(zip(*np.where(new_burning)))
            self.metrics.record_fire_spread(self.step_count, new_burning_coords)
        
        # Store current fire state for next comparison
        self.previous_fire_states = fire_states.copy()
        
        # Update frame history
        current_cells = np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8)
        self.update_frame_history(current_cells)
        
        # Create observation with stacked frames
        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': self.frame_history.copy(),
            'on_fire': int(self.state.get('on_fire', 0))
        }
        
        # Preprocess observation
        observation = self.preprocess_observation(observation)
        
        # Cache observation
        self.cached_obs = observation
        
        return observation, reward, done, False, {}
    
    def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        """Calculate reward (just for logging, not used in simulation)"""
        reward = 0
        
        # Track fire progress
        if not hasattr(self, 'prev_burning'):
            self.prev_burning = curr_burning
        newly_burnt = curr_burnt - prev_burnt
        burning_reduction = self.prev_burning - curr_burning
        
        # Core objectives with clear signals
        reward += extinguished_by_helitack * 10
        reward -= newly_burnt * 5
        reward -= curr_burning * 0.1
        reward -= 0.1
        
        # Update for next timestep
        self.prev_burning = curr_burning
        
        return reward
    
    def close(self):
        print("Closing environment...")
        
        try:
            # Send close message
            close_message = json.dumps({"action": "close"})
            asyncio.run(self._send_message(close_message))
        except Exception as e:
            print(f"❌ Error closing environment: {e}")
        
        # Clean up resources
        self.metrics = None
        self.state = None
        self.frame_history = None
        self.cached_obs = None
        clear_gpu_memory()

# Function to modify agent action with a chance to force helitack
def get_modified_action(model, obs, env, deterministic=True):
    """Get action from model with potential modification to ensure helitack use"""
    # Get the raw action from the model (with exploration if not deterministic)
    action, _ = model.predict(obs, deterministic=deterministic)
    
    # Maybe force a helitack action
    if np.random.random() < FORCE_HELITACK_PROB:
        # Check if there are burning cells near the helicopter
        heli_x, heli_y = env.state['helicopter_coord']
        cells = np.array(env.state['cells'])
        fire_states = cells // 3
        
        # Look for burning cells in vicinity
        y_min, y_max = max(0, heli_y-5), min(160, heli_y+6)
        x_min, x_max = max(0, heli_x-5), min(240, heli_x+6)
        
        nearby_burning = np.any(fire_states[y_min:y_max, x_min:x_max] == FireState.Burning)
        
        if nearby_burning:
            # Force helitack action (4)
            print("🔥 Forcing helitack action near burning cells")
            return 4
    
    return action

# Generate a report-ready JSON from the metrics
def generate_report_data(metrics_file, output_file="fire_report_data.json"):
    """Convert metrics into a report-ready format"""
    try:
        # Load metrics data
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            
        # Extract key information
        report_data = {
            "fire_summary": {
                "total_burnt_area": data["episode_statistics"]["final_burnt_area"],
                "max_burning_cells": data["episode_statistics"]["max_burning_cells"],
                "containment_time": data["episode_statistics"]["containment_time"]
            },
            "helitack_operations": {
                "total_helitacks": len(data["helitack_actions"]),
                "helitack_coordinates": data["helitack_actions"]
            },
            "fire_progression": {
                "burnt_area_over_time": data["burnt_area_over_time"],
                "burning_cells_over_time": data["burning_cells_over_time"]
            }
        }
        
        # Save the report data
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, cls=NumpyEncoder)
            
        print(f"✅ Report data saved to {output_file}")
        return report_data
    except Exception as e:
        print(f"❌ Error generating report data: {e}")
        traceback.print_exc()
        return None

# Main function to run simulation
def main():
    global model
    
    lat = os.environ.get("LAT")
    lon = os.environ.get("LON")
    date = os.environ.get("DATE")

    print(f"🌎 Running simulation at lat={lat}, lon={lon}, date={date}")
    env = None
    server_thread = None
    
    try:
        # Start WebSocket server
        print("Starting WebSocket server...")
        server_thread = threading.Thread(target=websocket_server_thread, daemon=True)
        server_thread.start()
        
        # Wait for WebSocket server to start
        print("Waiting for WebSocket server to start...")
        time.sleep(2)
        
        # Wait for React to connect (with a longer timeout for simulation)
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
        env = FireEnvSync(lat=lat, lon=lon, date=date)
        
        # Check if model file exists
        if not os.path.isfile(MODEL_FILE):
            print(f"❌ Model file not found at {MODEL_FILE}")
            print("Please ensure the model exists before running simulation")
            return
        
        # Configure device
        device, is_gpu = get_optimal_device()
        
        # Load trained model
        print(f"🔍 Loading model from {MODEL_FILE}...")
        try:
            model = PPO.load(MODEL_FILE, device=device)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            return
        
        # List to track report files
        report_files = []
        
        # Run simulation for specified number of episodes
        print(f"🎮 Starting simulation for {NUM_EPISODES} episodes")
        
        for episode in range(NUM_EPISODES):
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"\n🔥 Starting Episode {episode+1}/{NUM_EPISODES}")
            
            # Run episode
            while not done:
                # Get action from model - using modified function that may force helitack
                action = get_modified_action(model, obs, env, deterministic=USE_DETERMINISTIC)
                
                # Take action in environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                # Optional: Slow down simulation for visualization
                time.sleep(SIMULATION_SPEED)
                
                # End early if max steps reached or done signal received
                if truncated or done or step >= MAX_TIMESTEPS:
                    break
            
            print(f"✅ Episode {episode+1} completed")
            print(f"   Total steps: {step}")
            print(f"   Episode reward: {episode_reward:.2f}")
            print(f"   Cells burnt: {env.state.get('cellsBurnt', 0)}")
            print(f"   Cells quenched: {env.state.get('quenchedCells', 0)}")
            print(f"   Helitack actions: {len(env.metrics.helitack_actions)}")
            
            # Generate report data for this episode
            metrics_file = os.path.join(ANALYTICS_DIR, f"metrics_episode_{episode+1}.json")
            report_file = os.path.join(ANALYTICS_DIR, f"report_episode_{episode+1}.json")
            generate_report_data(metrics_file, report_file)
            report_files.append(report_file)
            
            # Pause between episodes
            if episode < NUM_EPISODES - 1:
                print(f"Pausing between episodes... (3 seconds)")
                time.sleep(3)
        
        # Generate combined report from all episodes
        if report_files:
            print("\n📊 Generating combined fire assessment report...")
            
            # Read and combine all metrics
            combined_data = {
                "episodes": []
            }
            
            for i, file in enumerate(report_files):
                with open(file, 'r') as f:
                    episode_data = json.load(f)
                    combined_data["episodes"].append({
                        "episode_number": i + 1,
                        "summary": episode_data["fire_summary"],
                        "helitack_operations": episode_data["helitack_operations"]["total_helitacks"],
                        "helitack_coordinates": episode_data["helitack_operations"]["helitack_coordinates"],
                        "final_burnt_area": episode_data["fire_summary"]["total_burnt_area"]
                    })
            
            # Calculate averages and other statistics
            combined_data["summary"] = {
                "avg_burnt_area": sum(ep["summary"]["total_burnt_area"] for ep in combined_data["episodes"]) / len(combined_data["episodes"]),
                "avg_helitack_operations": sum(ep["helitack_operations"] for ep in combined_data["episodes"]) / len(combined_data["episodes"]),
                "total_episodes": len(combined_data["episodes"])
            }
            
            # Save combined report
            combined_file = os.path.join(ANALYTICS_DIR, "combined_fire_assessment.json")
            with open(combined_file, 'w') as f:
                json.dump(combined_data, f, indent=2, cls=NumpyEncoder)
                
            print(f"✅ Combined assessment saved to {combined_file}")
            
        print("\n🏆 Simulation complete!")
        print(f"Ran {NUM_EPISODES} episodes successfully")
        print(f"All metrics and reports saved to: {ANALYTICS_DIR}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        traceback.print_exc()
    finally:
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
            print("✅ WebSocket server stopped.")
        
        print("👋 Simulation complete. Exiting.")

if __name__ == "__main__":
    main()