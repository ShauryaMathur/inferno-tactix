# train.py - Optimized for performance and memory efficiency

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import signal
import sys
import traceback
import os
import csv
from datetime import datetime
import gc
import torch
import torch.cuda.amp as amp  # For mixed precision
from collections import deque
import environment.helper as helper
import config
from environment.cell import Cell
from fire_engine.fire_engine import FireEngine
from environment.vector import Vector2
from environment.wind import Wind
from environment.zone import Zone
from environment.enums import BurnIndex, FireState
import torch
import platform
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
# from gym.wrappers import Monitor
from gymnasium.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.utils import linear_schedule


from FeatureExtractor import FireEnvLSTMPolicy,FireEnvLSTMCNN
from stable_baselines3.common.utils import get_schedule_fn

print('HI! FROM NEW TRAIN BACKEND')

# Directory for logs
logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)
logFileName = 'run2'
# Environment settings
MAX_TIMESTEPS = 1000
HELICOPTER_SPEED = 3

USE_TRAINED_AGENT = True

# Ensure the save path exists
save_path = os.environ.get("MODEL_DIR", ".")
SAVED_AGENT_NAME = "ppo_firefighter_improved"
MODEL_FILE = os.path.join(save_path, SAVED_AGENT_NAME + ".zip")

ENVID_VS_CELLS = {}

def getCellsDataByEnvId(env_id,zones):
    env_id = 0
    if env_id not in ENVID_VS_CELLS:
        ENVID_VS_CELLS[env_id] = helper.populateCellsData(env_id,zones)
        print("Cell data populated for env_id:", env_id)
    return copy.deepcopy(ENVID_VS_CELLS[env_id])

def linear_schedule(initial_value):
    """
    Returns a function that computes a linearly decreasing schedule.
    Useful for learning rate or clip range.
    """
    def schedule(progress):
        return progress * 0.0 + initial_value * (1 - progress)
    return schedule

def get_optimal_device():
    """
    Automatically selects the best available device for PyTorch:
    - CUDA for NVIDIA GPUs
    - MPS for Apple Silicon (M1/M2/M3)
    - CPU as fallback
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
        bool: Whether the device is a GPU (CUDA or MPS)
    """
    
    # Check for CUDA availability (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = 'cuda'
        is_gpu = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"🚀 Using CUDA device: {gpu_name} ({gpu_count} device{'s' if gpu_count > 1 else ''} available)")
        
        # Set memory fraction if needed
        if os.environ.get('GPU_MEMORY_FRACTION'):
            fraction = float(os.environ.get('GPU_MEMORY_FRACTION', 0.9))
            torch.cuda.set_per_process_memory_fraction(fraction)
            print(f"💾 Limited GPU memory usage to {fraction*100:.0f}% of available memory")
            
    # Check for MPS availability (Apple Silicon)
    elif platform.system() == 'Darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        is_gpu = True
        # Try to get system info on macOS
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
        
        # Limit CPU threads to avoid oversubscription
        torch.set_num_threads(min(4, cpu_count))
        print(f"💾 Limited PyTorch to {min(4, cpu_count)} CPU threads")
    
    # Return both the device name and whether it's a GPU
    return device, is_gpu

# Set PyTorch thread limits to avoid oversubscription
torch.set_num_threads(4)  # Limit CPU threads used by PyTorch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)  # Limit GPU memory usage

# Memory management function
def clear_gpu_memory():
    """Clear GPU memory to prevent memory leaks"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon
        torch.mps.empty_cache()

# Learning rate scheduler callback
class LearningRateScheduleCallback(BaseCallback):
    def __init__(self, lr_schedule, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.lr_schedule = lr_schedule
        self.total_timesteps = total_timesteps
        
    def _on_step(self):
        progress = self.num_timesteps / self.total_timesteps
        new_lr = self.lr_schedule(progress)
        
        # Apply the new learning rate to the optimizer
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"Timestep {self.num_timesteps}/{self.total_timesteps}: Learning rate = {new_lr}")
        
        return True
# Define the schedule function
def lr_schedule(progress):
    return 0.0003 * (1.0 - progress)

# Reward logging callback
class RewardLoggingCallback(BaseCallback):
    """Custom callback for logging episode rewards to TensorBoard"""
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

# Memory cleanup callback
class MemoryCleanupCallback(BaseCallback):
    def __init__(self, cleanup_freq=1000, verbose=0):
        super().__init__(verbose)
        self.cleanup_freq = cleanup_freq
        self.episodes_seen = 0
        
    def _on_step(self):
        # Regular cleanup based on steps
        if self.n_calls % self.cleanup_freq == 0:
            clear_gpu_memory()
            if self.verbose > 0:
                print(f"Memory cleanup at step {self.n_calls}")
        
        # Episode-based cleanup
        dones = self.locals.get('dones', [False])
        if np.any(dones):
            # Increment counter for episode completions
            num_dones = np.sum(dones)
            self.episodes_seen += num_dones
            
            # Run garbage collection between episodes
            gc.collect()
            
            if self.verbose > 0 and num_dones > 0:
                print(f"Episode cleanup: {self.episodes_seen} episodes completed")
        
        return True

model = None  # Global model reference for signal handler
vec_env = None
# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\n⚠️ Received termination signal. Cleaning up...")
    if model is not None:
        try:
            print("💾 Saving model...")
            model.save(os.path.join(save_path, SAVED_AGENT_NAME))

            print(">>> MODEL_DIR:", os.environ.get("MODEL_DIR"))
            print(">>> cwd     :", os.getcwd())

            print("✅ Model saved as ppo_firefighter")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
        
        # Save VecNormalize statistics
        # try:
        #     if isinstance(vec_env, VecNormalize):
        #         vec_env.save("vecnormalize.pkl")
        #         print("✅ VecNormalize statistics saved.")
        # except Exception as e:
        #     print(f"❌ Error saving VecNormalize: {e}")
    
    # stop_event.set()
    sys.exit(0)

# # Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

class LSTMResetCallback(BaseCallback):
    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        extractor = self.model.policy.features_extractor
        
        done_indices = [i for i, done in enumerate(dones) if done]
        if done_indices and hasattr(extractor, "reset_hidden"):
            extractor.reset_hidden(env_indices=done_indices)
        
        return True
    
# Synchronous environment implementation with optimizations
class FireEnvSync(gym.Env):
    def __init__(self, env_id=0):
        super().__init__()
        
        # Initialize spaces
        self.env_id = env_id
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            # 'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0.0, high=1, shape=(5, 160, 240), dtype=np.float32),
            'on_fire': spaces.Discrete(2)
        })
        
        # Environment configuration
        self.cell_size = config.cellSize
        self.gridWidth = config.gridWidth
        self.gridHeight = config.gridHeight
        self.zones: list[Zone] = [Zone(**zone_dict) for zone_dict in config.ZONES]
        
        # Initialize state variables
        self._reset_state_variables()
        
        # Frame stacking
        self.frame_history = np.zeros((4, 160, 240), dtype=np.int8)
        
        # Cached observation
        self.cached_obs = None
        self._seed = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._seed = seed
        return [seed]
    
    def get_helicopter_position_map(self,x, y, height=160, width=240):
        map = np.zeros((height, width), dtype=np.float32)
        map[y, x] = 1.0
        return map
    def _reset_state_variables(self):
        """Reset all state variables to initial values"""
        self.step_count = 0
        self.time = 0
        self.prev_tick_time = None
        self.simulation_running = True
        self.cells = []
        self.engine = None
        
        # Reset state dict
        self.state = {
            'helicopter_coord': np.array([70, 30], dtype=np.int32),
            'cells': np.zeros((160, 240), dtype=np.int32),
            'on_fire': 0,
            'prevBurntCells': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenchedCells': 0,
            'prev_burning': 0,
            'last_action': None
        }
    
    def _default_state(self):
        return {
            'helicopter_coord': np.array([70, 30], dtype=np.int32),
            'cells': np.zeros((160, 240), dtype=np.int32),
            'on_fire': 0,
            'prevBurntCells': 0,
            'cellsBurnt': 0,
            'cellsBurning': 0,
            'quenchedCells': 0
        }
    
    def tick(self, time_step: float):
        """Update fire simulation by one time step"""
        if self.engine and self.simulation_running:
            self.time += time_step
            self.engine.update_fire(self.cells, self.time)
            
            if self.engine.fire_did_stop:
                self.simulation_running = False
                print(f"[Env {self.env_id}] Fire simulation stopped at time {self.time}")

    def _get_current_fire_stats(self):
            """Get current fire statistics from cells"""
            cells_burning = len([cell for cell in self.cells if cell.fireState == FireState.Burning])
            cells_burnt = len([cell for cell in self.cells if cell.fireState == FireState.Burnt])
            return cells_burning, cells_burnt
    
    def _update_simulation_state(self):
            """Update state dictionary from current simulation state"""
            # Generate binned cells from current simulation state
            binnedCells = helper.generate_fire_status_map_from_cells(
                self.cells, self.gridWidth, self.gridHeight
            )
            binnedCells = np.array(binnedCells, dtype=np.int32)
            # binnedCells = binnedCells.astype(np.float32)
            binnedCells = np.clip(binnedCells, -1, 8)
            binnedCells = (binnedCells + 1) / 9.0
            
            # print(binnedCells.shape)
            # Get current fire statistics
            cells_burning, cells_burnt = self._get_current_fire_stats()
            
            # Update state
            self.state.update({
                'cells': binnedCells,
                'cellsBurning': cells_burning,
                'cellsBurnt': cells_burnt,
                'prevBurntCells': self.state.get('cellsBurnt', 0),  # Store previous value
                'prev_burning': self.state.get('cellsBurning', 0)   # Store previous value
            })
    def step_simulation(self, current_time_ms: float):
        """Step the simulation forward by appropriate time"""
        if not self.simulation_running:
            return
            
        # Calculate time step
        if self.prev_tick_time is None:
            self.prev_tick_time = current_time_ms
            time_step = 1.0  # Default first step
        else:
            real_time_diff_minutes = (current_time_ms - self.prev_tick_time) / 60000
            self.prev_tick_time = current_time_ms
            
            ratio = 86400 / getattr(config, 'modelDayInSeconds', 8)
            optimal_time_step = ratio * 0.000277
            time_step = min(
                getattr(config, 'maxTimeStep', 180),
                optimal_time_step * 4,
                ratio * real_time_diff_minutes
            )
        
        self.tick(time_step)

    def _get_fallback_observation(self):
            """Get fallback observation in case of errors"""
            return {
                'helicopter_coord': np.array([70, 30], dtype=np.int32),
                'cells': np.zeros((4, 160, 240), dtype=np.int8),
                'on_fire': 0
            }
    def update_frame_history(self, new_frame):
        """Update frame history efficiently"""
        self.frame_history = np.roll(self.frame_history, -1, axis=0)
        self.frame_history[3] = new_frame
    
    def preprocess_observation(self, obs):
        """Preprocess observation for more efficient learning while avoiding normalizing huge arrays"""
        # Keep helicopter_coord as int32 to match observation space
        obs['helicopter_coord'] = obs['helicopter_coord'].astype(np.int32)
        
        # Ensure cells has the right dtype
        obs['cells'] = obs['cells'].astype(np.int32)
        
        # Ensure on_fire is correct format
        obs['on_fire'] = np.array(obs['on_fire'], dtype=np.int32)
        
        return obs
    
    def reset(self, *, seed=None, options=None):

        try:
            super().reset(seed=seed)
            if seed is not None:
                self.seed(seed)
            
            print(f"🧼 Resetting environment {self.env_id} with seed={self._seed}")
            # input("Resetting environment. Press Enter to continue...")
            
            # Reset all state variables
            self._reset_state_variables()
            self.episode_count = getattr(self, 'episode_count', 0) + 1
            
            # Initialize cells - ensure complete isolation
            self.cells = getCellsDataByEnvId(self.env_id, self.zones)
            
            # Initialize fire spark
            spark = Vector2(60000 - 1, 40000 - 1)
            grid_x = int(spark.x // self.cell_size)
            grid_y = int(spark.y // self.cell_size)
            spark_cell : Cell = self.cells[helper.get_grid_index_for_location(grid_x, grid_y, self.gridWidth)]
            spark_cell.ignitionTime = 0
            # spark_cell.fireState = FireState.Burning
            
            # Initialize fire engine
            self.engine = FireEngine(Wind(0.0, 0.0), config)
            
            # Update state from simulation
            self._update_simulation_state()
            
            # Initialize frame history
            initial_cells = np.clip(np.array(self.state['cells'], dtype=np.int8), -1, 8)
            initial_cells = (initial_cells + 1) / 9.0  # normalize to [0, 1]

            self.frame_history.fill(0)
            for i in range(4):
                self.frame_history[i] = initial_cells.copy()
            
            heli_x, heli_y = self.state['helicopter_coord']
            heli_map = np.zeros_like(initial_cells, dtype=np.float32)
            heli_map[heli_y, heli_x] = 1.0  # Note: (y, x) indexing

            spatial_obs = np.concatenate([self.frame_history.copy(), heli_map[None, :, :]], axis=0)

            observation = {
                'cells': spatial_obs,
                'on_fire': np.array([int(self.state['on_fire'])], dtype=np.float32)
            }
            assert observation["cells"].shape == (5, 160, 240), f"Bad shape: {observation['cells'].shape}"
            # Create observation
            # observation = {
            #     'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            #     'cells': self.frame_history.copy(),
            #     'on_fire': int(self.state['on_fire'])
            # }
            
            self.cached_obs = observation
            
            print(f"[Env {self.env_id}] Reset complete - Initial burning cells: {self.state['cellsBurning']}")
            
            return observation, {}
            
        except Exception as e:
            print(f"Error in reset for env {self.env_id}: {e}")
            traceback.print_exc()
            return self._get_fallback_observation(), {}
    
    def apply_action(self, action):
        """Apply action and return new helicopter position and quenched cells"""
        self.step_count += 1
        self.state['last_action'] = action
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0:   # Down
            heli_y += HELICOPTER_SPEED
        elif action == 1: # Up
            heli_y -= HELICOPTER_SPEED
        elif action == 2: # Left
            heli_x -= HELICOPTER_SPEED
        elif action == 3: # Right
            heli_x += HELICOPTER_SPEED
        elif action == 4: # Helitack
            print(f"[Env {self.env_id}] Helitack at ({heli_x}, {heli_y})")
        
        # Clip coordinates to valid range
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        
        # Perform helitack if action is 4
        quenched_cells = 0
        if action == 4:
            quenched_cells = helper.perform_helitack(self.cells, heli_x, heli_y)
            if quenched_cells > 0:
                print(f"[Env {self.env_id}] Helitack quenched {quenched_cells} cells")
        
        return heli_x, heli_y, quenched_cells
    
    def step(self, action):
        try:
            # Store previous state for reward calculation
            prev_burnt = self.state.get('cellsBurnt', 0)
            prev_burning = self.state.get('cellsBurning', 0)
            
            # Apply action and get helicopter position
            heli_x, heli_y, quenched_cells = self.apply_action(action)

            heli_map = self.get_helicopter_position_map(heli_x, heli_y)  # shape (160, 240)

            # Run simulation step
            current_time_ms = time.time() * 1000
            self.step_simulation(current_time_ms)
            
            # Update state from simulation
            self._update_simulation_state()
            
            # Update helicopter position and fire status
            self.state['helicopter_coord'] = np.array([heli_x, heli_y], dtype=np.int32)
            self.state['quenchedCells'] = quenched_cells
            # print("self.state['cells'].shape",self.state['cells'].shape)
            # Check if helicopter is on fire
            on_fire = helper.is_helicopter_on_fire(self.state['cells'], heli_x, heli_y)
            self.state['on_fire'] = on_fire
            
            # Calculate reward
            reward = self.calculate_reward_improved_scaled(
                prev_burnt,
                self.state['cellsBurnt'],
                self.state['cellsBurning'],
                quenched_cells
            )
            
            # Check if episode is done
            # done = (self.state['cellsBurning'] == 0) or (self.step_count >= MAX_TIMESTEPS)
            terminated = (self.state['cellsBurning'] == 0)
            truncated = (self.step_count >= MAX_TIMESTEPS)
            
            # Update frame history
            current_cells = np.clip(np.array(self.state['cells'], dtype=np.int8), -1, 8)
            self.update_frame_history(current_cells)

            spatial_obs = np.concatenate([self.frame_history.copy(), heli_map[None, :, :]], axis=0)

            # Create observation
            # observation = {
            #     'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            #     'cells': self.frame_history.copy(),
            #     'on_fire': int(self.state['on_fire'])
            # }

            observation = {
                'cells': spatial_obs,
                'on_fire': np.array([int(self.state['on_fire'])], dtype=np.float32)
            }
            # print("Obs cells shape:", observation['cells'].shape)
            assert observation["cells"].shape == (5, 160, 240), f"Bad shape: {observation['cells'].shape}"

            self.cached_obs = observation
            
            print(f"[Env {self.env_id}] Step {self.step_count}: Burning={self.state['cellsBurning']}, "
                      f"Burnt={self.state['cellsBurnt']}, Reward={reward:.3f}")
            # if self.step_count % 10 == 0:  # Log every 10 steps
            #     print(f"[Env {self.env_id}] Step {self.step_count}: Burning={self.state['cellsBurning']}, "
            #           f"Burnt={self.state['cellsBurnt']}, Reward={reward:.3f}")
            # print(f"Returning: terminated={terminated}, truncated={truncated}")

            return observation, reward, terminated, truncated, {}
            
        except Exception as e:
            print(f"Error in step for env {self.env_id}: {e}")
            traceback.print_exc()
            return self.cached_obs or self._get_fallback_observation(), 0, True, False, {}

    
    def calculate_reward_simple(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_agent):
        reward = 0

        # 1. Penalize increase in burnt area
        newly_burnt = curr_burnt - prev_burnt
        reward -= 0.1 * newly_burnt   # Strong penalty for damage

        # 2. Reward reduction in active burning cells
        if not hasattr(self.state, 'prev_burning'):
            self.state['prev_burning'] = curr_burning

        burning_reduction = self.state['prev_burning'] - curr_burning
        reward += 0.1 * burning_reduction  # Encourages fire containment

        # 3. Reward cells extinguished by agent
        reward += 1.0 * extinguished_by_agent  # Direct intervention bonus

        # 4. Optional: Small step penalty (to encourage faster control)
        reward -= 0.1

        # Save current for next step
        self.state['prev_burning'] = curr_burning
        return np.clip(reward, -10, 10)
    
    def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0
        if not hasattr(self.state, 'prev_burning'):
            self.state['prev_burning'] = curr_burning

        newly_burnt = curr_burnt - prev_burnt
        burning_reduction = self.state['prev_burning'] - curr_burning

        # Reward breakdown components
        proximity_reward = 0.0
        hit_reward = 0.0
        firebreak_reward = 0.0
        wasted_penalty = 0.0
        unburnable_penalty = 0.0
        reduction_reward = 0.0

        # ✅ Core reward logic (scaled down)
        reward += extinguished_by_helitack * 5     # Reduced from 10 → to soften sparse spikes
        reward -= newly_burnt * 1.0                # Reduced penalty for better shaping
        reward -= curr_burning * 0.05              # Reduced running penalty
        reward -= 0.01                             # Softer time penalty

        # ✅ Reward for reducing burning cells
        if burning_reduction > 0:
            reduction_reward = burning_reduction * 0.2
            reward += reduction_reward

        # Extract state info
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burning_mask = fire_states == FireState.Burning

        # ✅ Proximity reward
        if np.any(burning_mask):
            burning_coords = np.argwhere(burning_mask)
            distances = np.sqrt((burning_coords[:, 0] - heli_y) ** 2 + (burning_coords[:, 1] - heli_x) ** 2)
            min_distance = np.min(distances)
            proximity_reward = 20 / (1 + min_distance)   # Sharper early reward
            reward += proximity_reward

        # ✅ Helitack logic
        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]

            if cell_value == -1:
                unburnable_penalty = -5                 # Less harsh
                reward += unburnable_penalty
            else:
                fire_state = fire_states[heli_y, heli_x]
                burn_index = cell_value % 3

                if fire_state == FireState.Burning:
                    intensity_bonus = (burn_index + 1)
                    hit_reward = 10 * intensity_bonus   # Softer hit reward
                    reward += hit_reward
                elif fire_state == FireState.Burnt:
                    wasted_penalty = -2                 # Softer penalty
                    reward += wasted_penalty
                elif fire_state == FireState.Unburnt:
                    nearby_burning = np.sum(burning_mask[
                        max(0, heli_y - 5):min(160, heli_y + 6),
                        max(0, heli_x - 5):min(240, heli_x + 6)
                    ])
                    if nearby_burning > 0:
                        firebreak_reward = 2 * min(nearby_burning, 5)
                    else:
                        firebreak_reward = -2
                    reward += firebreak_reward

        self.state['prev_burning'] = curr_burning

        # ✅ FINAL SCALING
        scaled_reward = reward / 50.0
        # self.step_count += 1
        return scaled_reward
    
    def calculate_reward_scaled(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0
        if not hasattr(self.state, 'prev_burning'):
            self.state['prev_burning'] = curr_burning

        newly_burnt = curr_burnt - prev_burnt
        burning_reduction = self.state['prev_burning'] - curr_burning

        # Reward breakdown components
        proximity_reward = 0.0
        hit_reward = 0.0
        firebreak_reward = 0.0
        wasted_penalty = 0.0
        unburnable_penalty = 0.0
        reduction_reward = 0.0

        # 🔧 Scaled reward terms
        reward += extinguished_by_helitack * 1.0        # +1 per extinguish
        reward -= newly_burnt * 5.0                     # -0.1 per cell burnt after /50
        reward -= curr_burning * 0.25                   # mild ongoing fire penalty
        reward -= 5.0                                   # step penalty

        if burning_reduction > 0:
            reduction_reward = burning_reduction * 5.0  # +0.1 per cell reduction after /50
            reward += reduction_reward

        # 🔧 Proximity reward
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burning_mask = fire_states == FireState.Burning

        if np.any(burning_mask):
            burning_coords = np.argwhere(burning_mask)
            distances = np.sqrt((burning_coords[:, 0] - heli_y) ** 2 + (burning_coords[:, 1] - heli_x) ** 2)
            min_distance = np.min(distances)
            proximity_reward = 5.0 / (1 + min_distance)
            reward += proximity_reward

        # 🔧 Helitack logic
        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]

            if cell_value == -1:
                unburnable_penalty = -250
                reward += unburnable_penalty
            else:
                fire_state = fire_states[heli_y, heli_x]
                burn_index = cell_value % 3

                if fire_state == FireState.Burning:
                    intensity_bonus = (burn_index + 1)
                    hit_reward = 50 * intensity_bonus
                    reward += hit_reward
                elif fire_state == FireState.Burnt:
                    wasted_penalty = -100
                    reward += wasted_penalty
                elif fire_state == FireState.Unburnt:
                    nearby_burning = np.sum(burning_mask[
                        max(0, heli_y - 5):min(160, heli_y + 6),
                        max(0, heli_x - 5):min(240, heli_x + 6)
                    ])
                    if nearby_burning > 0:
                        firebreak_reward = 100 * min(nearby_burning, 5)
                    else:
                        firebreak_reward = -100
                    reward += firebreak_reward

        self.state['prev_burning'] = curr_burning

        scaled_reward = reward / 50.0
        return scaled_reward
    
    def calculate_reward_improved(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0.0

        # Penalize new burnt cells (damage)
        newly_burnt = curr_burnt - prev_burnt
        reward -= 0.05 * newly_burnt

        # Reward extinguishing burning cells
        reward += 0.5 * extinguished_by_helitack

        # Reward reduction in burning cells
        burning_reduction = self.state.get('prev_burning', curr_burning) - curr_burning
        reward += 0.2 * burning_reduction

        # Small penalty for ongoing burning cells
        reward -= 0.01 * curr_burning

        # Small step penalty to encourage efficiency
        reward -= 0.01

        # Gentle penalty for acting on unburnable/burnt cells
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3

        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]
            if cell_value == -1 or fire_states[heli_y, heli_x] == FireState.Burnt:
                reward -= 0.1  # Gentle penalty

        # Optional: Remove or reduce proximity reward
        # (If you keep it, make it small and only when not extinguishing)
        # if np.any(burning_mask):
        #     min_distance = ...
        #     reward += 0.01 / (1 + min_distance)

        # Update prev_burning
        self.state['prev_burning'] = curr_burning

        # Clip reward to reasonable range
        reward = np.clip(reward, -1, 1)
        return reward

    def calculate_reward_improved_scaled(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0.0

        # Penalize new burnt cells (damage) strongly
        newly_burnt = curr_burnt - prev_burnt
        reward -= 1.0 * newly_burnt  # Increased penalty for new burnt cells

        # Reward extinguishing burning cells — encourage actively putting out fires
        reward += 2.0 * extinguished_by_helitack  # Larger reward for extinguishing

        # Reward reduction in burning cells (i.e., burning areas shrinking)
        burning_reduction = self.state.get('prev_burning', curr_burning) - curr_burning
        reward += 1.0 * burning_reduction  # Reward reducing fire size

        # Small penalty for ongoing burning cells to discourage letting fires burn longer
        reward -= 0.05 * curr_burning

        # Small step penalty to encourage faster containment
        reward -= 0.01

        # Gentle penalty for acting on unburnable/burnt cells (to avoid wasting moves)
        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3

        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]
            if cell_value == -1 or fire_states[heli_y, heli_x] == FireState.Burnt:
                reward -= 0.2  # Slightly stronger penalty here

        # Update prev_burning for next step
        self.state['prev_burning'] = curr_burning

        # Clip reward to a reasonable range for PPO stability
        reward = np.clip(reward, -10, 10)
        return reward

    
    def close(self):
        """Clean up environment resources"""
        print(f"Closing environment {self.env_id}...")
        self.simulation_running = False
        self.cells = []
        self.engine = None
        self.state = None
        self.frame_history = None
        self.cached_obs = None
        clear_gpu_memory()

    
def make_env(rank=0, base_seed=1000):
    def _init():
        env = FireEnvSync(env_id=rank)
        env = Monitor(env,logdir)
        print(f"[ENV {rank}]")
        # check_env(env)
        # seed = base_seed + rank
        # env.seed(seed)  # Seed right here
        
        return env
    return _init

# Main function
def main():
    global model
    global vec_env

    # Create vectorized environment
    vec_env = SubprocVecEnv([make_env(i) for i in range(8)])

    # Restore VecNormalize if available
    if os.path.exists("vecnormalize_improved.pkl") and os.path.getsize("vecnormalize_improved.pkl") > 0:
        print("🔄 Restoring VecNormalize state...")
        vec_env = VecNormalize.load("vecnormalize_improved.pkl", vec_env)
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False, 
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )

    device, is_gpu = get_optimal_device()

    print("🔍 Checking for existing model...")
    if USE_TRAINED_AGENT and os.path.isfile(MODEL_FILE):
        print("🔄 Found existing model, loading for continued training...")
        model = PPO.load(MODEL_FILE, device=device, tensorboard_log=logdir)
        model.set_env(vec_env)
        model.lr_schedule = get_schedule_fn(3e-4)
        model.clip_range = get_schedule_fn(0.2)
        # model.lr_schedule = lambda _: 5e-5
        model.policy.optimizer.param_groups[0]['lr'] = 5e-5
        # Use a function (lambda) for clip_range
        # model.clip_range = lambda _: 0.3

        # These two must also be callables if you want dynamic behavior
        model.vf_coef =  0.5
        model.gamma = 0.995

        print("✅ Model loaded successfully!")
    else:
        print("🆕 No existing model found, initializing new model...")
        model = PPO(
            FireEnvLSTMPolicy,
            vec_env,
            n_steps=128,
            batch_size=64,
            n_epochs=3,
            learning_rate= 3e-4,
            clip_range= 0.1,
            gamma=0.95,
            gae_lambda=0.9,
            ent_coef=0.2,
            vf_coef=0.4,
            max_grad_norm=1.0,
            target_kl=0.03,
            verbose=1,
            device=device,
            tensorboard_log=logdir,
            policy_kwargs = dict(
                features_extractor_class=FireEnvLSTMCNN,
                features_extractor_kwargs=dict(features_dim=512)
            )
        )
        print("✅ New model initialized successfully!")
        
    if device == 'mps':
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available, falling back to CPU")
            device = 'cpu'
        else:
            print("Configuring MPS-specific optimizations for Apple Silicon")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            model.batch_size = 64
            model.n_steps = 128
            print(f"MPS-optimized batch configuration: batch_size={model.batch_size}, n_steps={model.n_steps}")

    reward_callback = RewardLoggingCallback()
    # lr_callback = LearningRateScheduleCallback(lr_schedule)
    memory_callback = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
    # lr_callback = LearningRateScheduleCallback(lr_schedule, total_timesteps=1_000_000, verbose=1)

    callbacks = CallbackList([
        reward_callback, 
        LSTMResetCallback(),
        memory_callback
    ])

    os.makedirs(logdir, exist_ok=True)
    clear_gpu_memory()

    if device != 'cpu':
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster training")
        torch.backends.cudnn.benchmark = True

    print("🚀 Starting training...")
    try:
        model.learn(
            total_timesteps=100000,
            reset_num_timesteps=False,
            tb_log_name=logFileName,
            callback=callbacks
        )
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        traceback.print_exc()

    finally:
        if model is not None:
            try:
                print("💾 Saving model...")
                model.save(os.path.join(save_path, SAVED_AGENT_NAME))
                vec_env.save("vecnormalize_improved.pkl")
                print("✅ Model and normalization state saved.")
            except Exception as e:
                print(f"❌ Error saving model: {e}")

        print("👋 Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()