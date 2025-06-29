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

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.utils import linear_schedule


from FeatureExtractor import FireEnvLSTMPolicy,FireEnvLSTMCNN

print('HI! FROM NEW TRAIN BACKEND')

# Directory for logs
logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)

# Environment settings
MAX_TIMESTEPS = 2000
HELICOPTER_SPEED = 3

USE_TRAINED_AGENT = False

# Ensure the save path exists
save_path = os.environ.get("MODEL_DIR", ".")
SAVED_AGENT_NAME = "ppo_firefighter_new"
MODEL_FILE = os.path.join(save_path, SAVED_AGENT_NAME + ".zip")

def linear_schedule(initial_value):
    """
    Returns a function that computes a linearly decreasing schedule.
    Useful for learning rate or clip range.
    """
    def schedule(progress):
        return progress * 0.0 + initial_value * (1 - progress)
    return schedule

def fix_env_reset(model):
    """Apply a minimal fix to handle the environment reset issue."""
    if hasattr(model, 'env'):
        # Store the original reset method
        original_reset = model.env.reset

        # Create a patched reset method
        def patched_reset(*args, **kwargs):
            """A reset method that works with both old and new gym versions"""
            try:
                result = original_reset(*args, **kwargs)
                # Just return the result as-is, without trying to access [0]
                return result
            except Exception as e:
                print(f"Warning in patched reset: {e}")
                # If all else fails, return a dummy observation
                import numpy as np
                return np.zeros((model.env.num_envs, *model.env.observation_space.shape))
        
        # Replace the reset method
        model.env.reset = patched_reset
        print("✓ Applied minimal environment reset fix")
    
    return model
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
    def __init__(self,env_id=0):
        super().__init__()
        
        # Initialize spaces
        self.env_id = env_id
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=-1, high=8, shape=(4, 160, 240), dtype=np.int8),  # 4 stacked frames
            'on_fire': spaces.Discrete(2)
        })
        
        # Initialize state
        self.step_count = 0
        self.episode_count = 0
        self.state = self._default_state()
        
        # Frame stacking - optimized with numpy array
        self.frame_history = np.zeros((4, 160, 240), dtype=np.int8)
        
        # Metrics tracking
        # self.metrics = MetricsTracker()
        
        # Cached observation for reuse
        self.cached_obs = None

        self.gridWidth =  config.gridWidth
        self.gridHeight = config.gridHeight
        self.zones: list[Zone] = [Zone(**zone_dict) for zone_dict in config.ZONES]
        self.cells: list[Cell] = []
        self.time = 0
        self.prev_tick_time = None
        self.spark : Vector2 = None
        self.engine : FireEngine = None
        self._seed = None  # Store for future use

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._seed = seed
        return [seed]
    def make_copy(self):
        return FireEnvSync()
    
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
    
    def tick(self, time_step: float):
        
        if self.engine:
            self.time += time_step
            self.engine.cells = self.cells
            self.engine.update_fire(self.time)

            if self.engine.fire_did_stop:
                self.simulation_running = False
                print("Final timestep:", time_step)
                print("Simulation time:", self.time)

    def step_simulation(self, current_time_ms: float):

        real_time_diff_minutes = None
        if self.prev_tick_time is None:
            self.prev_tick_time = current_time_ms
        else:
            real_time_diff_minutes = (current_time_ms - self.prev_tick_time) / 60000
            self.prev_tick_time = current_time_ms

        time_step = 1  # default fallback
        if real_time_diff_minutes:
            ratio = 86400 / getattr(config,'modelDayInSeconds',8)
            optimal_time_step = ratio * 0.000277
            time_step = min(
                getattr(config,'maxTimeStep',180),
                optimal_time_step * 4,
                ratio * real_time_diff_minutes
            )

        self.tick(time_step)

    def populateCellsData(self):
        self.cells = []
        zoneIndex = helper.get_land_cover_zone_index(config,"landcover_1200x813.png")
        elevation = helper.get_elevation_data(config,"heightmap_1200x813_2.png")
        nonBurnableZones = [12,14,16,17,18]
        for y in range(self.gridHeight):
            for x in range(self.gridWidth):
                index = helper.get_grid_index_for_location(x, y, self.gridWidth)
                zi = zoneIndex[index] if zoneIndex is not None else 0
                is_edge = (
                    config.fillTerrainEdges and
                    (x == 0 or x == self.gridWidth - 1 or y == 0 or y == self.gridHeight - 1)
                )

                cell_options = {
                    "x": x,
                    "y": y,
                    "zone": self.zones[zi],
                    "zoneIdx": zi,
                    "baseElevation": 0 if is_edge else (elevation[index] if elevation else None),
                    "isRiver": True if zi in nonBurnableZones else False
                }

                # self.totalCellCountByZone[zi] = self.totalCellCountByZone.get(zi, 0) + 1
                self.cells.append(Cell(**cell_options) )
        # print('Cells',self.cells)

    def update_frame_history(self, new_frame):
        """Update frame history efficiently using numpy operations"""
        # Roll the frames (shift all frames one position back)
        self.frame_history = np.roll(self.frame_history, -1, axis=0)
        # Add the new frame to the last position
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
            if seed is not None:
                self.seed(seed)
            print(f"🧼 Resetting environment with seed={self._seed}")
            
            # Reset LSTM hidden states
            # if hasattr(self, 'model') and hasattr(self.model.policy, 'features_extractor'):
            #     self.model.policy.features_extractor.hidden = None
                
            # Reset metrics tracker
            # self.metrics = MetricsTracker()
            
            self.step_count = 0
            self.episode_count += 1
            
            self.populateCellsData()

            # Create reset message
            # reset_message = json.dumps({"action": "reset", "episode": self.episode_count})

            self.spark = Vector2(60000 - 1, 40000 - 1)
            self.engine = FireEngine(self.cells,Wind(0.0,0.0),self.spark,config)
            # self.cells = self.engine.cells
            binnedCells = helper.generate_fire_status_map_from_cells(self.cells, self.gridWidth, self.gridHeight)

            self.state = self._default_state()
            self.state = {
                **self.state,
                'cells': binnedCells
            }
            # print(self.state)
            # Initialize frame history more efficiently
            initial_cells = np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8)
            self.frame_history.fill(0)  # Clear all frames
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
        except Exception as e:
            print(f"Error in reset: {e}")
            return self._default_state(), {}
    
    def apply_action(self,action):
        self.step_count += 1
        self.state['last_action'] = action

        print(f"\n[Step {self.step_count}] Action taken: {action}")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0: heli_y += HELICOPTER_SPEED
        elif action == 1: heli_y -= HELICOPTER_SPEED  
        elif action == 2: heli_x -= HELICOPTER_SPEED 
        elif action == 3: heli_x += HELICOPTER_SPEED  
        elif action == 4: 
            print(f"Helitack performed at ({heli_x}, {heli_y})")
            # Record helitack action for metrics
            # self.metrics.record_helitack(self.step_count, heli_x, heli_y)
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 0, 239))
        heli_y = int(np.clip(heli_y, 0, 159))
        quenched_cells = 0
        if action == 4:
            quenched_cells = helper.perform_helitack(self.cells, heli_x, heli_y)
        
        return heli_x, heli_y, quenched_cells
    def step(self, action):
        
        try:
            self.cells = self.engine.cells
            heli_x, heli_y, quenched_cells = self.apply_action(action)

            # Simulate one tick with current wall-clock time
            current_time_ms = time.time() * 1000  # time in ms
            self.step_simulation(current_time_ms)
            self.cells = self.engine.cells
            binnedCells = helper.generate_fire_status_map_from_cells(self.cells, self.gridWidth, self.gridHeight)
            cells_burning = len([cell for cell in self.cells if cell.fireState == FireState.Burning])
            cells_burnt = len([cell for cell in self.cells if cell.fireState == FireState.Burnt])
            on_fire = helper.is_helicopter_on_fire(binnedCells, heli_x, heli_y)

            # Create step message
            self.cells = self.engine.cells
            # Update state
            self.state = {
                **self.state,
                'prevBurntCells': self.state['cellsBurnt'],
                'cells': binnedCells,
                'cellsBurning': cells_burning,
                'cellsBurnt': cells_burnt,
                'quenchedCells': quenched_cells,
                'on_fire': on_fire,
                'helicopter_coord': [heli_x, heli_y]
            }
            
            # Calculate reward
            reward = self.calculate_reward(
                self.state.get('prevBurntCells', 0),
                self.state.get('cellsBurnt', 0),
                self.state.get('cellsBurning', 0),
                self.state.get('quenchedCells', 0)
            )
            
            # Check if episode is done
            done = False
            print(f"cellsBurning: {self.state.get('cellsBurning')}",'count',self.step_count)
            if self.state.get('cellsBurning', 1) == 0 or self.step_count >= MAX_TIMESTEPS:
                done = True
            
            # Track metrics
            cells = np.array(self.state['cells'])
            fire_states = cells // 3
            # burnt_cells = np.sum(fire_states == FireState.Burnt)
            # burning_cells = np.sum(fire_states == FireState.Burning)
            
            # Track fire spread (only every 5 steps to save computation)
            if self.step_count % 5 == 0 and hasattr(self, 'previous_fire_states'):
                new_burning = np.logical_and(
                    fire_states == FireState.Burning,
                    self.previous_fire_states != FireState.Burning
                )
                new_burning_coords = list(zip(*np.where(new_burning)))
                # self.metrics.record_fire_spread(self.step_count, new_burning_coords)
            
            # Store current fire state for next comparison
            self.previous_fire_states = fire_states.copy()
            
            # Update frame history
            current_cells = np.clip(np.array(self.state['cells'], dtype=np.int8), -1, 8)
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
        
        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            return self.cached_obs, 0, True, False, {}
    
    def calculate_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        reward = 0
        if not hasattr(self, 'prev_burning'):
            self.prev_burning = curr_burning

        newly_burnt = curr_burnt - prev_burnt
        burning_reduction = self.prev_burning - curr_burning

        # Reward breakdown components
        proximity_reward = 0.0
        hit_reward = 0.0
        firebreak_reward = 0.0
        wasted_penalty = 0.0
        unburnable_penalty = 0.0

        # Core reward logic
        reward += extinguished_by_helitack * 10
        reward -= newly_burnt * 5
        reward -= curr_burning * 0.1
        reward -= 0.1  # time penalty

        heli_x, heli_y = self.state['helicopter_coord']
        last_action = self.state.get('last_action', None)
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burning_mask = fire_states == FireState.Burning

        # Proximity reward
        if np.any(burning_mask):
            burning_coords = np.argwhere(burning_mask)
            distances = np.sqrt((burning_coords[:, 0] - heli_y) ** 2 + (burning_coords[:, 1] - heli_x) ** 2)
            min_distance = np.min(distances)
            proximity_reward = 2 * np.exp(-min_distance / 20)
            reward += proximity_reward

        # Helitack reward evaluation
        if last_action == 4 and 0 <= heli_y < cells.shape[0] and 0 <= heli_x < cells.shape[1]:
            cell_value = cells[heli_y, heli_x]

            if cell_value == -1:
                unburnable_penalty = -15
                reward += unburnable_penalty
            else:
                fire_state = fire_states[heli_y, heli_x]
                burn_index = cell_value % 3

                if fire_state == FireState.Burning:
                    intensity_bonus = (burn_index + 1) * 2
                    hit_reward = 20 * intensity_bonus
                    reward += hit_reward
                elif fire_state == FireState.Burnt:
                    wasted_penalty = -5
                    reward += wasted_penalty
                elif fire_state == FireState.Unburnt:
                    nearby_burning = np.sum(burning_mask[
                        max(0, heli_y - 5):min(160, heli_y + 6),
                        max(0, heli_x - 5):min(240, heli_x + 6)
                    ])
                    if nearby_burning > 0:
                        firebreak_reward = 5 * min(nearby_burning, 5)
                    else:
                        firebreak_reward = -10
                    reward += firebreak_reward

        self.prev_burning = curr_burning

        # 🧾 Logging
        # if not hasattr(self, "step_count"):
        #     self.step_count = 0
        # if not hasattr(self, "log_path"):
        #     env_id = getattr(self, "env_id", 0)
        #     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     os.makedirs("reward_logs", exist_ok=True)
        #     self.log_path = f"reward_logs/reward_log_{env_id}_{ts}.csv"
        #     with open(self.log_path, mode='w', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([
        #             "step", "helicopter_coord", "action", "prev_burnt", "curr_burnt", "newly_burnt", "curr_burning",
        #             "burning_reduction", "extinguished_by_helitack", "proximity_reward", "hit_reward",
        #             "firebreak_reward", "wasted_penalty", "unburnable_penalty", "final_reward"
        #         ])

        # with open(self.log_path, mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([
        #         self.step_count,
        #         (heli_x, heli_y),
        #         last_action,
        #         prev_burnt,
        #         curr_burnt,
        #         newly_burnt,
        #         curr_burning,
        #         burning_reduction,
        #         extinguished_by_helitack,
        #         round(proximity_reward, 3),
        #         round(hit_reward, 3),
        #         round(firebreak_reward, 3),
        #         round(wasted_penalty, 3),
        #         round(unburnable_penalty, 3),
        #         round(reward, 3)
        #     ])

        self.step_count += 1

        return reward
    
    def close(self):
        print("Closing environment...")
        
        # Clean up resources
        # self.metrics = None
        self.state = None
        self.frame_history = None
        self.cached_obs = None
        clear_gpu_memory()

class DebugRolloutCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True  # Required abstract method

    def _on_rollout_start(self) -> None:
        print("🚀 Starting rollout...")

    def _on_rollout_end(self) -> bool:
        print(f"✅ Rollout complete. Buffer full: {self.model.rollout_buffer.full}")
        print(f"Buffer position: {self.model.rollout_buffer.pos}")
        return True
    
def make_env(rank=0, base_seed=1000):
    def _init():
        env = FireEnvSync(env_id=rank)
        seed = base_seed + rank
        env.seed(seed)  # Seed right here
        print(f"[ENV {rank}] Seeded with {seed}")
        return Monitor(env, logdir)
    return _init

# Main function
def main():
    global model
    vec_env = SubprocVecEnv([make_env(i) for i in range(8)])

    # Restore VecNormalize if available
    if os.path.exists("vecnormalize.pkl") and os.path.getsize("vecnormalize.pkl") > 0:
        print("🔄 Restoring VecNormalize state...")
        vec_env = VecNormalize.load("vecnormalize.pkl", vec_env)
    else:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_obs_keys=["helicopter_coord"],  
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
        print("✅ Model loaded successfully!")
    else:
        print("🆕 No existing model found, initializing new model...")
        model = PPO(
            FireEnvLSTMPolicy,
            vec_env,
            n_steps=96,
            batch_size=384,
            n_epochs=5,
            learning_rate= linear_schedule(1e-4),
            clip_range= linear_schedule(0.2),
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.01,
            verbose=1,
            device=device,
            tensorboard_log=logdir,
            policy_kwargs = dict(
                features_extractor_class=FireEnvLSTMCNN,
                features_extractor_kwargs=dict(features_dim=512)
            )
        )
        print("✅ New model initialized successfully!")
        

    model = fix_env_reset(model)

    if device == 'mps':
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available, falling back to CPU")
            device = 'cpu'
        else:
            print("Configuring MPS-specific optimizations for Apple Silicon")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            model.batch_size = 24
            model.n_steps = 96
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
            total_timesteps=1000000,
            reset_num_timesteps=False,
            tb_log_name="run2",
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
                vec_env.save("vecnormalize.pkl")
                print("✅ Model and normalization state saved.")
            except Exception as e:
                print(f"❌ Error saving model: {e}")

        print("👋 Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()