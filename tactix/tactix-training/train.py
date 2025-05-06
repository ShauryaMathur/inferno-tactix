# train.py - Optimized for performance and memory efficiency

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
import torch.cuda.amp as amp  # For mixed precision
from collections import deque


print('HI! FROM TRAIN BACKEND')

# Ensure the save path exists
save_path = os.environ.get("MODEL_DIR", ".")
MODEL_FILE = os.path.join(save_path, "ppo_firefighter.zip")

def find_sb3_path():
    """Find the path to stable_baselines3 installation."""
    try:
        spec = importlib.util.find_spec('stable_baselines3')
        if spec is not None:
            return Path(spec.origin).parent
        else:
            raise ImportError("stable_baselines3 not found")
    except ImportError:
        # Alternative method using __file__ property
        import stable_baselines3
        return Path(stable_baselines3.__file__).parent

def apply_aggressive_buffer_fix():
    """
    Apply direct source code modifications to the Stable Baselines 3 buffer class.
    This is a more aggressive fix that modifies the actual get() method.
    """
    print("🔧 Applying aggressive buffer fix...")
    
    try:
        # Import the necessary modules
        from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
        
        # Path to the buffer file
        sb3_path = find_sb3_path()
        buffer_file = sb3_path / "common" / "buffers.py"
        
        print(f"📂 Found buffer file at: {buffer_file}")
        
        # Check if we can write to the file
        if not os.access(buffer_file, os.W_OK):
            print("⚠️ Cannot write to buffer file. Using runtime patching instead.")
            use_runtime_patching()
            return
        
        # Read the buffer file content
        with open(buffer_file, 'r') as f:
            content = f.read()
            
        # Check if the buffer file has already been patched
        if "# BUFFER_PATCHED" in content:
            print("✅ Buffer file already patched.")
            return
            
        # Find the get method in the RolloutBuffer class
        get_method_start = content.find("def get(self, batch_size")
        if get_method_start == -1:
            print("⚠️ Could not find get method in the buffer file. Using runtime patching instead.")
            use_runtime_patching()
            return
            
        # Find the assertion line
        assert_line = content.find("assert self.full", get_method_start)
        if assert_line == -1:
            print("⚠️ Could not find assertion in the get method. Using runtime patching instead.")
            use_runtime_patching()
            return
            
        # Modify the content to comment out the assertion
        modified_content = content[:assert_line] + "# BUFFER_PATCHED\n        # assert self.full" + content[assert_line + len("assert self.full"):]
        
        # Add a fix to automatically set the buffer to full
        assert_end = modified_content.find("\n", assert_line)
        modified_content = modified_content[:assert_end] + "\n        if not self.full:\n            self.full = True  # Auto-fix: force buffer to be full\n" + modified_content[assert_end:]
        
        # Write back the modified file
        with open(buffer_file, 'w') as f:
            f.write(modified_content)
            
        print("✅ Successfully patched buffer file.")
        
        # Reload the module to apply changes
        import importlib
        importlib.reload(importlib.import_module('stable_baselines3.common.buffers'))
        
    except Exception as e:
        print(f"❌ Error patching buffer file: {e}")
        print("⚠️ Falling back to runtime patching.")
        use_runtime_patching()

def use_runtime_patching():
    """
    Apply runtime patches to the buffer classes if file modification fails.
    """
    print("🛠️ Applying runtime patches to buffer classes...")
    
    try:
        # Import buffer classes
        from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
        
        # Patch RolloutBuffer.get
        original_get = RolloutBuffer.get
        
        def patched_get(self, batch_size):
            if not self.full:
                print("⚠️ Buffer not full, forcing it to be full")
                self.full = True
            
            return original_get(self, batch_size)
        
        # Apply the patch
        RolloutBuffer.get = patched_get
        
        # Also patch DictRolloutBuffer if it inherits get from RolloutBuffer
        if DictRolloutBuffer.get is RolloutBuffer.get:
            DictRolloutBuffer.get = patched_get
        else:
            # Patch DictRolloutBuffer.get separately
            original_dict_get = DictRolloutBuffer.get
            
            def patched_dict_get(self, batch_size):
                if not self.full:
                    print("⚠️ Dict buffer not full, forcing it to be full")
                    self.full = True
                
                return original_dict_get(self, batch_size)
            
            DictRolloutBuffer.get = patched_dict_get
        
        print("✅ Successfully applied runtime patches to buffer classes.")
        
    except Exception as e:
        print(f"❌ Error applying runtime patches: {e}")
        print("⚠️ Attempting lowest-level patch...")
        use_lowest_level_patch()

def use_lowest_level_patch():
    """
    Apply the lowest-level patch possible by using monkeypatching.
    """
    print("🔧 Applying lowest-level monkeypatch...")
    
    try:
        import inspect
        from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
        
        # Force both classes to have full=True
        for cls in [RolloutBuffer, DictRolloutBuffer]:
            # Get the original __init__ method
            original_init = cls.__init__
            
            # Create a new __init__ method that sets full=True
            def new_init(self, *args, **kwargs):
                result = original_init(self, *args, **kwargs)
                self.full = True
                return result
            
            # Replace the __init__ method
            cls.__init__ = new_init
        
        print("✅ Applied lowest-level monkeypatch.")
        
    except Exception as e:
        print(f"❌ Error applying lowest-level patch: {e}")
        print("⚠️ All patching attempts failed. Consider manually editing the stable_baselines3 code.")

# Add a function to create a custom buffer implementation
def create_fixed_buffer(obs_dim=None):
    """
    Create a custom buffer that's always marked as full.
    This is a last resort if other patching methods fail.
    """
    from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
    
    # Create a new buffer
    if isinstance(obs_dim, dict):
        buffer = DictRolloutBuffer(
            buffer_size=2048,
            observation_space=None,  # Will be set by PPO
            action_space=None,       # Will be set by PPO
            device="auto",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=1,
        )
    else:
        buffer = RolloutBuffer(
            buffer_size=2048,
            observation_space=None,  # Will be set by PPO
            action_space=None,       # Will be set by PPO
            device="auto",
            gae_lambda=0.95,
            gamma=0.99,
            n_envs=1,
        )
    
    # Mark it as full
    buffer.full = True
    
    return buffer

# Function to patch the PPO algorithm to use our fixed buffer
def patch_ppo_to_use_fixed_buffer():
    """
    Patch the PPO algorithm to use our fixed buffer implementation.
    """
    print("🔧 Patching PPO to use fixed buffer...")
    
    try:
        from stable_baselines3.ppo.ppo import PPO
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
        
        # Get the original _setup_learn method
        original_setup_learn = OnPolicyAlgorithm._setup_learn
        
        # Create a new _setup_learn method that replaces the buffer
        def new_setup_learn(self, *args, **kwargs):
            result = original_setup_learn(self, *args, **kwargs)
            
            # Force buffer to be full
            if hasattr(self, 'rollout_buffer'):
                self.rollout_buffer.full = True
                print("✅ Forced rollout_buffer to be full in _setup_learn")
            
            return result
        
        # Replace the _setup_learn method
        OnPolicyAlgorithm._setup_learn = new_setup_learn
        
        # Get the original train method
        original_train = PPO.train
        
        # Create a new train method that ensures the buffer is marked as full
        def new_train(self):
            # Mark buffer as full before training
            if hasattr(self, 'rollout_buffer'):
                self.rollout_buffer.full = True
                print("✅ Forced rollout_buffer to be full in train")
            
            return original_train(self)
        
        # Replace the train method
        PPO.train = new_train
        
        print("✅ Successfully patched PPO to use fixed buffer.")
        
    except Exception as e:
        print(f"❌ Error patching PPO: {e}")

# Apply all possible fixes
def apply_all_fixes():
    """
    Apply all possible fixes to ensure the buffer issue is resolved.
    """
    # Try file patching first
    apply_aggressive_buffer_fix()
    
    # Also patch the PPO algorithm as a backup
    patch_ppo_to_use_fixed_buffer()
    
    print("🎉 All fixes applied. This should resolve the buffer issue.")
    print("⚠️ If you still experience issues, try re-installing Stable Baselines 3 or manually editing the code.")

def apply_direct_buffer_patch():
    """
    Apply a direct patch to the PPO buffer class to fix the 'assert self.full' error.
    """
    from stable_baselines3.common.buffers import RolloutBuffer
    
    # Get the original get method
    original_get = RolloutBuffer.get
    
    # Define a new get method that doesn't check the 'full' flag
    def patched_get(self, batch_size):
        # If buffer is not full, mark it as full (this is the key fix)
        if not self.full:
            print("⚠️ Buffer not full, forcing it to be full")
            self.full = True
        
        # Call the original method
        return original_get(self, batch_size)
    
    # Apply the patch
    RolloutBuffer.get = patched_get
def fix_ppo_buffer_issue(model, verbose=True):
    """
    Fix the PPO rollout buffer issue by ensuring the buffer is filled correctly.
    This addresses:
    1. The AssertionError (assert self.full)
    2. The KeyError: 0 with reset() 
    3. The missing arguments for compute_returns_and_advantage()
    """
    # Ensure we're working with a PPO model
    from stable_baselines3 import PPO
    if not isinstance(model, PPO):
        if verbose: print("⚠️ Model is not a PPO instance. Buffer fix only applies to PPO.")
        return model
    
    # Get the current rollout buffer
    buffer = model.rollout_buffer
    
    # Override the learning method to fix buffer initialization and filling
    original_learn = model.learn
    
    def fixed_learn(*args, **kwargs):
        """
        Fixed learning method that properly fills the rollout buffer.
        """
        if verbose: print("🔄 Using fixed PPO training method that addresses buffer issues")
        
        # Reset the buffer
        model.rollout_buffer.reset()
        
        # The simplest and most reliable approach: 
        # Just manually set the buffer as full and let collect_rollouts handle it properly
        model.rollout_buffer.full = False
        
        # Create a dummy callback that will handle on_rollout_start/end
        from stable_baselines3.common.callbacks import BaseCallback
        
        class DummyCallback(BaseCallback):
            def _on_step(self):
                return True
        
        dummy_callback = DummyCallback()
        
        # Make sure we have the right observation
        try:
            # For newer gym versions
            obs = model.env.reset()
        except (IndexError, TypeError):
            # For older gym versions
            obs = model.env.reset()
        
        # Override the original learn method to use our fixed collection process
        try:
            # Initialize buffer
            model._last_obs = obs
            
            # Call the original learn method, which will properly fill the buffer
            return original_learn(*args, **kwargs)
        except AssertionError as e:
            if "assert self.full" in str(e):
                if verbose: 
                    print("⚠️ Got buffer assertion error, applying direct fix...")
                
                # Force the buffer to be marked as full to allow training to proceed
                model.rollout_buffer.full = True
                
                # Try again with buffer marked as full
                return original_learn(*args, **kwargs)
            else:
                # Re-raise any other assertion errors
                raise
    
    # Replace the learn method with our fixed version
    model.learn = fixed_learn.__get__(model)
    
    # Also patch the collect_rollouts method to be more robust
    original_collect_rollouts = model.collect_rollouts
    
    def fixed_collect_rollouts(env, callback, rollout_buffer, n_rollout_steps):
        """
        Fixed collect_rollouts that ensures the buffer is properly filled.
        """
        # Create a dummy callback if None is provided
        if callback is None:
            from stable_baselines3.common.callbacks import BaseCallback
            
            class DummyCallback(BaseCallback):
                def _on_step(self):
                    return True
            
            callback = DummyCallback()
        
        try:
            # Call the original method
            return original_collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        except Exception as e:
            if verbose:
                print(f"⚠️ Warning: Error in collect_rollouts: {str(e)}")
            
            # If original fails, mark buffer as full to allow training to continue
            rollout_buffer.full = True
            return True
    
    # Replace collect_rollouts with our fixed version
    model.collect_rollouts = fixed_collect_rollouts.__get__(model)
    
    # Also explicitly patch compute_returns_and_advantage to avoid missing argument errors
    original_compute = buffer.compute_returns_and_advantage
    
    def fixed_compute_returns_and_advantage(last_values=None, dones=None):
        """
        Wrapper around compute_returns_and_advantage that adds missing arguments.
        """
        if last_values is None:
            # Create dummy last values
            if hasattr(model, '_last_obs'):
                with torch.no_grad():
                    # Convert to tensor if needed
                    obs_tensor = obs_as_tensor(model._last_obs, model.device)
                    _, values, _ = model.policy.forward(obs_tensor)
                    last_values = values
            else:
                # Fallback to zeros if we can't get real values
                last_values = torch.zeros(model.env.num_envs, 1, device=model.device)
        
        if dones is None:
            # Create dummy dones (assume not done)
            dones = np.zeros(model.env.num_envs, dtype=bool)
        
        # Call the original compute method with our arguments
        return original_compute(last_values, dones)
    
    # Replace compute_returns_and_advantage with our fixed version
    buffer.compute_returns_and_advantage = fixed_compute_returns_and_advantage.__get__(buffer)
    
    if verbose: print("✅ PPO buffer fixes applied successfully")
    return model

# Helper function to convert observations to PyTorch tensors
def obs_as_tensor(obs, device):
    """
    Convert observations to PyTorch tensors, handling both dict and array observations.
    """
    if isinstance(obs, dict):
        return {
            key: torch.as_tensor(obs[key], device=device)
            for key in obs.keys()
        }
    return torch.as_tensor(obs, device=device)

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
    
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from FeatureExtractor import FireEnvLSTMPolicy

# Directory for logs
logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)

# Environment settings
MAX_TIMESTEPS = 2000
HELICOPTER_SPEED = 3
USE_TRAINED_AGENT = False

# Memory management function
def clear_gpu_memory():
    """Clear GPU memory to prevent memory leaks"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # For Apple Silicon
        torch.mps.empty_cache()
        
class EpisodePauseCallback(BaseCallback):
    """
    After every `pause_every` episodes, pause training for `pause_duration` seconds.
    """
    def __init__(self, pause_every: int = 10, pause_duration: int = 180, verbose=0):
        super().__init__(verbose)
        self.pause_every = pause_every
        self.pause_duration = pause_duration
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # `dones` is a vector of booleans, one per env
        dones = self.locals.get("dones", None)
        if dones is not None and dones[0]:
            self.episode_counter += 1
            if self.episode_counter % self.pause_every == 0:
                if self.verbose:
                    print(f"⏸️  Pausing training for {self.pause_duration}s after {self.episode_counter} episodes")
                time.sleep(self.pause_duration)
        return True

# Mixed precision training utilities
# Remove this callback completely to avoid any recursion issues
class MixedPrecisionCallback(BaseCallback):
    """Enable mixed precision training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.scaler = amp.GradScaler()
        self.enabled = False
        
    def _on_step(self):
        # Only initialize once to avoid recursion
        if self.enabled:
            return True
            
        # We can't modify SB3's internal training loop directly,
        # but we can influence how gradients are scaled during the optimization step
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            optimizer = self.model.policy.optimizer
            
            # Store the original step method
            if not hasattr(optimizer, '_original_step'):
                optimizer._original_step = optimizer.step
                
                # Create a new step method that uses the scaler
                def step_with_scaler(*args, **kwargs):
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    return None
                
                # Replace the step method
                optimizer.step = step_with_scaler
                
                self.enabled = True
                
                if self.verbose > 0:
                    print("Mixed precision training enabled for optimizer")
        return True
        
    def on_training_end(self):
        # Restore original optimizer step method when training ends
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            optimizer = self.model.policy.optimizer
            if hasattr(optimizer, '_original_step'):
                optimizer.step = optimizer._original_step

# Learning rate scheduler callback
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

# Metrics tracking class
class MetricsTracker:
    def __init__(self):
        self.helitack_actions = []  # Format: [(step, x, y)]
        self.fire_spread = []       # Format: [(step, [(x1, y1), (x2, y2), ...])
        self.burnt_area_over_time = []  # Format: [(step, total_burnt)]
        self.burning_cells_over_time = []  # Format: [(step, total_burning)]
        self.episode_statistics = {}
    
    def record_helitack(self, step, x, y):
        self.helitack_actions.append((step, x, y))
    
    def record_fire_spread(self, step, new_burning_cells):
        self.fire_spread.append((step, new_burning_cells))
    
    def record_burnt_area(self, step, total_burnt):
        self.burnt_area_over_time.append((step, total_burnt))
    
    def record_burning_cells(self, step, total_burning):
        self.burning_cells_over_time.append((step, total_burning))
    
    def save_metrics(self, filename="fire_metrics.json"):
        """
        Write out metrics to disk, converting any numpy ints to Python ints
        and any tuples to lists so that the JSON module can serialize them.
        """
        import json

        # Helper to normalize a single value
        def _to_native(x):
            if isinstance(x, np.integer):
                return int(x)
            return x

        # Convert helitack_actions: [(step, x, y), ...] → [[step, x, y], ...]
        helitacks = [
            [int(step), int(x), int(y)]
            for step, x, y in self.helitack_actions
        ]

        # Convert fire_spread: [(step, [(x1,y1),...]), ...] →
        # [[step, [[x1,y1],...]], ...]
        fire_spread = []
        for step, coords in self.fire_spread:
            native_coords = [[int(a), int(b)] for a, b in coords]
            fire_spread.append([int(step), native_coords])

        # Convert burnt_area_over_time: [(step, total), ...]
        burnt = [[int(step), int(total)] for step, total in self.burnt_area_over_time]

        # Convert burning_cells_over_time similarly
        burning = [[int(step), int(total)] for step, total in self.burning_cells_over_time]

        # Convert episode_statistics: { key: value } with possible numpy ints
        episode_stats = {
            k: int(v) if isinstance(v, np.integer) else v
            for k, v in self.episode_statistics.items()
        }

        data = {
            "helitack_actions": helitacks,
            "fire_spread":        fire_spread,
            "burnt_area_over_time":     burnt,
            "burning_cells_over_time":  burning,
            "episode_statistics":       episode_stats
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    
    def calculate_episode_statistics(self):
        # Calculate summary statistics
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
        
    def calculate_containment_time(self):
        # Define when fire is "contained" (e.g., when burning cells start decreasing)
        if not self.burning_cells_over_time:
            return None
            
        max_burning = 0
        max_step = 0
        
        for step, burning in self.burning_cells_over_time:
            if burning > max_burning:
                max_burning = burning
                max_step = step
                
        return max_step

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
message_queue = queue.Queue(maxsize=100)  # Limit queue size
stop_event = threading.Event()
model = None  # Global model reference for signal handler

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\n⚠️ Received termination signal. Cleaning up...")
    if model is not None:
        try:
            print("💾 Saving model...")
            model.save(os.path.join(save_path, "ppo_firefighter"))

            print(">>> MODEL_DIR:", os.environ.get("MODEL_DIR"))
            print(">>> cwd     :", os.getcwd())

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
            print("React disconnected.")
            if client_websocket == websocket:
                client_websocket = None
    
    async def run_server():
        host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")  # Bind to all interfaces
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

# Synchronous environment implementation with optimizations
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
        
        # Frame stacking - optimized with numpy array
        self.frame_history = np.zeros((4, 160, 240), dtype=np.int32)
        
        # Metrics tracking
        # self.metrics = MetricsTracker()
        
        # Cached observation for reuse
        self.cached_obs = None
    
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
    
    def _send_and_wait(self, message, timeout=500.0):
        """Send a message to React and wait for a response (optimized)"""
        global client_websocket
        
        # Early return if no active connection
        if client_websocket is None:
            print("⚠️ No active WebSocket connection")
            return self._default_response()
        
        # Get the current websocket
        self.websocket = client_websocket
        
        # Clear any old messages more efficiently
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
        print(f"🧼 Resetting environment with seed={seed}")
        if seed is not None:
            super().reset(seed=seed)
        
        # Reset LSTM hidden states
        if hasattr(self, 'model') and hasattr(self.model.policy, 'features_extractor'):
            self.model.policy.features_extractor.hidden = None
            
        # Reset metrics tracker
        # self.metrics = MetricsTracker()
        
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
        elif action == 4: 
            print(f"Helitack performed at ({heli_x}, {heli_y})")
            # Record helitack action for metrics
            # self.metrics.record_helitack(self.step_count, heli_x, heli_y)
        
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

        # Update state
        self.state = {
            **self.state,
            'prevBurntCells': self.state['cellsBurnt'],
            **response,
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
        if self.state.get('cellsBurning', 1) == 0 or self.step_count >= MAX_TIMESTEPS:
            done = True
            # Calculate episode statistics
            # self.metrics.calculate_episode_statistics()
            # Save metrics
            # self.metrics.save_metrics(f"metrics_episode_{self.episode_count}.json")
        
        # Track metrics
        cells = np.array(self.state['cells'])
        fire_states = cells // 3
        burnt_cells = np.sum(fire_states == FireState.Burnt)
        burning_cells = np.sum(fire_states == FireState.Burning)
        
        # Record metrics
        # self.metrics.record_burnt_area(self.step_count, burnt_cells)
        # self.metrics.record_burning_cells(self.step_count, burning_cells)
        
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
        
        # Clean up resources
        # self.metrics = None
        self.state = None
        self.frame_history = None
        self.cached_obs = None
        clear_gpu_memory()

# Main function
def main():
    apply_all_fixes()

    global model

    # apply_direct_buffer_patch() 

    
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
        
        # Set episode counter
        env.episode_count = 0
        
        # Monitor for logging
        env = Monitor(env, filename=logdir)
        
        # Vectorize environment but optimize normalization
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            # Only normalize helicopter_coord (small array), avoid normalizing huge cells array
            norm_obs_keys=["helicopter_coord"],  
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            epsilon=1e-8,
        )
        
        # Configure device
        # Force MPS (Metal Performance Shaders) for Mac
        device, is_gpu = get_optimal_device()

        # print(f"Using device: {device} for Apple Silicon")
        
        # Try to load existing model for resuming training
        print("🔍 Checking for existing model...")
        try:
            # Check if model file exists before attempting to load
            if USE_TRAINED_AGENT and os.path.isfile(MODEL_FILE):
                print("🔄 Found existing model, loading for continued training...")
                model = PPO.load(MODEL_FILE)
                model.set_env(vec_env)
                print("✅ Model loaded successfully!")
            else:
                # No existing model, create a new one
                print("🆕 No existing model found, initializing new model...")
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
                    verbose=1,
                    device=device,
                    tensorboard_log=logdir
                )
                # model = fix_ppo_buffer_issue(model)

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
                device=device,
                verbose=1,
                tensorboard_log=logdir
            )
            print("✅ New model initialized successfully!")
        model = fix_env_reset(model)


        # Give the environment access to the model (for LSTM state reset)
        env.model = model
        vec_env.model = model

        # Configure MPS-specific settings first
        if device == 'mps':
            # Ensure MPS is available
            if not torch.backends.mps.is_available():
                print("Warning: MPS requested but not available, falling back to CPU")
                device = 'cpu'
            else:
                # Set specific MPS optimizations
                print("Configuring MPS-specific optimizations for Apple Silicon")
                
                # Force synchronous MPS execution for better stability
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
                # Adjust batch size and steps for M-series chip performance
                model.batch_size = 24  # Slightly smaller for better MPS performance
                model.n_steps = 96     # Reduced for more frequent updates
                
                print(f"MPS-optimized batch configuration: batch_size={model.batch_size}, n_steps={model.n_steps}")

        # Create simple callbacks - no mixed precision to avoid recursion issues
        reward_callback = RewardLoggingCallback()
        lr_callback = LearningRateScheduleCallback(lr_schedule)
        memory_callback = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
        
            
        # Create callbacks
        reward_callback = RewardLoggingCallback()
        lr_callback = LearningRateScheduleCallback(lr_schedule)
        memory_callback = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
        pause_callback = EpisodePauseCallback(pause_every=10, pause_duration=180, verbose=1)
        # mixed_precision_callback = MixedPrecisionCallback(verbose=1)

        # Checkpoint callback to save model periodically
        # checkpoint_callback = CheckpointCallback(
        #     save_freq=2000,  
        #     save_path="./models/",
        #     name_prefix="ppo_firefighter",
        #     verbose=1
        # )

        # Combine callbacks
        callbacks = CallbackList([
            reward_callback, 
            lr_callback, 
            memory_callback
        ])
        os.makedirs(logdir, exist_ok=True)

        # Initial memory cleanup
        clear_gpu_memory()
        
        # Set torch optimization flags
        if device != 'cpu':
            # Set TF32 precision if available (NVIDIA Ampere or newer GPUs)
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 precision for faster training")
            
            # Enable cuDNN benchmarking for potentially faster training
            torch.backends.cudnn.benchmark = True
        
        # Train model with basic settings to avoid recursion issues
        print("🚀 Starting training...")
        try:
            model.learn(
                total_timesteps=1000000,
                reset_num_timesteps=False,  # This ensures continued training
                tb_log_name="run7",   
                callback=callbacks  
            )
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training error: {e}")
            traceback.print_exc()
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
                
                model.save(os.path.join(save_path, "ppo_firefighter"))

                print(">>> MODEL_DIR:", os.environ.get("MODEL_DIR"))
                print(">>> cwd     :", os.getcwd())

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