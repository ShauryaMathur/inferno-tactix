# infernoenv.py
import threading
import asyncio
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traceback
import pdb
from wildfireenv import FireSimulationGymEnv

class FireEnvSyncWrapper(gym.Env):
    def __init__(self, env: FireSimulationGymEnv):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self._async_env = env  # Use the provided async environment directly
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        self.step_count = 0
        self.metadata = {"render.modes": []}
        self.action_space = self._async_env.action_space
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0, high=8, shape=(240, 160), dtype=np.int32),
            'on_fire': spaces.Discrete(2)
        })
    
    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async(self, coro, timeout=120):
        print("🚀 Scheduling async coroutine on loop:", self.loop)
        fut = asyncio.run_coroutine_threadsafe(coro, self.loop)
        print("⏳ Awaiting result...")
        # pdb.set_trace()

        try:
            result = fut.result(timeout=timeout)
            print("✅ Coroutine completed.")
            return result
        except Exception as e:
            print(f"❌ Async call failed inside _run_async: {e}")
            traceback.print_exc()
            raise

    def reset(self, *, seed=None, options=None):
        print(f"🧼 Resetting environment with seed={seed}")
        super().reset(seed=seed)
        print("Resetting async environment...")
        self.step_count = 0
        print("👉 Calling async_env.reset")
        print(f"Loop running: {self.loop.is_running()}")
        observation, info = self._run_async(self._async_env.reset(seed=seed, options=options))
        print("✅ async_env.reset completed")
        final_observation = {
            'helicopter_coord': np.array(observation.get('helicopter_coord', (0, 0)), dtype=np.int32),
            'cells': np.clip(np.array(observation.get('cells', np.zeros((240, 160))), dtype=np.int32), 0, 8),
            'on_fire': int(observation.get('on_fire', 0))
        }
        print("State after reset:", self._describe_state(final_observation))
        return final_observation, info


    def step(self, action):
        self.step_count += 1
        print(f"\n[Step {self.step_count}] Action taken: {action}")
        state, reward, done, truncated, info = self._run_async(self._async_env.step(action, self.step_count))
        print(f"State: {self._describe_state(state)}")
        print(f"Reward: {reward}, Done: {done}, Info: {info}")
        done = bool(done)
        truncated = bool(truncated)
        observation = {
            'cells': np.array(state['cells'], dtype=np.int32),
            'helicopter_coord': np.array(state['helicopter_coord'], dtype=np.int32),
            'on_fire': int(state['on_fire'])
        }
        return observation, reward, done, truncated, info

    def close(self):
        print("Closing environment...")
        # No need to manage a separate loop here
        pass

    def _describe_state(self, state):
        try:
            coords = state['helicopter_coord']
            fire = state['on_fire']
            cell_mean = np.mean(state['cells'])
            return f"Coord: {coords}, On fire: {fire}, Cell mean: {cell_mean:.2f}"
        except Exception as e:
            return f"Invalid state! Error: {e}, Raw state: {state}"