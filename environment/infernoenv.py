import threading
import asyncio
import gym
import numpy as np
from gym import spaces


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from wildfireenv import FireSimulationGymEnv

class FireEnvSyncWrapper(gym.Env):
    def __init__(self, env: FireSimulationGymEnv):
        super(FireEnvSyncWrapper, self).__init__()
        self._async_env = env
        self.action_space = env.action_space
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=0, high=240, shape=(2,), dtype=np.int32),
            # 'cells': spaces.Box(low=0, high=500, shape=(240, 160), dtype=np.float32),
            'cells': spaces.Box(low=0, high=8, shape=(240,160), dtype=np.int32),
            'on_fire':spaces.Discrete(2)
        })
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        self.step_count = 0

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def reset(self):
        state = self._run_async(self._async_env.reset())
        return state

    def step(self, action):
        self.step_count += 1

        state, reward, done, info = self._run_async(self._async_env.step(action, self.step_count))
        return state, reward, done, info

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()