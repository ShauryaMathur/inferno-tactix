#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rllib_fire_training.py

End-to-end multi-agent PPO training with Ray RLlib (2.x+), single-process.
Two helitack agents (shared policy) mitigating a fire simulated in React over WebSocket.
"""

import os
import sys
import time
import gc
import json
import signal
import threading
import queue
import asyncio

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Debug: Print the type of PPOConfig to verify it's a class and not a string
print(f"PPOConfig type: {type(PPOConfig)}")

import websockets

# -----------------------------------------------------------------------------
# 1. WebSocket server + FireEnvSync environment
# -----------------------------------------------------------------------------

client_ws = None
msg_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class FireEnvSync(gym.Env):
    """Gym env proxying to React over WebSocket for fire simulation."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "helicopter_coord": spaces.Box(low=np.array([0,0]),
                                           high=np.array([239,159]),
                                           dtype=np.int32),
            "cells": spaces.Box(low=0, high=8, shape=(4,160,240), dtype=np.int32),
            "on_fire": spaces.Discrete(2),
        })
        self.step_count = 0
        self.episode_count = 0
        self._make_state()
        self.frame_history = np.zeros((4,160,240), dtype=np.int32)

    def _make_state(self):
        self.state = {
            "helicopter_coord": [70,30],
            "cells": np.zeros((160,240), dtype=int).tolist(),
            "on_fire": 0,
            "prevBurntCells": 0,
            "cellsBurnt": 0,
            "cellsBurning": 0,
            "quenchedCells": 0
        }

    async def _ws_handler(self, ws, path):
        global client_ws
        if client_ws:
            await client_ws.close()
        client_ws = ws
        print("🔥 React connected")
        try:
            async for msg in ws:
                try:
                    d = json.loads(msg)
                    if d.get("type") == "ping":
                        await ws.send(json.dumps({"type":"pong"}))
                        continue
                except:
                    pass
                try:
                    msg_queue.put_nowait(msg)
                except queue.Full:
                    msg_queue.get_nowait()
                    msg_queue.put_nowait(msg)
        except Exception as e:
            print("WebSocket error:", e)
        finally:
            print("React disconnected")
            if client_ws == ws:
                client_ws = None

    def start_ws(self):
        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            server = loop.run_until_complete(
                websockets.serve(self._ws_handler,
                                 os.environ.get("WEBSOCKET_HOST","0.0.0.0"),
                                 int(os.environ.get("WEBSOCKET_PORT","8765"))))
            print(f"🟢 WS server on {server.sockets[0].getsockname()}")
            loop.run_until_complete(self._wait_stop())
            server.close()
            loop.run_until_complete(server.wait_closed())
            loop.close()
        threading.Thread(target=runner, daemon=True).start()

    async def _wait_stop(self):
        while not stop_event.is_set():
            await asyncio.sleep(0.1)

    def _rpc(self, payload, timeout=10.0):
        global client_ws
        if client_ws is None:
            return self.state
        try:
            while True: msg_queue.get_nowait()
        except queue.Empty:
            pass
        asyncio.run(client_ws.send(json.dumps(payload)))
        start = time.time()
        while time.time() - start < timeout:
            try:
                raw = msg_queue.get(timeout=0.5)
                return json.loads(raw)
            except queue.Empty:
                continue
        return self.state

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        self.episode_count += 1
        self._make_state()
        resp = self._rpc({"action":"reset","episode":self.episode_count})
        self.state.update(resp)
        cells = np.clip(np.array(self.state["cells"],dtype=int),0,8)
        for i in range(4): self.frame_history[i] = cells.copy()
        obs = {
            "helicopter_coord": np.array(self.state["helicopter_coord"],dtype=int),
            "cells": self.frame_history.copy(),
            "on_fire": int(self.state["on_fire"])
        }
        return obs, {}

    def step(self, action):
        self.step_count += 1
        hx,hy = self.state["helicopter_coord"]
        s=3
        if action==0: hy+=s
        if action==1: hy-=s
        if action==2: hx-=s
        if action==3: hx+=s
        hx,hy = int(np.clip(hx,0,239)), int(np.clip(hy,0,159))
        resp = self._rpc({"action":str(action),"helicopter_coord":[hx,hy]})
        self.state.update(resp)
        self.state["helicopter_coord"] = [hx,hy]
        prev = self.state.get("prevBurntCells",0)
        curr = self.state.get("cellsBurnt",0)
        burning = self.state.get("cellsBurning",0)
        quenched = self.state.get("quenchedCells",0)
        reward = quenched*10 - (curr-prev)*5 - burning*0.1 - 0.1
        done = (burning==0 or self.step_count>=2000)
        cells_arr = np.clip(np.array(self.state["cells"],dtype=int),0,8)
        self.frame_history = np.roll(self.frame_history,-1,axis=0)
        self.frame_history[3] = cells_arr
        obs = {
            "helicopter_coord": np.array(self.state["helicopter_coord"],dtype=int),
            "cells": self.frame_history.copy(),
            "on_fire": int(self.state["on_fire"])
        }
        return obs, reward, done, False, {}

    def close(self):
        stop_event.set()
        if client_ws:
            try: asyncio.run(client_ws.send(json.dumps({"action":"close"})))
            except: pass

# -----------------------------------------------------------------------------
# 2. MultiFireEnv wrapper
# -----------------------------------------------------------------------------

from ray.rllib.env.multi_agent_env import MultiAgentEnv

class MultiFireEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.env = FireEnvSync()
        self.env.start_ws()
        self.env.reset()
        self.agent_ids = ["helitack_0","helitack_1"]
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, *, seed=None, options=None):
        obs, _ = self.env.reset()
        obs_dict = {aid: obs for aid in self.agent_ids}
        info_dict = {aid: {} for aid in self.agent_ids}
        return obs_dict, info_dict

    def step(self, action_dict):
        total_rew = 0.0
        terminated = False
        info = {}
        for aid, act in action_dict.items():
            obs, rew, term, _, info = self.env.step(act)
            total_rew += rew
            if term:
                terminated = True
                break
        obs_dict = {aid: obs for aid in self.agent_ids}
        rew_dict = {aid: total_rew for aid in self.agent_ids}
        term_dict = {aid: terminated for aid in self.agent_ids}
        term_dict["__all__"] = terminated
        trunc_dict = {aid: False for aid in self.agent_ids}
        trunc_dict["__all__"] = False
        info_dict = {aid: info for aid in self.agent_ids}
        return obs_dict, rew_dict, term_dict, trunc_dict, info_dict

# -----------------------------------------------------------------------------
# 3. Custom callbacks
# -----------------------------------------------------------------------------

class FireCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if result["timesteps_total"] % 5000 < result["train_batch_size"]:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

# -----------------------------------------------------------------------------
# 4. Training with PPOConfig
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # Dummy env for space extraction
    dummy = MultiFireEnv({})
    obs_space = dummy.observation_space
    act_space = dummy.action_space

    # Create a dictionary configuration instead of using the builder pattern
    config = {
        "env": MultiFireEnv,
        "disable_env_checking": True,
        "num_rollout_workers": 0,
        "num_envs_per_worker": 1,
        "framework": "torch",
        "rollout_fragment_length": 128,
        "train_batch_size": 128,
        "sgd_minibatch_size": 32,
        "num_sgd_iter": 4,
        "lr": 3e-4,
        "clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "gamma": 0.99,
        "lambda_": 0.95,
        "kl_target": 0.01,
        "callbacks": FireCallbacks,
        "multiagent": {
            "policies": {"shared_helitack": (None, obs_space, act_space, {})},
            "policy_mapping_fn": lambda aid, **kw: "shared_helitack",
            "policies_to_train": ["shared_helitack"],
        }
    }

    # Create the PPO trainer with the dictionary config
    trainer = PPO(config=config)

    for i in range(200):
        result = trainer.train()
        print(f"Iter {i}: reward_mean={result['episode_reward_mean']:.2f}")
        if (i+1) % 10 == 0:
            ckpt = trainer.save()
            print("Checkpoint at", ckpt)

    trainer.stop()
    stop_event.set()
    sys.exit(0)
