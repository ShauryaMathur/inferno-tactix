import numpy as np
import asyncio
import json
import websockets
import gymnasium as gym
from gymnasium import spaces

MAX_TIMESTEPS = 1000
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class FireSimulationGymEnv(gym.Env):
    def __init__(self, websocket=None, loop=None, msg_queue=None):
        super().__init__()
        self.websocket = websocket
        self.loop = loop or asyncio.get_event_loop()
        self.msg_queue = msg_queue
        self._reader_task_running = False

        # Initialize spaces
        self.action_space = spaces.Discrete(5)
        self.state = self._initial_state()

        # Start the reader task only once
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: self.loop.create_task(self._init_async_fields()))
        else:
            raise RuntimeError("Event loop not provided to FireSimulationGymEnv!")

    async def _init_async_fields(self):
        print("Loop inside _init_async_fields:", asyncio.get_running_loop())
        print("Loop inside _reader_task:", asyncio.get_running_loop())

        if self.websocket and not self._reader_task_running:
            self._reader_task_running = True
            print("🚀 Launching _reader_task from inside _init_async_fields()")
            asyncio.create_task(self._reader_task())

    async def _reader_task(self):
        print(f"📥 _reader_task started for websocket id: {id(self.websocket)}")
        try:
            while True:
                print("📥 _reader_task(): waiting for message...")
                msg = await self.msg_queue.get()  # Read from the queue
                print(f"📨 _reader_task(): got message: {msg}")
                # You can process the message here (e.g., update the environment state)
                await self.process_message(msg)
        except Exception as e:
            print(f"[RECV THREAD ERROR] {e}")
            self._reader_task_running = False

    async def process_message(self, msg):
        try:
            client_state = json.loads(msg)
            print(f"📨 Processed message: {client_state}")
            # Process the client message and update state
            self.state = self._initial_state()  # Just for illustration; update this with your logic
        except Exception as e:
            print(f"❌ Failed to process message: {e}")

    async def reset(self, *, seed=None, options=None):
        print("🧼 async reset() called")

        super().reset(seed=seed)
        print(f"🔁 Inside async reset(), running loop: {asyncio.get_running_loop()}")

        self.state = self._initial_state()

        if not self.websocket:
            raise RuntimeError("WebSocket not connected!")

        print("📨 reset(): sending 'reset' message to frontend")
        reset_message = json.dumps({"action": "reset"})
        await self.websocket.send(reset_message)

        print("⏳ reset(): waiting for client update...")
        try:
            # Timeout set to 30 seconds to avoid hanging indefinitely
            self.state = await asyncio.wait_for(self.wait_for_client_update(), timeout=30.0)
            print("✅ reset(): client responded")
        except asyncio.TimeoutError:
            print("⚠️ reset(): timeout waiting for client update")
            self.state = self._initial_state()

        data = self.state[0] if isinstance(self.state, tuple) else self.state

        obs = {
            'helicopter_coord': np.array(data.get('helicopter_coord', (0, 0)), dtype=np.int32),
            'cells': np.clip(np.array(data.get('cells', np.zeros((240, 160))), dtype=np.int32), 0, 8),
            'on_fire': int(data.get('on_fire', 0))
        }

        print("🎯 reset(): observation created")
        return obs, {}
    
    async def step(self, action, stepcount):
        reward = 0
        heli_x, heli_y = self.state['helicopter_coord']

        if action == 0: heli_y += 5
        elif action == 1: heli_y -= 5
        elif action == 2: heli_x -= 5
        elif action == 3: heli_x += 5
        elif action == 4: print(f"Helitack performed at ({heli_x}, {heli_y})")

        heli_x = int(np.clip(heli_x, 10, 239))
        heli_y = int(np.clip(heli_y, 10, 159))
        self.state['helicopter_coord'] = (heli_x, heli_y)

        if self.websocket:
            await self.websocket.send(json.dumps({
                "action": str(action),
                "helicopter_coord": list(self.state['helicopter_coord'])
            }, cls=NumpyEncoder))

        self.state['prevBurntCells'] = self.state['cellsBurnt']
        self.state = await self.wait_for_client_update()

        reward = self.compute_reward(
            self.state['prevBurntCells'],
            self.state['cellsBurnt'],
            self.state['cellsBurning'],
            self.state['quenchedCells']
        )

        if stepcount >= MAX_TIMESTEPS:
            self.state['done'] = 1

        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8),
            'on_fire': int(self.state.get('on_fire', 0)),
        }
        return observation, reward, self.state['done'], False, {}
    def compute_reward(self,prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
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

    async def wait_for_client_update(self):
        try:
            print("[QUEUE] Waiting for next message from queue...")
            message = await self.msg_queue.get()  # Get message from the queue
            print("[QUEUE] Got message from queue:", message)

            try:
                client_state = json.loads(message)
            except Exception as e:
                print("❌ Failed to parse JSON:", e)
                return self._initial_state()

            print(f"📨 Processed message: {client_state}")
            return client_state

        except Exception as e:
            print(f"🔥 Error in client update: {e}")
            return self._initial_state()

    def _initial_state(self):
        return {
            'helicopter_coord': (0, 0),
            'cells': np.zeros((240, 160)),
            'done': 0,
            'cellsBurning': 0,
            'cellsBurnt': 0,
            'prevBurntCells': 0,
            'quenchedCells': 0,
            'on_fire': 0,
        }
