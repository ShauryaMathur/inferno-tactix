# wildfireenv.py
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
                # Process the message
                await self.process_message(msg)
        except Exception as e:
            print(f"[RECV THREAD ERROR] {e}")
            self._reader_task_running = False

    async def process_message(self, msg):
        try:
            client_state = json.loads(msg)
            print(f"📨 Processed message: {client_state}")
            # Update state
            if isinstance(client_state, dict):
                self.state.update(client_state)
        except Exception as e:
            print(f"❌ Failed to process message: {e}")

    async def wait_for_client_update(self, timeout=15.0):
        """Wait for next message from React client"""
        try:
            print("[QUEUE] Waiting for next message from queue...")
            # Get message from queue with timeout
            message = await asyncio.wait_for(self.msg_queue.get(), timeout=timeout)
            print("[QUEUE] Got message from queue:", message)
            
            try:
                # Parse the message
                client_state = json.loads(message)
                print(f"📨 Processed message: {client_state}")
                return client_state
            except Exception as e:
                print(f"❌ Failed to parse JSON: {e}")
                return self._initial_state()
        except asyncio.TimeoutError:
            print(f"⚠️ Timeout waiting for response (after {timeout}s)")
            return self._initial_state()
        except Exception as e:
            print(f"🔥 Error in client update: {e}")
            return self._initial_state()

    async def reset(self, *, seed=None, options=None):
        print("🧼 async reset() called")
        super().reset(seed=seed)
        print(f"🔁 Inside async reset(), running loop: {asyncio.get_running_loop()}")
        
        self.state = self._initial_state()
        
        if not self.websocket or not self.websocket.open:
            raise RuntimeError("WebSocket not connected!")
        
        # Send reset message to frontend
        print("📨 reset(): sending 'reset' message to frontend")
        reset_message = json.dumps({"action": "reset"})
        await self.websocket.send(reset_message)
        
        # Wait for response from React
        print("⏳ reset(): waiting for client update...")
        response = await self.wait_for_client_update(timeout=15.0)
        
        # Update local state
        if isinstance(response, dict):
            self.state.update(response)
        
        # Extract observation from state
        data = self.state
        
        obs = {
            'helicopter_coord': np.array(data.get('helicopter_coord', (0, 0)), dtype=np.int32),
            'cells': np.clip(np.array(data.get('cells', np.zeros((240, 160))), dtype=np.int32), 0, 8),
            'on_fire': int(data.get('on_fire', 0))
        }
        
        print("🎯 reset(): observation created")
        return obs, {}
    
    async def step(self, action, stepcount):
        print(f"🔄 async step({action}) called")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state.get('helicopter_coord', (0, 0))
        
        if action == 0: heli_y += 5      # Move down
        elif action == 1: heli_y -= 5    # Move up
        elif action == 2: heli_x -= 5    # Move left
        elif action == 3: heli_x += 5    # Move right
        elif action == 4: 
            print(f"Helitack performed at ({heli_x}, {heli_y})")
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 10, 239))
        heli_y = int(np.clip(heli_y, 10, 159))
        self.state['helicopter_coord'] = (heli_x, heli_y)
        
        # Send action to the frontend
        if self.websocket and self.websocket.open:
            message = json.dumps({
                "action": str(action),
                "helicopter_coord": [heli_x, heli_y]
            }, cls=NumpyEncoder)
            await self.websocket.send(message)
            print(f"📤 Step message sent: action={action}, pos=({heli_x}, {heli_y})")
        
        # Wait for response from React
        print("⏳ step(): waiting for client update...")
        response = await self.wait_for_client_update(timeout=10.0)
        
        # Update local state with response
        if isinstance(response, dict):
            self.state.update(response)
        
        # Extract reward and done from state
        reward = float(self.state.get('reward', -0.1))
        done = bool(self.state.get('done', False))
        
        # Get observation from state
        observation = {
            'helicopter_coord': np.array(self.state.get('helicopter_coord', (heli_x, heli_y)), dtype=np.int32),
            'cells': np.clip(np.array(self.state.get('cells', np.zeros((240, 160))), dtype=np.int32), 0, 8),
            'on_fire': int(self.state.get('on_fire', 0))
        }
        
        # Check for max steps
        if stepcount >= MAX_TIMESTEPS:
            done = True
        
        return observation, reward, done, False, {}
    
    def compute_reward(self, prev_burnt, curr_burnt, curr_burning, extinguished_by_helitack):
        """Compute reward based on fire state"""
        newly_burnt = curr_burnt - prev_burnt
        
        reward = 0
        reward += extinguished_by_helitack * 10  # reward for extinguishing
        reward -= newly_burnt * 5                # penalty for fire spread
        reward -= curr_burning * 1               # penalty for ongoing fires
        reward -= 0.1                            # step penalty
        
        return reward

    async def close(self):
        print("🔄 Closing async environment")
        if self.websocket and self.websocket.open:
            try:
                await self.websocket.send(json.dumps({"action": "close"}))
            except Exception as e:
                print(f"Error sending close message: {e}")
        return True

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
            'reward': -0.1
        }