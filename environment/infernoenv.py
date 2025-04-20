# infernoenv.py - A more direct approach using subroutines

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import json
import time
import threading
import queue

class FireEnvSyncWrapper(gym.Env):
    def __init__(self, websocket, msg_queue):
        super().__init__()
        # Store WebSocket directly
        self.websocket = websocket
        self.msg_queue = msg_queue
        
        # Create a response queue for direct communication
        self.response_queue = queue.Queue()
        
        # Start a thread to listen for messages
        self.running = True
        self.listener_thread = threading.Thread(target=self._message_listener, daemon=True)
        self.listener_thread.start()
        
        # Initialize spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            'helicopter_coord': spaces.Box(low=np.array([0, 0]), high=np.array([239, 159]), dtype=np.int32),
            'cells': spaces.Box(low=0, high=8, shape=(240, 160), dtype=np.int32),
            'on_fire': spaces.Discrete(2)
        })
        
        # Initialize state
        self.step_count = 0
        self.state = self._default_state()
    
    def _default_state(self):
        return {
            'helicopter_coord': np.array([0, 0], dtype=np.int32),
            'cells': np.zeros((240, 160), dtype=np.int32),
            'on_fire': 0
        }
    
    def _message_listener(self):
        """Thread that listens for messages from the queue and processes them"""
        while self.running:
            try:
                # Get a message from the async queue (with timeout to allow checking running flag)
                message = None
                while self.running and message is None:
                    try:
                        # Use a short timeout to periodically check if we should stop
                        message = self.msg_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                
                if not self.running:
                    break
                
                # Process the message
                print(f"💬 Message received in listener thread: {message}")
                try:
                    data = json.loads(message)
                    
                    # Put the response in our synchronous queue
                    self.response_queue.put(data)
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
            except Exception as e:
                print(f"❌ Error in message listener: {e}")
    
    def _send_and_wait(self, message, timeout=10.0):
        """Send a message to React and wait for a response"""
        # Clear any old messages
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break
        
        # Send the message
        print(f"📤 Sending message to React: {message}")
        self.websocket.send(message)
        
        # Wait for a response with timeout
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to get a response (with a short timeout)
                response = self.response_queue.get(timeout=0.1)
                print(f"📥 Received response: {response}")
                return response
            except queue.Empty:
                # No response yet, continue waiting
                pass
        
        # Timeout reached
        print(f"⚠️ Timeout waiting for response after {timeout}s")
        return None
    
    def reset(self, *, seed=None, options=None):
        print(f"🧼 Resetting environment with seed={seed}")
        super().reset(seed=seed)
        self.step_count = 0
        
        # Create reset message
        reset_message = json.dumps({"action": "reset"})
        
        # Send reset message and wait for response
        response = self._send_and_wait(reset_message, timeout=5.0)
        
        # Update state based on response
        if response:
            try:
                # Update state with response data
                if 'cells' in response:
                    cells = response['cells']
                    self.state['cells'] = np.array(cells, dtype=np.int32)
                
                if 'helicopter_coord' in response:
                    self.state['helicopter_coord'] = np.array(response['helicopter_coord'], dtype=np.int32)
                
                if 'on_fire' in response:
                    self.state['on_fire'] = int(response['on_fire'])
            except Exception as e:
                print(f"❌ Error updating state: {e}")
        
        # Return observation
        final_observation = {
            'helicopter_coord': np.array(self.state.get('helicopter_coord', (0, 0)), dtype=np.int32),
            'cells': np.clip(np.array(self.state.get('cells', np.zeros((240, 160))), dtype=np.int32), 0, 8),
            'on_fire': int(self.state.get('on_fire', 0))
        }
        
        print("State after reset:", self._describe_state(final_observation))
        return final_observation, {}
    
    def step(self, action):
        self.step_count += 1
        print(f"\n[Step {self.step_count}] Action taken: {action}")
        
        # Calculate new helicopter position
        heli_x, heli_y = self.state['helicopter_coord']
        
        if action == 0: heli_y += 5      # Move down
        elif action == 1: heli_y -= 5    # Move up
        elif action == 2: heli_x -= 5    # Move left
        elif action == 3: heli_x += 5    # Move right
        elif action == 4: 
            print(f"Helitack performed at ({heli_x}, {heli_y})")
        
        # Clip coordinates
        heli_x = int(np.clip(heli_x, 10, 239))
        heli_y = int(np.clip(heli_y, 10, 159))
        
        # Create step message
        step_message = json.dumps({
            "action": str(action),
            "helicopter_coord": [heli_x, heli_y]
        })
        
        # Send step message and wait for response
        response = self._send_and_wait(step_message, timeout=2.0)
        
        # Update state based on response
        reward = -0.1  # Default reward
        done = False
        
        if response:
            try:
                # Update state with response data
                if 'cells' in response:
                    cells = response['cells']
                    self.state['cells'] = np.array(cells, dtype=np.int32)
                
                if 'helicopter_coord' in response:
                    self.state['helicopter_coord'] = np.array(response['helicopter_coord'], dtype=np.int32)
                
                if 'on_fire' in response:
                    self.state['on_fire'] = int(response['on_fire'])
                
                # Extract reward and done from response
                if 'reward' in response:
                    reward = float(response['reward'])
                
                if 'done' in response:
                    done = bool(response['done'])
            except Exception as e:
                print(f"❌ Error updating state: {e}")
        else:
            # No response, update state with calculated position
            self.state['helicopter_coord'] = np.array([heli_x, heli_y], dtype=np.int32)
        
        # Check for max steps
        if self.step_count >= 1000:
            done = True
        
        # Create observation
        observation = {
            'helicopter_coord': np.array(self.state['helicopter_coord'], dtype=np.int32),
            'cells': np.clip(np.array(self.state['cells'], dtype=np.int32), 0, 8),
            'on_fire': int(self.state['on_fire'])
        }
        
        print(f"State: {self._describe_state(observation)}")
        print(f"Reward: {reward}, Done: {done}")
        
        return observation, reward, done, False, {}
    
    def close(self):
        print("Closing environment...")
        
        # Stop the listener thread
        self.running = False
        if self.listener_thread.is_alive():
            self.listener_thread.join(timeout=2.0)
        
        # Send close message to React (don't wait for response)
        try:
            close_message = json.dumps({"action": "close"})
            self.websocket.send(close_message)
        except Exception as e:
            print(f"Error sending close message: {e}")
    
    def _describe_state(self, state):
        try:
            coords = state['helicopter_coord']
            fire = state['on_fire']
            cell_mean = np.mean(state['cells'])
            return f"Coord: {coords}, On fire: {fire}, Cell mean: {cell_mean:.2f}"
        except Exception as e:
            return f"Invalid state! Error: {e}, Raw state: {state}"