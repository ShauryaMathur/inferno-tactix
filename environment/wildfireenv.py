import numpy as np
import asyncio
import websockets
import json
import gym
from gym import spaces
import random

class FireSimulationGymEnv(gym.Env):
    def __init__(self):
        super(FireSimulationGymEnv, self).__init__()

        # Define action space as discrete actions
        # 0: 'U' (Up), 1: 'D' (Down), 2: 'L' (Left), 3: 'R' (Right), 4: 'Helitack'
        self.action_space = spaces.Discrete(5)
        
        # Initial state setup
        self.state = {
            'helicopter_coord': (0, 0),  # Initial coordinates (x, y)
            'cells': np.zeros((240, 160))  # Placeholder for the grid of ignition times
        }
        
        self.websocket = None  # Will be assigned when connecting

    def reset(self):
        """
        Resets the environment and sends a reset message to the client.
        """
        # Reset the environment state
        self.state = {
            'helicopter_coord': (0, 0),
            'cells': np.zeros((240, 160))  # Reset the grid of ignition times
        }
        
        # Send reset message to client (websocket)
        if self.websocket:
            reset_message = json.dumps({"action": "reset"})
            asyncio.create_task(self.websocket.send(reset_message))
            print("Reset message sent to frontend")

        # Return the initial state (observation)
        return self.state

    def step(self, action):
        """
        Perform one step in the environment given the action, then return the new state and reward.
        """
        # Get current helicopter coordinates
        heli_x, heli_y = self.state['helicopter_coord']
        
        # Define response dictionary
        response = {}
        reward = 0  # Default reward (can be modified later based on environment behavior)
        done = False  # Whether the episode is done
        
        # Map discrete actions to behaviors
        if action == 0:  # 'U' - Move Up
            heli_y += 1
        elif action == 1:  # 'D' - Move Down
            heli_y -= 1
        elif action == 2:  # 'L' - Move Left
            heli_x -= 1
        elif action == 3:  # 'R' - Move Right
            heli_x += 1
        elif action == 4:  # 'Helitack' - Perform heli attack
            response = {"action": "Helitack", "heli_coord": (heli_x, heli_y)}
            reward = 10  # Example reward for performing the heli attack

        # Clip helicopter coordinates to stay within the grid limits (240x160)
        heli_x = np.clip(heli_x, 0, 239)  # x should be between 0 and 239
        heli_y = np.clip(heli_y, 0, 159)  # y should be between 0 and 159
        
        # Update helicopter coordinates in the state
        self.state['helicopter_coord'] = (heli_x, heli_y)
        
        # If it's not a heli attack, update response with new state
        if action != 4:
            response = {"state": self.state}
        
        # Send action and updated state to client (websocket)
        if self.websocket:
            asyncio.create_task(self.websocket.send(json.dumps(response)))
            print(f"Step message sent to frontend: {action}, heli coordinates: {(heli_x, heli_y)}")

        # Example condition to end the episode: if the helicopter goes out of bounds
        if heli_x < 0 or heli_x >= 240 or heli_y < 0 or heli_y >= 160:
            done = True  # Episode ends if helicopter moves out of bounds

        # Return the next state, reward, and whether the episode is done
        return self.state, reward, done, {}

    def set_state_from_client(self, client_state):
        """
        Set the state of the environment from the client-provided state.
        """
        self.state['cells'] = np.array(client_state['cells'])
        self.state['helicopter_coord'] = tuple(client_state['helicopter_coord'])

async def websocket_handler(websocket, path):
    # Create Gym environment
    env = FireSimulationGymEnv()
    env.websocket = websocket  # Assign the WebSocket to the environment

    try:
        # Reset environment when starting
        initial_state = env.reset()

        # Send initial state to the React frontend
        await websocket.send(json.dumps({"state": initial_state}))

        # Process incoming messages from React frontend
        async for message in websocket:
            # Parse incoming action (or reset)
            action_data = json.loads(message)
            print(f"Received action from frontend: {action_data}")

            # Handle reset action
            if action_data.get("action") == "reset":
                initial_state = env.reset()
                await websocket.send(json.dumps({"state": initial_state}))
            else:
                # Set environment state from client data (if provided)
                if "state" in action_data:
                    client_state = action_data["state"]
                    env.set_state_from_client(client_state)
                
                # Perform the next step in the environment
                action = action_data.get("action")
                new_state, reward, done, info = env.step(action)
                await websocket.send(json.dumps({"state": new_state, "reward": reward, "done": done}))

    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed")


# Start the WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8765)

print("WebSocket server started on ws://localhost:8765")

# Run the WebSocket server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()