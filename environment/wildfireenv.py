import numpy as np
import asyncio
import websockets
import json
import gym
from gym import spaces
import plotly.graph_objects as go
import numpy as np


# Custom JSON encoder to handle NumPy types
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

    def plotHeatmap(self):
        
        fig = go.Figure(data=go.Heatmap(
                        z=self.state['cells'],
                        colorscale='viridis',
                        zmin=self.state['cells'].min(),
                        zmax=self.state['cells'].max(),
                    ))

        fig.update_layout(
            title='Elevation Heatmap',
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            coloraxis_colorbar_title='Intensity',
            width=800,
            height=600,
        )

        # Display the heatmap
        fig.show()
    def __init__(self):
        super(FireSimulationGymEnv, self).__init__()

        # Define action space as discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Initial state setup
        self.state = {
            'helicopter_coord': (0, 0),  # Initial coordinates (x, y)
            'cells': np.zeros((240, 160))  # Placeholder for the grid of ignition times
        }
        
        self.websocket = None  # Will be assigned when connecting

    async def reset(self):
        """
        Resets the environment and sends a reset message to the client.
        """
        # Reset the environment state
        self.state = {
            'helicopter_coord': (195, 90),
            'cells': np.zeros((240, 160))  # Reset the grid of ignition times
        }
        
        # Send reset message to client (websocket)
        if self.websocket:
            reset_message = json.dumps({"action": "reset"})
            asyncio.create_task(self.websocket.send(reset_message))
            print("Reset message sent to frontend")
        new_state = await self.wait_for_client_update()
        self.state['cells'] = new_state['cells']

        # Return the initial state (observation)
        return self.state

    async def step(self, action, stepcount):
        # Initialize reward to 0 by default
        reward = 0

        # Get current helicopter coordinates
        heli_x, heli_y = self.state['helicopter_coord']

        # Map discrete actions to behaviors
        if action == 0:  # 'U' - Move Up
            heli_y += 5
        elif action == 1:  # 'D' - Move Down
            heli_y -= 5
        elif action == 2:  # 'L' - Move Left
            heli_x -= 5
        elif action == 3:  # 'R' - Move Right
            heli_x += 5
        elif action == 4:  # 'Helitack' - Perform heli attack
            print(f"Helitack performed at ({heli_x}, {heli_y})")
            reward = 10  # Reward for performing heli attack

        # Clip helicopter coordinates to stay within the grid limits (240x160)
        heli_x = np.clip(heli_x, 10, 239)  # x should be between 0 and 239
        heli_y = np.clip(heli_y, 10, 159)  # y should be between 0 and 159

        # Convert NumPy values to Python native types
        heli_x = int(heli_x)
        heli_y = int(heli_y)

        # Update helicopter coordinates in the state
        self.state['helicopter_coord'] = (heli_x, heli_y)

        # Prepare action message to send to the client (physics engine)
        try:
            action_message = json.dumps({
                "action": str(action),
                "helicopter_coord": list(self.state['helicopter_coord'])
            }, cls=NumpyEncoder)
        except Exception as e:
            print(f"Error creating action message: {e}")
            return self.state, reward, False, {}

        # Send action to client (websocket)
        if self.websocket:
            await self.websocket.send(action_message)
            print(f"Step message sent to frontend: {action_message}")

        # Wait for the client to send back the updated cells
        self.state = await self.wait_for_client_update()
        print(self.state)

        # print(self.state['cells'], type(self.state['cells']))
        
        if stepcount in [450]:
            self.plotHeatmap()

        # Return the combined state, reward, and done flag
        
        return self.state, reward, self.state['done'], {}

    async def wait_for_client_update(self):
        """
        This function asynchronously waits for the client to send back the updated cells data.
        """
        try:
            message = await self.websocket.recv()  # Waits for a message from the client
            # print(f"Received message from client: {message}")
            
            # Parse the message from the client
            client_state = json.loads(message)
            print(client_state)
            # Extract the updated 'cells' from the client state
            cells_list = json.loads(client_state['cells'])

            # if 'cells' in client_state:
                
            # else:
            #     print("Error: No 'cells' data received from client.")
            #     return {
            #         'cells': np.zeros((240, 160))  # Return default empty cells if no valid data
            #     }

            new_state = {
                    **self.state,
                    **client_state,
                    'cells': np.array(cells_list) 
                    }
            return new_state

        except Exception as e:
            print(f"Error while receiving data from client: {e}")
            return {
                'cells': np.zeros((240, 160))  # Return default empty cells in case of an error
            }


async def websocket_handler(websocket, path):
    # Create Gym environment
    env = FireSimulationGymEnv()
    env.websocket = websocket  # Assign the WebSocket to the environment

    try:
        
        for episodes in range(5):
            # Reset environment when starting
            initial_state = await env.reset()
            for step in range(500):  # Run 100 steps
                # Automatically pick a discrete action from the action space
                action = env.action_space.sample()  # Random action from the action space
                print(f"Chosen action at step {step}: {action}")

                # Perform the environment step (sending action to the client)
                new_state, reward, done, _ = await env.step(action,step)

                # # Send updated state, reward, and done flag back to the client
                # # Convert NumPy arrays and types to JSON-serializable format
                # message = {
                #     "state": {
                #         "helicopter_coord": list(new_state['helicopter_coord']),
                #         "cells": new_state['cells'].tolist()  # Convert NumPy array to list
                #     },
                #     "reward": float(reward),  # Convert to Python native float
                #     "done": bool(done)  # Convert to Python native boolean
                # }
                
                # await websocket.send(json.dumps(message, cls=NumpyEncoder))

                if done:
                    print("Episode finished!")
                    break
            

    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed")

# Start the WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8765)

print("WebSocket server started on ws://localhost:8765")

# Run the WebSocket server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()