import asyncio
import websockets
import json
import gym

class FireSimulationGymEnv(gym.Env):
    def __init__(self):
        super(FireSimulationGymEnv, self).__init__()
        # Initialize your Gym environment here
        self.state = None  # Placeholder for the simulation state

    async def update_simulation(self, action):
        # Process the action from the RL agent and update the simulation
        # This is where you handle your simulation state update logic
        self.state = action  # Just an example; use your actual simulation logic here
        return self.state

async def websocket_handler(websocket, path):
    # Create your Gym environment
    env = FireSimulationGymEnv()

    try:
        async for message in websocket:
            # Parse action received from the client (React frontend)
            action = json.loads(message)
            print(f"Received action: {action}")

            # Update the simulation in the Gym environment
            new_state = await env.update_simulation(action)

            # Send the new state back to the client (React frontend)
            await websocket.send(json.dumps({"state": new_state}))

    except websockets.exceptions.ConnectionClosedOK:
        print("Connection closed")

# Start the WebSocket server
start_server = websockets.serve(websocket_handler, "localhost", 8765)

print("WebSocket server started on ws://localhost:8765")

# Run the WebSocket server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

