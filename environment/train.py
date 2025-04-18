import asyncio
from ws_server import start_ws_server, get_client_ws
from wildfireenv import FireSimulationGymEnv
from infernoenv import FireEnvSyncWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

msg_queue = asyncio.Queue()


async def main():
    # 1. Start WebSocket server (React will connect here)
    await start_ws_server(msg_queue)
    print("🧠 Waiting for React to connect...")
    websocket = await get_client_ws(msg_queue)
    print(f"🧩 [train.py] Got websocket id: {id(websocket)}")

   # Wait for the WebSocket to be ready
    if websocket and websocket.open:
        print("WebSocket is already connected!")
    else:
        while websocket is None or not websocket.open:
            print("Waiting for WebSocket connection...")
            await asyncio.sleep(0.5)


    # 2. Create env with shared websocket
    print("🔄 Creating asynchronous environment...", flush=True)
    loop = asyncio.get_running_loop()
    
    print("Loop from train.py:", loop)
    await asyncio.sleep(0.5)

    
    async_env = FireSimulationGymEnv(websocket=websocket, loop=loop,msg_queue=msg_queue)
    await async_env._init_async_fields()

    env = FireEnvSyncWrapper(async_env)
    
    # Add explicit flush after each print to ensure output is visible
    print("🧪 Beginning environment check...", flush=True)
    print("WebSocket is open:", websocket.open)

    # print("🚨 Manually calling reset() for debugging")
    # observation, info = await env.reset(seed=0)  # Call reset() directly
    # print("✅ Reset completed:", observation)

    check_env(env)
    print("✅ Environment check completed successfully!", flush=True)
    
    print("🎉 Environment created and verified.", flush=True)
    
    # 3. Train PPO agent
    print("🚀 Starting training...", flush=True)
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("ppo_firefighter")
    print("✅ Training complete and model saved.", flush=True)

if __name__ == "__main__":
    asyncio.run(main())