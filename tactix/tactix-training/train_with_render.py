
import signal
import sys
import traceback
import os
import torch
import asyncio
import threading
import websockets
import queue
import time
import json

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import get_schedule_fn

from FireEnv import FireEnvSync
from FeatureExtractor import FireEnvLSTMPolicy,FireEnvLSTMCNN
from callbacks import LearningRateScheduleCallback, MemoryCleanupCallback, RewardLoggingCallback

from training_utils import clear_gpu_memory,get_optimal_device

# Global variables for WebSocket connection
client_websocket = None
message_queue = queue.Queue(maxsize=100)
stop_event = threading.Event()
model = None

# WebSocket server thread function
def websocket_server_thread():
    """Run the WebSocket server in a separate thread"""
    async def handler(websocket):
        global client_websocket
        
        if client_websocket is not None:
            await client_websocket.close()
        
        print("🔥 React frontend connected!")
        client_websocket = websocket
        
        try:
            # Handle ping messages
            async def handle_ping():
                try:
                    await websocket.send(json.dumps({"type": "pong"}))
                    print("📤 Sent pong response")
                except Exception as e:
                    print(f"Error sending pong: {e}")
            
            # Process incoming messages
            async for message in websocket:
                # Handle ping specially
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        print("🏓 Received ping, sending pong")
                        await handle_ping()
                        continue
                except Exception:
                    pass
                
                # Put other messages in the queue for the environment
                try:
                    message_queue.put(message, block=False)
                except queue.Full:
                    # If queue is full, remove oldest message and add new one
                    try:
                        message_queue.get_nowait()
                        message_queue.put(message)
                    except:
                        pass
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            print("React frontend disconnected.")
            if client_websocket == websocket:
                client_websocket = None
    
    async def run_server():
        host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
        port = int(os.environ.get("WEBSOCKET_PORT", "8765"))

        server = await websockets.serve(
            handler, 
            host, 
            port, 
            max_size=None, 
            compression='deflate'
        )
        print(f"🟢 WebSocket server started on {host}:{port}")
        
        # Keep server running until stop event is set
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        
        server.close()
        await server.wait_closed()
        print("🔴 WebSocket server stopped")
    
    # Set up event loop in this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run the server
    try:
        loop.run_until_complete(run_server())
    except Exception as e:
        print(f"Error in WebSocket server: {e}")
    finally:
        loop.close()

print('EXECUTING TRAINING SCRIPT...')

# Directory for logs
logdir = "runs/ppo_firefighter/"
os.makedirs(logdir, exist_ok=True)
logFileName = 'run1'
TRAINING_TIMESTEPS = 5000

USE_TRAINED_AGENT = False

# Ensure the save path exists
save_path = os.environ.get("MODEL_DIR", ".")
SAVED_AGENT_NAME = "ppo_firefighter_improved"
SAVED_VEC_NORMALIZATION_FILENAME = "vecnormalize_improved.pkl"
MODEL_FILE = os.path.join(save_path, SAVED_AGENT_NAME + ".zip")

# Set PyTorch thread limits to avoid oversubscription
# torch.set_num_threads(4)  # Limit CPU threads used by PyTorch
# if torch.cuda.is_available():
#     torch.cuda.set_per_process_memory_fraction(0.9)  # Limit GPU memory usage

model = None  # Global model reference for signal handler
vec_env = None

# Signal handler for graceful termination
def signal_handler(sig, frame):
    print("\n⚠️ Received termination signal. Cleaning up...")
    if model is not None:
        try:
            print("💾 Saving model...")
            model.save(os.path.join(save_path, SAVED_AGENT_NAME))

            print(">>> MODEL_DIR:", os.environ.get("MODEL_DIR"))
            print(">>> cwd     :", os.getcwd())

            print("✅ Model saved as ppo_firefighter")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
        
    sys.exit(0)

# # Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

class LSTMResetCallback(BaseCallback):
    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        extractor = self.model.policy.features_extractor
        
        done_indices = [i for i, done in enumerate(dones) if done]
        if done_indices and hasattr(extractor, "reset_hidden"):
            extractor.reset_hidden(env_indices=done_indices)
        
        return True
    
def make_env(rank=0, base_seed=1000):
    def _init():
        env = FireEnvSync(env_id=rank)
        env = Monitor(env,logdir)
        print(f"[ENV {rank}]")
        # check_env(env)
        # seed = base_seed + rank
        # env.seed(seed)  # Seed right here
        
        return env
    return _init

# Main function
def main():
    global model

    # Create vectorized environment
    env = make_env(0)

    device, is_gpu = get_optimal_device()
    
    server_thread = None
    
    # Start WebSocket server
    print("Starting WebSocket server...")
    server_thread = threading.Thread(target=websocket_server_thread, daemon=True)
    server_thread.start()
    
    # Wait for WebSocket server to start
    print("Waiting for WebSocket server to start...")
    time.sleep(2)
    
    # Wait for React to connect (with a longer timeout for simulation)
    print("Waiting for React to connect...")
    timeout = 500
    start_time = time.time()
    
    while client_websocket is None:
        time.sleep(0.5)
        if time.time() - start_time > timeout:
            print("Timeout waiting for React to connect")
            return
    
    print(f"✅ React client connected: {id(client_websocket)}")

    print("🔍 Checking for existing model...")
    if USE_TRAINED_AGENT and os.path.isfile(MODEL_FILE):
        print("🔄 Found existing model, loading for continued training...")
        model = PPO.load(MODEL_FILE, device=device, tensorboard_log=logdir)
        model.set_env(env)
        model.lr_schedule = get_schedule_fn(3e-4)
        model.clip_range = get_schedule_fn(0.2)
        # model.lr_schedule = lambda _: 5e-5
        model.policy.optimizer.param_groups[0]['lr'] = 5e-5
        # Use a function (lambda) for clip_range
        # model.clip_range = lambda _: 0.3

        # These two must also be callables if you want dynamic behavior
        model.vf_coef =  0.5
        model.gamma = 0.995

        print("✅ Model loaded successfully!")
    else:
        print("🆕 No existing model found, initializing new model...")
        model = PPO(
            FireEnvLSTMPolicy,
            env,
            n_steps=128,
            batch_size=64,
            n_epochs=3,
            learning_rate= 3e-4,
            clip_range= 0.1,
            gamma=0.95,
            gae_lambda=0.9,
            ent_coef=0.2,
            vf_coef=0.4,
            max_grad_norm=1.0,
            target_kl=0.03,
            verbose=1,
            device=device,
            tensorboard_log=logdir,
            policy_kwargs = dict(
                features_extractor_class=FireEnvLSTMCNN,
                features_extractor_kwargs=dict(features_dim=512)
            )
        )
        print("✅ New model initialized successfully!")
        
    if device == 'mps':
        if not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available, falling back to CPU")
            device = 'cpu'
        else:
            print("Configuring MPS-specific optimizations for Apple Silicon")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            model.batch_size = 64
            model.n_steps = 128
            print(f"MPS-optimized batch configuration: batch_size={model.batch_size}, n_steps={model.n_steps}")

    reward_callback = RewardLoggingCallback()
    # lr_callback = LearningRateScheduleCallback(lr_schedule)
    memory_callback = MemoryCleanupCallback(cleanup_freq=5000, verbose=1)
    lstm_reset_callback = LSTMResetCallback()

    callbacks = CallbackList([
        reward_callback, 
        lstm_reset_callback,
        memory_callback
    ])

    os.makedirs(logdir, exist_ok=True)
    clear_gpu_memory()

    if device != 'cpu':
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 precision for faster training")
        torch.backends.cudnn.benchmark = True

    print("🚀 Starting training...")
    try:
        model.learn(
            total_timesteps=TRAINING_TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=logFileName,
            callback=callbacks
        )
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        traceback.print_exc()

    finally:
        if model is not None:
            try:
                print("💾 Saving model...")
                model.save(os.path.join(save_path, SAVED_AGENT_NAME))
                vec_env.save(SAVED_VEC_NORMALIZATION_FILENAME)
                print("✅ Model and normalization state saved.")
            except Exception as e:
                print(f"❌ Error saving model: {e}")

        print("👋 Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()