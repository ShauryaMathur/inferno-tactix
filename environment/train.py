from wildfireenv import FireSimulationGymEnv
from infernoenv import FireEnvSyncWrapper

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

def main():
    print("Starting training...")
    # Create the async env
    async_env = FireSimulationGymEnv()
    print("Created async env",async_env)
    # Wrap it
    env = FireEnvSyncWrapper(async_env)
    print("Wrapped env",env)
    # Optional sanity check
    check_env(env, warn=True)
    print("Checked env")
    # Train the agent
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)

    # Save model
    model.save("ppo_firefighter")

if __name__ == "__main__":
    main()