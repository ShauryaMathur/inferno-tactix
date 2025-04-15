from wildfireenv import FireSimulationGymEnv
from infernoenv import FireEnvSyncWrapper

from stable_baselines3 import DQN,PPO
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
    # model = DQN("MlpPolicy", env, verbose=1)
    model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gae_lambda=0.95,
    gamma=0.99,
    ent_coef=0.01,
)
    model.learn(total_timesteps=1000)

    # Save model
    model.save("ppo_firefighter")

if __name__ == "__main__":
    main()