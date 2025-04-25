import time
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_hanoi.envs.hanoi_env11 import HanoiEnv

# === Settings ===
num_disks = 5
total_timesteps = 2_000_000
max_steps = 6000
eval_episodes = 5

# === Environment ===
def make_env():
    return Monitor(HanoiEnv(num_disks=num_disks, max_steps=max_steps, render_mode="human"))

vec_env = DummyVecEnv([make_env])

# === DQN Agent ===
model = DQN(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=6e-5,             # ⬅️ Slower learning to handle longer episodes and sparse rewards
    buffer_size=36_000,             # ⬅️ Larger buffer to retain longer, varied episodes
    learning_starts=1_200,          # ⬅️ Slight delay to avoid learning from noise
    batch_size=96,                  # ⬅️ Slightly larger batch for stable updates
    tau=1.0,                        # ⬅️ Not used in DQN but required
    gamma=0.985,                    # ⬅️ Favor long-term planning even more
    train_freq=1,                   # ⬅️ Train after every step
    target_update_interval=600,     # ⬅️ Slightly more frequent target sync
    exploration_fraction=0.92,      # ⬅️ More time spent exploring
    exploration_initial_eps=0.88,   # ⬅️ Explore a bit more aggressively early on
    exploration_final_eps=0.0092,   # ⬅️ Still allow rare exploration late
    verbose=1,
    tensorboard_log="./dqn_hanoi_tensorboard/"
)

# === Train ===
model.learn(total_timesteps=total_timesteps)
model.save(f"dqn_hanoi_{num_disks}_disks")
print(f"\n✅ Training complete. Model saved.\n")

# === Evaluate ===
success_count = 0
for ep in range(eval_episodes):
    obs = vec_env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += reward[0]
        steps += 1

    print(f"Episode {ep+1}: Reward = {total_reward:.1f}, Steps = {steps}")
    if reward[0] >= 100:
        success_count += 1

print(f"\n✅ Success rate: {success_count}/{eval_episodes}")

# === Final Run Visualization ===
obs = vec_env.reset()
done = False
steps = 0
print("\n🎬 Final Render:\n")
while not done and steps < max_steps:
    print(f"Step {steps}")                     
    vec_env.envs[0].render()
    time.sleep(0.3)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = vec_env.step(action)
    done = done[0]
    steps += 1

print("\n🏁 Final run complete.")

