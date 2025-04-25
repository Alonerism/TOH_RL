import gymnasium as gym
import time
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_hanoi.envs.hanoi_env11 import HanoiEnv

# === Parameters ===
disk_count = 5  # ✅ Set to the current disk level you want to resume
total_timesteps = 2_000_000
max_steps = 3000
eval_episodes = 5
prev_model_path = f"dqn_hanoi_{disk_count}_disks"  # 🔁 Previously saved model
new_model_path = f"dqn_hanoi_{disk_count}_disks_R2"  # 💾 New save name

# === Create Environment ===
def make_env():
    return Monitor(HanoiEnv(num_disks=disk_count, max_steps=max_steps, render_mode="human"))

vec_env = DummyVecEnv([make_env])

# === Load previous model and resume training ===
model = DQN.load(prev_model_path, env=vec_env, tensorboard_log="./dqn_hanoi_tensorboard/", verbose=1)

# === Resume Training ===
model.learn(total_timesteps=total_timesteps, tb_log_name=f"hanoi_{disk_count}_disks_R2", reset_num_timesteps=False)
model.save(new_model_path)
print(f"\n✅ Continued training complete. Model saved as '{new_model_path}'.")

# === Evaluate Trained Agent ===
success_count = 0
print("\n🔍 Starting evaluation...\n")
for ep in range(eval_episodes):
    obs = vec_env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        step_count += 1

    print(f"🧪 Episode {ep + 1}: Reward = {total_reward:.1f}, Steps = {step_count}")
    if reward[0] >= 100:
        success_count += 1

print("\n✅ Evaluation complete.")
print(f"🏁 Episodes: {eval_episodes}")
print(f"🎯 Successful solves: {success_count}")
print(f"📈 Success rate: {success_count / eval_episodes * 100:.1f}%")

# === Visualize Final Run ===
print("\n🎬 Final render episode...\n")
obs = vec_env.reset()
done = False
total_reward = 0
step_count = 0
env = vec_env.envs[0]

while not done and step_count < max_steps:
    env.render()
    time.sleep(0.3)
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

    reward = reward[0]
    done = done[0]
    total_reward += reward
    step_count += 1
    print(f"Step {step_count}: Action {action}")

print(f"\n🎯 Final reward: {total_reward:.1f}, Steps: {step_count}")
