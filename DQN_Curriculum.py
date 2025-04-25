import gymnasium as gym
import time
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_hanoi.envs.hanoi_env11 import HanoiEnv

# === Curriculum Parameter ===
disk_count = 4  # ✅ Change manually for each stage (2 to 6)

# === Parameters ===
total_timesteps = 1_000_000
max_steps = 2000
eval_episodes = 5

# === Create Environment ===
def make_env():
    return Monitor(HanoiEnv(num_disks=disk_count, max_steps=max_steps, render_mode="human"))

vec_env = DummyVecEnv([make_env])

# === Create new model with the current environment ===
model = DQN(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=5e-3,
    buffer_size=1000,
    learning_starts=1000,
    batch_size=128,
    tau=0.95,
    gamma=0.995,
    train_freq=4,
    target_update_interval=1200,
    exploration_fraction=0.95,
    exploration_initial_eps=0.65,
    exploration_final_eps=0.005,
    verbose=1,
    tensorboard_log="./dqn_hanoi_tensorboard/"
)

# === Transfer weights from previous model if disk_count > 2 ===
if disk_count > 2:
    prev_model_path = f"dqn_hanoi_{disk_count - 1}_disks"
    prev_model = DQN.load(prev_model_path)

    # Load only matching layers (skip input layer if shape mismatches)
    pretrained_dict = prev_model.policy.state_dict()
    model_dict = model.policy.state_dict()
    matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(matched_dict)
    model.policy.load_state_dict(model_dict)
    print(f"🔁 Transferred {len(matched_dict)} layers from '{prev_model_path}'")

# === Train the Model ===
model.learn(total_timesteps=total_timesteps, tb_log_name=f"hanoi_{disk_count}_disks")
model.save(f"dqn_hanoi_{disk_count}_disks")
print(f"\n✅ Training complete. Model saved as 'dqn_hanoi_{disk_count}_disks'.")

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
env = vec_env.envs[0]  # Unwrap the actual env

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
