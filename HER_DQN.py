import gymnasium as gym
import time
from gym_hanoi.envs.HER_hanoiEnv import HanoiGoalEnv
from stable_baselines3 import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env

# === Parameters ===
num_disks = 5
total_timesteps = 1_000_000
eval_episodes = 5
max_steps = 3000  # 🔁 Ensure this propagates to the env

# === Create HER-compatible Vectorized Environment ===
def make_env():
    return HanoiGoalEnv(num_disks=num_disks, max_steps=max_steps)

vec_env = make_vec_env(make_env, n_envs=1)

# === Initialize HER-enabled DQN Agent ===
model = DQN(
    policy="MultiInputPolicy",
    env=vec_env,
    learning_rate=5e-3,
    buffer_size=50_000,
    learning_starts=6000,
    batch_size=128,
    tau=0.9,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.9,
    exploration_initial_eps=0.6,
    exploration_final_eps=0.01,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    verbose=1,
    tensorboard_log="./dqn_hanoi_tensorboard/"
)

# === Train the Model ===
model.learn(total_timesteps=total_timesteps, tb_log_name="her_run")
model.save("dqn_hanoi_her")
print("\n✅ Training complete. Model saved as 'dqn_hanoi_her'.")

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

while not done and step_count < max_steps:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    total_reward += reward[0]
    vec_env.envs[0].render()  # 👈 Print tower visualization
    time.sleep(0.3)
    step_count += 1

print(f"\n🎯 Final reward: {total_reward:.1f}, Steps: {step_count}")

