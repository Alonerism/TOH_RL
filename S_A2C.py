import gymnasium as gym
import time
from gymnasium.wrappers import RecordEpisodeStatistics as Monitor
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_hanoi.envs.S_hanoiEnv import HanoiEnv

# === Curriculum Parameter ===
disk_count = 4

# === Parameters ===
total_timesteps = 1_000_000
max_steps = 5000
eval_episodes = 5
n_envs = 4

def make_env():
    def _init():
        return Monitor(HanoiEnv(num_disks=disk_count, max_steps=max_steps, render_mode="human"))
    return _init

if __name__ == '__main__':
    vec_env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=5e-3,
        gamma=0.995,
        n_steps=5,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./a2c_hanoi_tensorboard/"
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(f"a2c_hanoi_{disk_count}_disks")
    print(f"\n✅ Training complete. Model saved as 'a2c_hanoi_{disk_count}_disks'.")

    # === Evaluation ===
    eval_env = Monitor(HanoiEnv(num_disks=disk_count, max_steps=max_steps, render_mode="human"))
    success_count = 0
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0
        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
        print(f"🧪 Episode {ep + 1}: Reward = {total_reward:.1f}, Steps = {step_count}")
        if total_reward >= 100:
            success_count += 1

    print(f"\n✅ Evaluation complete. Success rate: {success_count}/{eval_episodes}")

    # === Final Render ===
    obs, _ = eval_env.reset()
    done = False
    print("\n🎬 Final Render:")
    while not done:
        eval_env.render()
        time.sleep(0.3)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
