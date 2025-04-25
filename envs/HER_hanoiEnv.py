from gymnasium import Env
from gymnasium import spaces
import numpy as np
import random


class HanoiGoalEnv(Env):  # ✅ Standard Gym Env is fine for HER if it returns a Dict
    def __init__(self, num_disks=4, env_noise=0.0, min_penalty_start=None, penalty_step_interval=2, max_steps=1000):
        super().__init__()

        self.num_disks = num_disks                                    # 🔧 Number of disks
        self.env_noise = env_noise                                    # 🌪️ Action noise probability
        self.min_moves = 2 ** self.num_disks - 1                      # 🧮 Optimal solve steps
        self.min_penalty_start = min_penalty_start or self.min_moves
        self.penalty_step_interval = penalty_step_interval            # 🔁 Penalty ramp-up frequency
        self.max_episode_steps = max_steps                            # ⏱️ Max steps per episode

        # === Spaces ===
        obs_space = spaces.Box(low=0, high=2, shape=(self.num_disks,), dtype=np.int32)
        self.observation_space = spaces.Dict({
            "observation": obs_space,
            "achieved_goal": obs_space,
            "desired_goal": obs_space
        })
        self.valid_actions = [0, 1, 2, 3]
        self.action_space = spaces.Discrete(len(self.valid_actions))  # 🎮 Valid adjacent moves
        self.goal_state = tuple([2] * self.num_disks)                 # 🎯 All disks on right peg
        self.current_state = None
        self.ACTION_LOOKUP = {
            0: (0, 1),
            1: (1, 0),
            2: (1, 2),
            3: (2, 1),
        }

        self.move_count = 0
        self.solved_counter = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.towers = [[i for i in range(self.num_disks)], [], []]
        self.current_state = self._get_state()
        self.move_count = 0
        obs = {
            "observation": np.array(self.current_state, dtype=np.int32),
            "achieved_goal": np.array(self.current_state, dtype=np.int32),
            "desired_goal": np.array(self.goal_state, dtype=np.int32)
        }
        return obs, {}  # ✅ Return (observation, info)


    def step(self, action):
        info = {}

        # Map action to move
        move = self.ACTION_LOOKUP[action]

        # If invalid, replace with a random valid action
        if not self.move_allowed(move):
            valid_actions = [a for a in self.valid_actions if self.move_allowed(self.ACTION_LOOKUP[a])]
            if valid_actions:
                action = random.choice(valid_actions)
                move = self.ACTION_LOOKUP[action]
            else:
                # No valid actions (should not happen in Hanoi), just return current state
                return self._get_obs(), 0, False, False, info

        # Apply move
        top_disk = min(self.disks_on_peg(move[0]))
        next_state = list(self.current_state)
        next_state[top_disk] = move[1]
        self.current_state = tuple(next_state)

        self.move_count += 1

        # === Reward Logic ===
        if self.current_state == self.goal_state:
            reward = 100
            reward += max(0, round(500 / (1 + 0.22 * max(0, self.move_count - self.min_moves)), 2))
            self.solved_counter += 1
        else:
            reward = 0

        done = self.current_state == self.goal_state
        truncated = self.move_count >= self.max_episode_steps
        return self._get_obs(), reward, done, truncated, info



    def _get_obs(self):
        state_array = np.array(self.current_state, dtype=np.int32)
        goal_array = np.array(self.goal_state, dtype=np.int32)
        return {
            "observation": state_array,
            "achieved_goal": state_array,
            "desired_goal": goal_array
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        # If input is batch, return reward for each sample
        reward = -(achieved_goal != desired_goal).any(axis=-1).astype(np.float32)
        return reward


    def _get_state(self):
        state = [0] * self.num_disks
        for peg_index, tower in enumerate(self.towers):
            for disk in tower:
                state[disk] = peg_index
        return tuple(state)

    def disks_on_peg(self, peg):
        return [disk for disk in range(self.num_disks) if self.current_state[disk] == peg]

    def move_allowed(self, move):
        from_peg = self.disks_on_peg(move[0])
        to_peg = self.disks_on_peg(move[1])
        if not from_peg:
            return False
        moving_disk = min(from_peg)
        return (not to_peg) or (min(to_peg) > moving_disk)  # ✅ Disallow placing large over small

    def render(self, mode='human'):
        pegs = {0: [], 1: [], 2: []}
        for i, peg in enumerate(self.current_state):
            pegs[peg].append(i)
        for k in pegs:
            pegs[k].sort(reverse=True)

        max_height = self.num_disks
        peg_width = self.num_disks * 2 + 1
        print("\n🏗️ TOWERS OF HANOI")
        for level in range(max_height - 1, -1, -1):
            row = ""
            for peg in range(3):
                if level < len(pegs[peg]):
                    disk = pegs[peg][level]
                    disk_str = "=" * (2 * (disk + 1) - 1)
                    pad = " " * (self.num_disks - disk)
                    row += f"{pad}{disk_str}{pad}".center(peg_width) + "   "
                else:
                    row += " " * self.num_disks + "|" + " " * self.num_disks + "   "
            print(row)
        print("=" * (peg_width * 3 + 6))

