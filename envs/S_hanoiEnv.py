import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class HanoiEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, num_disks=4, max_steps=3000, render_mode=None):
        super().__init__()
        self.num_disks = num_disks
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.min_moves = 2 ** self.num_disks - 1

        self.ACTION_LOOKUP = {
            0: (0, 1),
            1: (1, 0),
            2: (1, 2),
            3: (2, 1),
            4: (0, 2),  # ✅ Allow move from peg 0 to peg 2
            5: (2, 0),  # ✅ Allow move from peg 2 to peg 0
        }

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.num_disks,), dtype=np.int32)
        self.goal_state = self.num_disks * (2,)
        self.current_state = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.towers = [[i for i in range(self.num_disks)], [], []]
        self.current_state = self._get_state()
        self.move_count = 0
        self.reward_flags = set()  # ✅ Reset milestone rewards
        return self.current_state, {}

    def step(self, action):
        action = int(action)
        reward = 0  # Initialize reward

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            action = random.choice(valid_actions)

        move = self.ACTION_LOOKUP[action]
        top_disk = min(self.disks_on_peg(move[0]))
        next_state = list(self.current_state)
        next_state[top_disk] = move[1]
        self.current_state = tuple(next_state)
        self.move_count += 1

        # Final reward for solving the puzzle
        if self.current_state == self.goal_state:
            reward = 100
            reward += max(0, round(500 / (1 + 0.1 * max(0, self.move_count - self.min_moves)), 2))

        terminated = self.current_state == self.goal_state
        truncated = self.move_count >= self.max_steps
        return self.current_state, reward, terminated, truncated, {}




    def get_valid_actions(self):
        return [a for a, m in self.ACTION_LOOKUP.items() if self.move_allowed(m)]

    def disks_on_peg(self, peg):
        return [d for d in range(self.num_disks) if self.current_state[d] == peg]

    def move_allowed(self, move):
        from_peg = self.disks_on_peg(move[0])
        to_peg = self.disks_on_peg(move[1])
        if not from_peg:
            return False
        moving_disk = min(from_peg)
        return (not to_peg) or (min(to_peg) > moving_disk)

    def _get_state(self):
        state = [0] * self.num_disks
        for peg_index, tower in enumerate(self.towers):
            for disk in tower:
                state[disk] = peg_index
        return tuple(state)

    def render(self):
        if self.render_mode != "human":
            return

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
