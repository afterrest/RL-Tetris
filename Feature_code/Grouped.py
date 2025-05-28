import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np

from rl_tetris.envs.tetris import Tetris


class GroupedWrapper(gym.Wrapper):
    def __init__(self, env: Tetris, observation_wrapper=None):
        super().__init__(env)

        self.observation_wrapper = observation_wrapper

        self.action_space = Discrete((env.unwrapped.width) * 4)

        observation_space = {
            "boards": gym.spaces.Box(
                low=0,
                high=len(env.unwrapped.PIECES),
                shape=(self.action_space.n, env.unwrapped.height,
                       env.unwrapped.width),
                dtype=np.uint8,
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.action_space.n,),
                dtype=np.int8,
            ),
        }

        # observation_wrapper가 있는 경우 features 추가
        if observation_wrapper:
            observation_space["features"] = gym.spaces.Box(
                low=0,
                high=env.unwrapped.height * env.unwrapped.width,
                shape=(self.action_space.n, 8),
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_space)

    def encode_action(self, x, r):
        return x*4 + r

    def decode_action(self, action):
        return divmod(action, 4)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        grouped_obs = self.observation(obs)

        info["board"] = obs["board"]
        info["action_mapping"] = np.where(grouped_obs["action_mask"] == 1)[0]

        if wrapper := self.observation_wrapper:
            grouped_obs["features"] = wrapper.observation(grouped_obs)
            info["initial_feature"] = wrapper.extract_board_features(info["board"])


        return grouped_obs, info

    def step(self, action):
        x, r = self.decode_action(action)

        new_piece = [r[:] for r in self.env.unwrapped.piece]
        for _ in range(r):
            new_piece = self.env.unwrapped.get_rotated_piece(new_piece)

        self.env.unwrapped.x = x
        self.env.unwrapped.piece = new_piece

        obs, reward, done, truncted, info = self.env.step(
            self.env.unwrapped.actions.hard_drop
        )

        grouped_obs = self.observation(obs)

        if wrapper := self.observation_wrapper:
            grouped_obs["features"] = wrapper.observation(grouped_obs)

        info["board"] = obs["board"]
        info["action_mapping"] = np.where(grouped_obs["action_mask"] == 1)[0]

        return grouped_obs, reward, done, truncted, info

    def observation(self, observation):
        """현재 상태에서 가능한 모든 열(x)에서 가능한 모든 회전(r)에 대한 다음 상태를 반환하는 메서드"""

        boards = np.zeros((self.action_space.n, self.env.unwrapped.height,
                          self.env.unwrapped.width), dtype=np.uint8)
        mask = np.zeros(self.action_space.n, dtype=np.int8)

        # TODO: observation에서 현재 피스를 가져오는 방법 구현(padded_board->padded_piece...게임로직 변경)
        curr_piece = self.env.unwrapped.piece
        piece_id = observation["p_id"]

        if piece_id == 0:
            num_rotations = 1
        elif piece_id < 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for r in range(num_rotations):
            valid_xs = self.env.unwrapped.width - len(curr_piece[0])
            for x in range(valid_xs+1):
                action = self.encode_action(x, r)
                piece = [r[:] for r in curr_piece]
                y = 0
                while not self.env.unwrapped.check_collision(piece, x, y+1):
                    y += 1
                self.env.unwrapped.truncate_overflow_piece(piece, x, y)

                board = self.env.unwrapped.get_board_with_piece(piece, x, y)

                boards[action] = np.array(board)
                mask[action] = 1
            curr_piece = self.env.unwrapped.get_rotated_piece(curr_piece)

        return {
            "boards": boards,
            "action_mask": mask
        }
