import gymnasium as gym
import numpy as np

from rl_tetris.envs.tetris import Tetris


class BoardObservation(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)

    def observation(self, observation):
        # TODO: piece가 회전에도 고정 observation에 맞도록, n*n 크기로 바꾸기 -> board에 piece를 합쳐서 반환
        return observation["boards"]


class GroupedFeaturesObservation(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)

    def observation(self, observation):
        boards = observation["boards"]
        mask = observation["action_mask"]

        dummy = self.extract_board_features(np.zeros((self.env.unwrapped.height, self.env.unwrapped.width), dtype=np.uint8))
        feature_length = len(dummy)

        features = np.zeros((len(boards), feature_length), dtype=np.float32)

        for i, (board, m) in enumerate(zip(boards, mask)):
            if m == 1:
                features[i] = self.extract_board_features(board)

        return features


    def extract_board_features(self, board):
        lines_cleared, board = self.env.unwrapped.clear_full_rows_(board)
        holes = self.env.unwrapped.get_holes(board)
        bumpiness, height = self.env.unwrapped.get_bumpiness_and_height(board)
        landing_height = self.env.unwrapped.get_landing_height(board)
        row_transitions = self.env.unwrapped.get_row_transitions(board)
        col_transitions = self.env.unwrapped.get_col_transitions(board)
        cumulative_wells = self.env.unwrapped.get_cumulative_wells(board)

        return np.array([
            lines_cleared,
            holes,
            bumpiness,
            height,
            landing_height,
            row_transitions,
            col_transitions,
            cumulative_wells
        ], dtype=np.float32)