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
        env = self.env.unwrapped

        # 줄 제거 전의 board 복사 (eroded piece cells 계산용)
        board_before = np.copy(board)

        # 줄 제거 수행
        lines_cleared, board_after = env.clear_full_rows_(board)

        # Feature 계산
        holes = env.get_holes(board_after)
        bumpiness, height = env.get_bumpiness_and_height(board_after)
        #landing_height = env.get_landing_height(board_after)
        #row_transitions = env.get_row_transitions(board_after)
        #col_transitions = env.get_col_transitions(board_after)
        #cumulative_wells = env.get_cumulative_wells(board_after)
        eroded_piece_cells = env.get_eroded_piece_cells(lines_cleared, board_before, board_after)
        #hole_depth = env.get_hole_depth(board_after)
        #rows_with_holes = env.get_rows_with_holes(board_after)
        #pattern_diversity = env.get_pattern_diversity(board_after)
        #column_height_variance = env.get_column_height_variance(board_after)
        #max_well_depth = env.get_max_well_depth(board_after)
        
        return np.array([
            lines_cleared,
            holes,
            bumpiness,
            height,
            #landing_height,
            #row_transitions,
            #col_transitions,
            #cumulative_wells,
            eroded_piece_cells,
            #hole_depth,
            #rows_with_holes,
            #pattern_diversity,
            #column_height_variance
            #max_well_depth
        ], dtype=np.float32)
