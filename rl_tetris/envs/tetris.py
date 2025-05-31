import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from dataclasses import fields

from rl_tetris.game_state import GameStates
from rl_tetris.tetromino_queue import TetrominoQueue
from rl_tetris.renderer import Renderer
from rl_tetris.mapping.actions import GameActions


class Tetris(gym.Env):
    metadata = {
        "render_modes": ["human", "animate"],
        "render_fps": 1
    }

    PIECES = [
        # O
        [[1, 1],
         [1, 1]],

        # I
        [[2, 2, 2, 2]],

        # S
        [[0, 3, 3],
         [3, 3, 0]],

        # Z
        [[4, 4, 0],
         [0, 4, 4]],

        # T
        [[0, 5, 0],
         [5, 5, 5]],

        # L
        [[0, 0, 6],
         [6, 6, 6]],

        # J
        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(
            self,
            render_mode=None,
            height=20,
            width=10,
            block_size=30,
            randomizer=None):
        self.height = height
        self.width = width
        self.queue = TetrominoQueue(randomizer=randomizer)
        self.renderer = Renderer(height, width, block_size)

        """
        게임 상태 관련 주요 변수
        - board : 현재 보드 상태
        - queue : 다음 블록을 뽑기 위한 TetrominoQueue 인스턴스
        - score : 현재 점수
        - cleared_lines : 지워진 줄 수
        - gameover : 게임 종료 여부
        - x, y : 현재 블록 위치
        - piece : 현재 블록(2차원 배열)
        - idx: 현재 종류 블록 인덱스
        """

        # Gymnasium
        self.observation_space = gym.spaces.Dict(
            {
                "board": Box(
                    low=0,
                    high=len(self.PIECES),
                    shape=(self.height, self.width),
                    dtype=np.uint8,
                ),
                # "piece": Box(
                #     low=0,
                #     high=len(self.PIECES),
                #     shape=(4, 4),
                #     dtype=np.uint8,
                # ),
                "p_id": Discrete(len(self.PIECES)),
                "x": Discrete(self.width),
                "y": Discrete(self.height),
            }
        )

        self.actions = GameActions
        self.action_space = Discrete(len(fields(GameActions)))
        self.reward_range = (-4, 17)

        self.render_mode = render_mode

    def get_observation(self):
        return {
            "board": np.array(self.board, dtype=np.uint8),
            # "piece": padded_piece,
            "p_id": self.idx,
            "x": self.x,
            "y": self.y,
        }

    def get_info(self):
        return {
            "score": self.score,
            "cleared_lines": self.cleared_lines,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """게임 상태 초기화 후, 초기 상태 특징을 반환하는 메서드"""
        self.board = [[0] * self.width for _ in range(self.height)]

        self.score = 0
        self.cleared_lines = 0
        self.queue.reset()

        self.idx = self.queue.pop()
        self.piece = [r[:] for r in self.PIECES[self.idx]]
        self.x, self.y = self.width // 2 - len(self.piece[0]) // 2, 0
        self.gameover = False

        return self.get_observation(), self.get_info()

    def step(self, action: GameActions) -> tuple[dict, int, bool, dict]:
        reward = 0
        lines_cleared = 0

        # piece에 이동방향을 미리 적용하여 check_collision 메서드를 통해 충돌을 확인한 후 이동방향 적용
        if action == GameActions.move_left:
            if not self.check_collision(self.piece, self.x - 1, self.y):
                self.x -= 1
        elif action == GameActions.move_right:
            if not self.check_collision(self.piece, self.x + 1, self.y):
                self.x += 1
        elif action == GameActions.move_down:
            if not self.check_collision(self.piece, self.x, self.y + 1):
                self.y += 1
        elif action == GameActions.rotate:
            rotated_piece = self.get_rotated_piece(self.piece)
            if not self.check_collision(rotated_piece, self.x, self.y):
                self.piece = rotated_piece
        elif action == GameActions.hard_drop:
            while not self.check_collision(self.piece, self.x, self.y + 1):
                if self.render_mode == "animate":
                    self.render()
                self.y += 1

        # 현재 piece가 보드 상단을 벗어나는 경우 = 게임오버
        if self.truncate_overflow_piece(self.piece, self.x, self.y):
            self.gameover = True

            self.board = self.get_board_with_piece(self.piece, self.x, self.y)
            lines_cleared, self.board = self.clear_full_rows(self.board)
            self.cleared_lines += lines_cleared
            reward = self.get_reward(lines_cleared) - 5
            self.score += reward

            return (
                self.get_observation(),
                reward,
                self.gameover,
                False,  # truncated
                self.get_info()
            )

        # 게임오버가 아니지만 움직일 수 없는 경우, 다음 테트로미노를 뽑아서 현재 테트로미노로 설정
        if self.check_collision(self.piece, self.x, self.y + 1):
            self.board = self.get_board_with_piece(self.piece, self.x, self.y)

            if self.render_mode == "animate":
                self.render()

            lines_cleared, self.board = self.clear_full_rows(self.board)

            self.cleared_lines += lines_cleared
            reward = self.get_reward(lines_cleared)
            self.score += reward

            self.spawn_next_piece()

        return (
            self.get_observation(),
            reward,
            self.gameover,
            False,  # truncated
            self.get_info()
        )

##################################################

    def check_collision(self, piece, px, py):
        """현재 보드 상태에서, piece가 pos에 추가될 때 충돌이 발생하는지 여부를 반환하는 메서드"""

        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if py+y > self.height-1 or py+y < 0 or px+x > self.width-1 or px+x < 0:
                    return True
                if piece[y][x] == 0:
                    continue
                if self.board[py+y][px+x] > 0:
                    return True
        return False

    def get_board_with_piece(self, piece, px, py):
        """현재 보드의 복사본을 만들어서, piece를 pos에 추가한 보드를 반환하는 메서드"""

        board = [r[:] for r in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + py][x + px]:
                    board[y + py][x + px] = piece[y][x]
        return board

    def truncate_overflow_piece(self, piece, px, py):
        # 현재 보드에 대해선 수정하는 작업 없이, 게임종료 여부 반환
        # 이때 piece가 보드 밖으로 나가는 경우, in-place 연산으로 piece를 잘라서 보드 안에 들어오게 함

        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] == 0:
                    continue
                if self.board[py + y][px + x]:
                    if y > last_collision_row:
                        last_collision_row = y
                        break

        if py - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[py + y][px + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def spawn_next_piece(self):
        """다음 테트로미노를 뽑아서 현재 테트로미노로 설정하는 메서드"""

        self.idx = self.queue.pop()
        self.piece = [r[:] for r in self.PIECES[self.idx]]
        self.x, self.y = self.width // 2 - len(self.piece[0]) // 2, 0
        if self.check_collision(self.piece, self.x, self.y):
            self.gameover = True

    def get_rotated_piece(self, piece):
        """현재 테트로미노를 시계방향으로 90도 회전한 결과를 반환하는 메서드"""

        num_rows_orig = num_cols_new = len(piece)
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_reward(self, lines_cleared):
        """지워진 줄 수에 대한 보상을 반환하는 메서드"""
        return 1 + (lines_cleared ** 2) * self.width

##################################################

    def clear_full_rows_(self, board: np.ndarray):
        # np연산으로 전부 0이아닌 줄을 지우고 가장 위에 빈 줄을 추가하는 메서드

        mask = np.all(board != 0, axis=1)
        board = board[~mask]
        board = np.concatenate(
            [np.zeros((self.height - len(board), self.width)), board])
        return np.sum(mask), board

    def clear_full_rows(self, board):
        """보드에서 꽉 찬 줄을 지우고, 지워진 줄 수와 보드를 반환하는 메서드"""
        # in-place로 가득 찬 줄을 지우고, 지워진 줄 수와 보드를 반환

        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self._remove_rows(board, to_delete)
        return len(to_delete), board

    def get_holes(self, board):
        """보드에서 구멍 수를 반환하는 메서드"""
        # 위에서부터 블록있는 곳까지 내려가고, 이후부터 빈칸을 세는 방식으로 구멍 수를 반환

        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        """보드에서 인접 열간 높이 차이와 각 열의 높이 합을 반환하는 메서드"""
        # 인접 열간 높이 차이인 diffs의 합 total_bumpiness, 각 열의 높이 hight의 합 hights를 반환

        board = np.array(board)
        mask = board != 0
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def _remove_rows(self, board, indices):
        # 보드에서 indices에 해당하는 행을 in-place 삭제하고, 위에 빈 행을 추가하는 메서드
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board
    
    def get_landing_height(self, board):
        heights = self._get_column_heights(board)
        return float(np.mean(heights))


    def get_row_transitions(self, board):
        transitions = 0
        for row in board:
            prev = 1
            for cell in row:
                if cell == 0 and prev == 1:
                    transitions += 1
                elif cell != 0 and prev == 0:
                    transitions += 1
                prev = cell != 0
            if prev == 0:
                transitions += 1
        return float(transitions)


    def get_col_transitions(self, board):
        transitions = 0
        for col in zip(*board):
            prev = 1
            for cell in col:
                if cell == 0 and prev == 1:
                    transitions += 1
                elif cell != 0 and prev == 0:
                    transitions += 1
                prev = cell != 0
            if prev == 0:
                transitions += 1
        return float(transitions)


    def get_cumulative_wells(self, board):
        wells = 0
        board = np.array(board)
        h, w = board.shape
        for x in range(w):
            for y in range(h):
                if board[y, x] == 0:
                    left_filled = (x == 0) or (board[y, x - 1] != 0)
                    right_filled = (x == w - 1) or (board[y, x + 1] != 0)
                    if left_filled and right_filled:
                        wells += 1
                else:
                    break
        return float(wells)


    def _get_column_heights(self, board):
        board = np.array(board)
        heights = []
        for col in board.T:
            idx = np.argmax(col != 0)
            if col[idx] == 0:
                heights.append(0)
            else:
                heights.append(board.shape[0] - idx)
        return heights

    def get_eroded_piece_cells(self, lines_cleared, board_before, board_after):
        """지운 줄 수  지워진 piece cell 수"""
        if lines_cleared == 0:
            return 0.0
        diff = (np.array(board_before) != 0) & (np.array(board_after) == 0)
        return float(np.sum(diff))

    def get_hole_depth(self, board):
        board = np.array(board)
        depth = 0
        for x in range(board.shape[1]):
            col = board[:, x]
            found_block = False
            col_depth = 0
            for cell in col:
                if cell != 0:
                    found_block = True
                elif found_block:
                    col_depth += 1
            depth += col_depth
        return float(depth)


    def get_rows_with_holes(self, board):
        board = np.array(board)
        rows_with_holes = 0
        for row in board:
            hole_found = False
            block_found = False
            for cell in row:
                if cell == 0 and block_found:
                    hole_found = True
                elif cell != 0:
                    block_found = True
            if hole_found:
                rows_with_holes += 1
        return float(rows_with_holes)

    def get_pattern_diversity(self, board):
        heights = self._get_column_heights(board)
        patterns = set()
        for i in range(len(heights) - 1):
            pattern = (heights[i], heights[i+1])
            patterns.add(pattern)
        return float(len(patterns))

    def get_column_height_variance(self, board):
        heights = self._get_column_heights(board)
        return float(np.var(heights))
    
    def get_max_well_depth(self, board):
        board = np.array(board)
        h, w = board.shape
        max_depth = 0
        for x in range(w):
            depth = 0
            for y in range(h):
                if board[y, x] == 0:
                    left = x == 0 or board[y, x - 1] != 0
                    right = x == w - 1 or board[y, x + 1] != 0
                    if left and right:
                        depth += 1
                    else:
                        break
                else:
                    break
            max_depth = max(max_depth, depth)
        return float(max_depth)


##################################################

    def get_render_state(self) -> GameStates:
        """랜더링을 위해 현재 게임 상태를 반환하는 메서드"""

        board = self.get_board_with_piece(self.piece, self.x, self.y)
        next_idx = self.queue.peek()
        next_piece = self.PIECES[next_idx]

        return GameStates(board, self.score, next_piece)

    def render(self):
        """게임을 렌더링하는 메서드"""
        self.renderer.render(self.get_render_state())
