import argparse
import os
from collections import deque
from random import random, sample
import time

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from rl_tetris.wrapper.Grouped import GroupedWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation

from benchmark.Logger import TetrisLogger

def get_args():
    parser = argparse.ArgumentParser("""Tetris 게임 환경 강화학습""")

    # 게임 환경 설정
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    # 하이퍼파라미터 설정
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--replay_memory_size", type=int, default=30000)

    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--initial_epsilon", type=float, default=1.0)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=int, default=2000)

    # 로깅 설정
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="Tetris-DQL")
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])

    # 모델 저장
    parser.add_argument("--save_interval", type=int, default=200)

    args = parser.parse_args()
    return args


# DNN 모델
class DNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        """게임 상태에 대한 휴리스틱한 정보(지워진 줄, 구멍, 인접열 차이 합, 높이 합)를 입력으로 받아서, q-value를 출력하는 DNN 모델"""
        super(DNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


# 입실론 스케줄 함수
def epsilon_schedule(epoch, initial_epsilon, final_epsilon, num_decay_epochs):
    return final_epsilon + (max(num_decay_epochs - epoch, 0) *
                            (initial_epsilon - final_epsilon) / num_decay_epochs)


def train(opt, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # Tensorboard
    logger = TetrisLogger(log_dir=os.environ["TENSORBOARD_LOGDIR"], agent_prefix="dqn")

    # Model, Optimizer, Loss function
    model = DNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_fn = F.mse_loss

    # Environment
    env = gym.make("RL-Tetris-v0", render_mode=None)
    env = GroupedWrapper(
        env, observation_wrapper=GroupedFeaturesObservation(env))

    # Replay memory
    replay_memory = deque(maxlen=opt.replay_memory_size)

    max_cleared_lines = 0
    epoch = 0
    while epoch < opt.num_epochs:
        epsilon = epsilon_schedule(
            epoch, opt.initial_epsilon, opt.final_epsilon, opt.num_decay_epochs)

        obs, info = env.reset()
        feature = torch.from_numpy(info["initial_feature"]).float().to(device)

        done = False
        episode_len = 0
        while not done:
            # Epsilon-greedy policy로 행동 선택(Exploration, Expoitation)
            if random() <= epsilon:
                action = env.action_space.sample(obs["action_mask"])
            else:
                valid_features = torch.from_numpy(
                    obs["features"][obs["action_mask"] == 1]).float().to(device)
                with torch.no_grad():
                    q_values = model(valid_features)[:, 0]
                index = torch.argmax(q_values).item()
                action = info["action_mapping"][index]

            next_feature = torch.from_numpy(
                obs["features"][action]).float().to(device)

            # 환경과 상호작용
            obs, reward, done, _, info = env.step(action)

            # Replay memory에 저장
            replay_memory.append([feature, reward, next_feature, done])

            if not done:
                episode_len += 1
                feature = next_feature
            else:
                print(
                    f'# Epoch: {epoch}, Score: {info["score"]}, Cleared lines: {info["cleared_lines"]}')

                if epoch > 0:
                    logger.log_episode(
                        score=info["score"],
                        cleared_lines=info["cleared_lines"],
                        episode_len=episode_len,
                        epoch=epoch
                    )

                # Most(cleared_lines) model save
                if info["cleared_lines"] > max_cleared_lines:
                    max_cleared_lines = info["cleared_lines"]
                    model_path = f"models/{run_name}/tetris_{epoch}_{max_cleared_lines}"
                    torch.save(model.state_dict(), model_path)
                    print(f"Best model saved at {model_path}")

        # Replay memory가 충분히 쌓여야 학습 시작
        if len(replay_memory) < opt.replay_memory_size // 10:
            continue

        # 학습 시작
        epoch += 1

        # 배치 샘플링
        batch = sample(replay_memory, opt.batch_size)
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(
            tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(
            np.array(reward_batch, dtype=np.int32)[:, None]).to(device)
        next_state_batch = torch.stack(
            tuple(state for state in next_state_batch)).to(device)

        # q_values, target_q_values 계산
        q_values = model(state_batch)
        with torch.no_grad():
            next_q_values = model(next_state_batch)
            done_batch = torch.tensor(done_batch, dtype=torch.float32)[
                :, None].to(device)
            target_q_values = reward_batch + opt.gamma * \
                next_q_values * (1 - done_batch)

        # Learning
        optimizer.zero_grad()
        loss = loss_fn(q_values, target_q_values)
        loss.backward()
        optimizer.step()

        # Logging
        loss = loss.item()
        print(f"Epoch: {epoch}, Loss: {loss:.4f}, Epsilon: {epsilon:.3f}")
        logger.log_dqn_step(
            td_loss=loss,
            q_values=q_values,
            target_q_values=target_q_values,
            epsilon=epsilon,
            epoch=epoch
        )

        # Model save
        if epoch > opt.num_decay_epochs and epoch % opt.save_interval == 0:
            max_cleared_lines = max(max_cleared_lines, info["cleared_lines"])
            model_path = f"models/{run_name}/tetris_{epoch}"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")

    torch.save(model, f"models/{run_name}/tetris")


if __name__ == "__main__":
    opt = get_args()
    print(f"opt: {opt.__dict__}")

    run_name = f"{opt.exp_name}/{opt.num_epochs}_{opt.batch_size}_{opt.replay_memory_size}__{int(time.time())}"

    # TensorBoard 로그 디렉토리 경로 설정
    log_dir = os.path.join("./runs", run_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    os.environ["TENSORBOARD_LOGDIR"] = log_dir

    # Model 저장 디렉토리 경로 설정
    if not os.path.isdir(f"./models/{run_name}"):
        os.makedirs(f"./models/{run_name}")

    if opt.wandb:
        import wandb

        run = wandb.init(
            project=opt.wandb_project_name,
            sync_tensorboard=True,
            config=vars(opt),
            name=run_name,
        )

    train(opt, run_name)

    if opt.wandb:
        run.finish()
