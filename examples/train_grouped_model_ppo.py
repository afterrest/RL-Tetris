import argparse
import os
from collections import deque
import time
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from rl_tetris.wrapper.Grouped import GroupedWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation


def get_args():
    parser = argparse.ArgumentParser("""Tetris 게임 환경 PPO 강화학습""")

    # 게임 환경 설정
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    # PPO 하이퍼파라미터 설정 (테트리스에 최적화)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)  # 더 높은 학습률

    # PPO 특화 파라미터 (테트리스에 맞게 조정)
    parser.add_argument("--n_steps", type=int, default=512, help="스텝 수 per update")
    parser.add_argument("--n_epochs", type=int, default=4, help="PPO 업데이트 에포크 수")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="PPO clipping coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="할인 팩터")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="엔트로피 계수")
    parser.add_argument("--vf_coef", type=float, default=1.0, help="가치 함수 계수")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="그래디언트 클리핑")

    # 테트리스 특화 설정
    parser.add_argument("--reward_scale", type=float, default=0.01, help="보상 스케일링")
    parser.add_argument("--value_loss_scale", type=float, default=10.0, help="가치 손실 스케일링")

    # 로깅 설정
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="Tetris-PPO")
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])

    # 모델 저장
    parser.add_argument("--save_interval", type=int, default=100)

    args = parser.parse_args()
    return args


class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256):
        """
        개선된 Actor-Critic 네트워크
        """
        super(ActorCritic, self).__init__()

        # 더 깊은 공통 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor: 각 행동에 대한 점수
        self.actor = nn.Linear(hidden_dim // 2, 1)

        # Critic: 상태 가치 (별도 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def get_action_logits(self, valid_features):
        """유효한 행동들에 대한 로짓 계산"""
        features = self.feature_extractor(valid_features)
        logits = self.actor(features).squeeze(-1)
        return logits

    def get_value(self, state_feature):
        """상태 가치 계산 (대표 특징 사용)"""
        return self.critic(state_feature).squeeze(-1)

    def get_action_and_value(self, valid_features, state_feature, action=None):
        """
        행동 선택 및 가치 계산
        valid_features: 유효한 행동들의 특징 (N, 4)
        state_feature: 현재 상태를 대표하는 특징 (1, 4)
        """
        # 행동 선택
        logits = self.get_action_logits(valid_features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # 상태 가치
        value = self.get_value(state_feature)

        return action, log_prob, entropy, value


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """GAE 계산"""
    advantages = []
    gae = 0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value_step = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value_step = values[step + 1]

        delta = rewards[step] + gamma * next_value_step * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train(opt, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed 설정
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # TensorBoard 설정
    writer = SummaryWriter(log_dir=os.environ["TENSORBOARD_LOGDIR"])

    # 모델, 옵티마이저 설정
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=1e-5)

    # 환경 설정
    env = gym.make("RL-Tetris-v0", render_mode=None)
    env = GroupedWrapper(env, observation_wrapper=GroupedFeaturesObservation(env))

    # 통계 추적
    max_cleared_lines = 0
    global_step = 0
    update = 0

    # 성능 추적을 위한 이동 평균
    recent_scores = deque(maxlen=100)
    recent_cleared_lines = deque(maxlen=100)

    # 메인 학습 루프
    while update < opt.num_epochs:
        model.eval()

        # 데이터 수집용 저장소
        state_features = []  # 상태를 대표하는 특징
        valid_features_list = []  # 각 스텝의 유효한 특징들
        actions_buffer = []
        logprobs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []

        # 에피소드별 통계
        episode_rewards = []
        episode_lengths = []
        episode_cleared_lines = []

        # n_steps만큼 데이터 수집
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(opt.n_steps):
            global_step += 1

            # 현재 상태의 특징 벡터들
            features = torch.from_numpy(obs["features"]).float().to(device)
            action_mask = obs["action_mask"]

            # 유효한 행동들 필터링
            valid_indices = np.where(action_mask == 1)[0]
            valid_features = features[valid_indices]

            # 상태를 대표하는 특징 (첫 번째 유효한 행동의 특징 사용)
            state_feature = valid_features[0:1]

            # 행동 선택 및 가치 계산
            with torch.no_grad():
                action_idx, logprob, _, value = model.get_action_and_value(
                    valid_features, state_feature
                )

                # 실제 행동 매핑
                actual_action = valid_indices[action_idx.item()]

            # 환경과 상호작용
            next_obs, reward, done, _, next_info = env.step(actual_action)

            # 보상 스케일링 (학습 안정성을 위해)
            scaled_reward = reward * opt.reward_scale

            # 버퍼에 저장
            state_features.append(state_feature.cpu().numpy())
            valid_features_list.append(valid_features.cpu().numpy())
            actions_buffer.append(action_idx.item())
            logprobs_buffer.append(logprob.item())
            rewards_buffer.append(scaled_reward)
            dones_buffer.append(done)
            values_buffer.append(value.item())

            episode_reward += reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_cleared_lines.append(next_info["cleared_lines"])

                recent_scores.append(next_info["score"])
                recent_cleared_lines.append(next_info["cleared_lines"])

                print(f'Update {update}, Step {global_step}: '
                      f'Score: {next_info["score"]}, Lines: {next_info["cleared_lines"]}, '
                      f'Avg Score: {np.mean(recent_scores):.1f}, '
                      f'Avg Lines: {np.mean(recent_cleared_lines):.1f}')

                # 최고 기록 갱신 시 모델 저장
                if next_info["cleared_lines"] > max_cleared_lines:
                    max_cleared_lines = next_info["cleared_lines"]
                    model_path = f"models/{run_name}/tetris_best_{max_cleared_lines}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Best model saved: {model_path}")

                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
                info = next_info

        # 마지막 상태의 가치 계산
        if not done:
            features = torch.from_numpy(obs["features"]).float().to(device)
            action_mask = obs["action_mask"]
            valid_indices = np.where(action_mask == 1)[0]
            valid_features = features[valid_indices]
            state_feature = valid_features[0:1]

            with torch.no_grad():
                next_value = model.get_value(state_feature).item()
        else:
            next_value = 0.0

        # GAE 계산
        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer,
            next_value, opt.gamma, opt.gae_lambda
        )

        # 텐서로 변환
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(logprobs_buffer, dtype=torch.float32).to(device)

        # Advantage 정규화
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO 업데이트
        model.train()

        batch_size = min(opt.batch_size, len(rewards_buffer))
        batch_indices = np.arange(len(rewards_buffer))

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for epoch in range(opt.n_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, len(rewards_buffer), batch_size):
                end = start + batch_size
                mb_indices = batch_indices[start:end]

                # 미니배치 데이터 준비
                mb_state_features = torch.from_numpy(np.vstack([state_features[i] for i in mb_indices])).float().to(
                    device)
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]

                # 각 샘플별로 정책 및 가치 계산
                new_logprobs = []
                entropies = []
                new_values = []

                for i, idx in enumerate(mb_indices):
                    # 해당 샘플의 유효한 특징들
                    valid_feats = torch.from_numpy(valid_features_list[idx]).float().to(device)
                    state_feat = mb_state_features[i:i + 1]
                    action = actions_buffer[idx]

                    # 정책 및 가치 계산
                    _, log_prob, entropy, value = model.get_action_and_value(
                        valid_feats, state_feat, torch.tensor(action).to(device)
                    )

                    new_logprobs.append(log_prob)
                    entropies.append(entropy)
                    new_values.append(value)

                new_logprobs = torch.stack(new_logprobs)
                entropies = torch.stack(entropies)
                new_values = torch.stack(new_values)

                # PPO 손실 계산
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # Policy loss (clipped surrogate objective)
                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1 - opt.clip_coef, 1 + opt.clip_coef)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value function loss (더 강한 가중치)
                value_loss = F.mse_loss(new_values.squeeze(), mb_returns) * opt.value_loss_scale

                # Entropy loss
                entropy_loss = -entropies.mean()

                # 총 손실
                total_loss = policy_loss + opt.vf_coef * value_loss + opt.ent_coef * entropy_loss

                # 역전파
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        update += 1

        # 로깅
        if episode_rewards:
            avg_episode_reward = np.mean(episode_rewards)
            avg_episode_length = np.mean(episode_lengths)
            avg_cleared_lines = np.mean(episode_cleared_lines)

            print(f"Update {update}: "
                  f"Avg Reward: {avg_episode_reward:.2f}, "
                  f"Avg Lines: {avg_cleared_lines:.2f}, "
                  f"Policy Loss: {np.mean(policy_losses):.4f}, "
                  f"Value Loss: {np.mean(value_losses):.4f}")

            writer.add_scalar("train/episode_reward", avg_episode_reward, update)
            writer.add_scalar("train/episode_length", avg_episode_length, update)
            writer.add_scalar("train/cleared_lines", avg_cleared_lines, update)
            writer.add_scalar("train/policy_loss", np.mean(policy_losses), update)
            writer.add_scalar("train/value_loss", np.mean(value_losses), update)
            writer.add_scalar("train/entropy_loss", np.mean(entropy_losses), update)
            writer.add_scalar("train/avg_score_100", np.mean(recent_scores), update)
            writer.add_scalar("train/avg_lines_100", np.mean(recent_cleared_lines), update)

        # 주기적 모델 저장
        if update % opt.save_interval == 0:
            model_path = f"models/{run_name}/tetris_update_{update}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

    # 최종 모델 저장
    torch.save(model.state_dict(), f"models/{run_name}/tetris_final.pth")
    writer.close()

import argparse
import os
from collections import deque
import time
import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from rl_tetris.wrapper.Grouped import GroupedWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation

from benchmark.Logger import TetrisLogger


def get_args():
    parser = argparse.ArgumentParser("""Tetris 게임 환경 PPO 강화학습""")

    # 게임 환경 설정
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--block_size", type=int, default=30)

    # PPO 하이퍼파라미터 설정 (테트리스에 최적화)
    parser.add_argument("--num_epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)  # 더 높은 학습률

    # PPO 특화 파라미터 (테트리스에 맞게 조정)
    parser.add_argument("--n_steps", type=int, default=512, help="스텝 수 per update")
    parser.add_argument("--n_epochs", type=int, default=4, help="PPO 업데이트 에포크 수")
    parser.add_argument("--clip_coef", type=float, default=0.1, help="PPO clipping coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="할인 팩터")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="엔트로피 계수")
    parser.add_argument("--vf_coef", type=float, default=1.0, help="가치 함수 계수")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="그래디언트 클리핑")

    # 테트리스 특화 설정
    parser.add_argument("--reward_scale", type=float, default=0.01, help="보상 스케일링")
    parser.add_argument("--value_loss_scale", type=float, default=10.0, help="가치 손실 스케일링")

    # 로깅 설정
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--wandb_project_name", type=str, default="Tetris-PPO")
    parser.add_argument("--exp_name", type=str,
                        default=os.path.basename(__file__)[: -len(".py")])

    # 모델 저장
    parser.add_argument("--save_interval", type=int, default=100)

    args = parser.parse_args()
    return args


class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256):
        """
        개선된 Actor-Critic 네트워크
        """
        super(ActorCritic, self).__init__()

        # 더 깊은 공통 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Actor: 각 행동에 대한 점수
        self.actor = nn.Linear(hidden_dim // 2, 1)

        # Critic: 상태 가치 (별도 네트워크)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.constant_(m.bias, 0)

    def get_action_logits(self, valid_features):
        """유효한 행동들에 대한 로짓 계산"""
        features = self.feature_extractor(valid_features)
        logits = self.actor(features).squeeze(-1)
        return logits

    def get_value(self, state_feature):
        """상태 가치 계산 (대표 특징 사용)"""
        return self.critic(state_feature).squeeze(-1)

    def get_action_and_value(self, valid_features, state_feature, action=None):
        """
        행동 선택 및 가치 계산
        valid_features: 유효한 행동들의 특징 (N, 4)
        state_feature: 현재 상태를 대표하는 특징 (1, 4)
        """
        # 행동 선택
        logits = self.get_action_logits(valid_features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # 상태 가치
        value = self.get_value(state_feature)

        return action, log_prob, entropy, value


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """GAE 계산"""
    advantages = []
    gae = 0

    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value_step = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value_step = values[step + 1]

        delta = rewards[step] + gamma * next_value_step * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def train(opt, run_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Seed 설정
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # TensorBoard 및 Logger 설정
    logger = TetrisLogger(log_dir=os.environ["TENSORBOARD_LOGDIR"], agent_prefix="ppo")

    # 모델, 옵티마이저 설정
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, eps=1e-5)

    # 환경 설정
    env = gym.make("RL-Tetris-v0", render_mode=None)
    env = GroupedWrapper(env, observation_wrapper=GroupedFeaturesObservation(env))

    # 통계 추적
    max_cleared_lines = 0
    global_step = 0
    update = 0

    # 성능 추적을 위한 이동 평균
    recent_scores = deque(maxlen=100)
    recent_cleared_lines = deque(maxlen=100)

    # 메인 학습 루프
    while update < opt.num_epochs:
        model.eval()

        # 데이터 수집용 저장소
        state_features = []  # 상태를 대표하는 특징
        valid_features_list = []  # 각 스텝의 유효한 특징들
        actions_buffer = []
        logprobs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        values_buffer = []

        # 에피소드별 통계
        episode_rewards = []
        episode_lengths = []
        episode_cleared_lines = []

        # n_steps만큼 데이터 수집
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(opt.n_steps):
            global_step += 1

            # 현재 상태의 특징 벡터들
            features = torch.from_numpy(obs["features"]).float().to(device)
            action_mask = obs["action_mask"]

            # 유효한 행동들 필터링
            valid_indices = np.where(action_mask == 1)[0]
            valid_features = features[valid_indices]

            # 상태를 대표하는 특징 (첫 번째 유효한 행동의 특징 사용)
            state_feature = valid_features[0:1]

            # 행동 선택 및 가치 계산
            with torch.no_grad():
                action_idx, logprob, _, value = model.get_action_and_value(
                    valid_features, state_feature
                )

                # 실제 행동 매핑
                actual_action = valid_indices[action_idx.item()]

            # 환경과 상호작용
            next_obs, reward, done, _, next_info = env.step(actual_action)

            # 보상 스케일링 (학습 안정성을 위해)
            scaled_reward = reward * opt.reward_scale

            # 버퍼에 저장
            state_features.append(state_feature.cpu().numpy())
            valid_features_list.append(valid_features.cpu().numpy())
            actions_buffer.append(action_idx.item())
            logprobs_buffer.append(logprob.item())
            rewards_buffer.append(scaled_reward)
            dones_buffer.append(done)
            values_buffer.append(value.item())

            episode_reward += reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_cleared_lines.append(next_info["cleared_lines"])

                recent_scores.append(next_info["score"])
                recent_cleared_lines.append(next_info["cleared_lines"])

                print(f'Update {update}, Step {global_step}: '
                      f'Score: {next_info["score"]}, Lines: {next_info["cleared_lines"]}, '
                      f'Avg Score: {np.mean(recent_scores):.1f}, '
                      f'Avg Lines: {np.mean(recent_cleared_lines):.1f}')

                # Logger를 사용한 에피소드 로깅
                if update > 0:
                    logger.log_episode(
                        score=next_info["score"],
                        cleared_lines=next_info["cleared_lines"],
                        episode_len=episode_length,
                        epoch=update
                    )

                # 최고 기록 갱신 시 모델 저장
                if next_info["cleared_lines"] > max_cleared_lines:
                    max_cleared_lines = next_info["cleared_lines"]
                    model_path = f"models/{run_name}/tetris_best_{max_cleared_lines}.pth"
                    torch.save(model.state_dict(), model_path)
                    print(f"Best model saved: {model_path}")

                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
                info = next_info

        # 마지막 상태의 가치 계산
        if not done:
            features = torch.from_numpy(obs["features"]).float().to(device)
            action_mask = obs["action_mask"]
            valid_indices = np.where(action_mask == 1)[0]
            valid_features = features[valid_indices]
            state_feature = valid_features[0:1]

            with torch.no_grad():
                next_value = model.get_value(state_feature).item()
        else:
            next_value = 0.0

        # GAE 계산
        advantages, returns = compute_gae(
            rewards_buffer, values_buffer, dones_buffer,
            next_value, opt.gamma, opt.gae_lambda
        )

        # 텐서로 변환
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(logprobs_buffer, dtype=torch.float32).to(device)

        # Advantage 정규화
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO 업데이트
        model.train()

        batch_size = min(opt.batch_size, len(rewards_buffer))
        batch_indices = np.arange(len(rewards_buffer))

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for epoch in range(opt.n_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, len(rewards_buffer), batch_size):
                end = start + batch_size
                mb_indices = batch_indices[start:end]

                # 미니배치 데이터 준비
                mb_state_features = torch.from_numpy(np.vstack([state_features[i] for i in mb_indices])).float().to(
                    device)
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]

                # 각 샘플별로 정책 및 가치 계산
                new_logprobs = []
                entropies = []
                new_values = []

                for i, idx in enumerate(mb_indices):
                    # 해당 샘플의 유효한 특징들
                    valid_feats = torch.from_numpy(valid_features_list[idx]).float().to(device)
                    state_feat = mb_state_features[i:i + 1]
                    action = actions_buffer[idx]

                    # 정책 및 가치 계산
                    _, log_prob, entropy, value = model.get_action_and_value(
                        valid_feats, state_feat, torch.tensor(action).to(device)
                    )

                    new_logprobs.append(log_prob)
                    entropies.append(entropy)
                    new_values.append(value)

                new_logprobs = torch.stack(new_logprobs)
                entropies = torch.stack(entropies)
                new_values = torch.stack(new_values)

                # PPO 손실 계산
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                # Policy loss (clipped surrogate objective)
                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1 - opt.clip_coef, 1 + opt.clip_coef)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value function loss (더 강한 가중치)
                value_loss = F.mse_loss(new_values.squeeze(), mb_returns) * opt.value_loss_scale

                # Entropy loss
                entropy_loss = -entropies.mean()

                # 총 손실
                total_loss = policy_loss + opt.vf_coef * value_loss + opt.ent_coef * entropy_loss

                # 역전파
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
                optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        update += 1

        # 로깅
        if episode_rewards:
            avg_episode_reward = np.mean(episode_rewards)
            avg_episode_length = np.mean(episode_lengths)
            avg_cleared_lines = np.mean(episode_cleared_lines)

            print(f"Update {update}: "
                  f"Avg Reward: {avg_episode_reward:.2f}, "
                  f"Avg Lines: {avg_cleared_lines:.2f}, "
                  f"Policy Loss: {np.mean(policy_losses):.4f}, "
                  f"Value Loss: {np.mean(value_losses):.4f}")

            # Logger를 사용한 학습 메트릭 로깅
            values_tensor = torch.tensor(values_buffer).to(device)
            logger.log_dqn_step(
                td_loss=np.mean(policy_losses),  # PPO에서는 policy loss를 대신 사용
                q_values=values_tensor,
                target_q_values=returns_tensor,
                epsilon=0.0,  # PPO에서는 epsilon이 없으므로 0
                epoch=update
            )
            logger.log_perf(epoch=update)

        # 주기적 모델 저장
        if update % opt.save_interval == 0:
            model_path = f"models/{run_name}/tetris_update_{update}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved: {model_path}")

    # 최종 모델 저장
    torch.save(model.state_dict(), f"models/{run_name}/tetris_final.pth")


if __name__ == "__main__":
    opt = get_args()
    print(f"PPO Options: {opt.__dict__}")

    run_name = f"{opt.exp_name}/PPO_{opt.num_epochs}_{opt.batch_size}_{opt.n_steps}__{int(time.time())}"

    # TensorBoard 로그 디렉토리 설정
    log_dir = os.path.join("./runs", run_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    os.environ["TENSORBOARD_LOGDIR"] = log_dir

    # 모델 저장 디렉토리 설정
    model_dir = f"./models/{run_name}"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

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
if __name__ == "__main__":
    opt = get_args()
    print(f"PPO Options: {opt.__dict__}")

    run_name = f"{opt.exp_name}/PPO_{opt.num_epochs}_{opt.batch_size}_{opt.n_steps}__{int(time.time())}"

    # TensorBoard 로그 디렉토리 설정
    log_dir = os.path.join("./runs", run_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    os.environ["TENSORBOARD_LOGDIR"] = log_dir

    # 모델 저장 디렉토리 설정
    model_dir = f"./models/{run_name}"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

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