import argparse
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

from rl_tetris.randomizer import BagRandomizer
from rl_tetris.wrapper.Grouped import GroupedWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation


def get_args():
    parser = argparse.ArgumentParser("""PPO RL-Tetris 모델 테스트""")
    parser.add_argument("--width", type=int, default=10,
                        help="The common width for all images")
    parser.add_argument("--height", type=int, default=20,
                        help="The common height for all images")
    parser.add_argument("--block_size", type=int,
                        default=30, help="Size of a block")
    parser.add_argument("--model_dir", type=str,
                        default="models/train_grouped_model_ppo/PPO_5000_256_512__1748225452",
                        help="PPO 모델이 저장된 디렉토리")
    parser.add_argument("--model_name", type=str,
                        default="tetris_best_190.pth",
                        help="불러올 모델 파일명")
    parser.add_argument("--deterministic", type=bool, default=True,
                        help="결정론적 행동 선택 (True) vs 확률적 행동 선택 (False)")
    parser.add_argument("--num_games", type=int, default=1,
                        help="플레이할 게임 수")

    args = parser.parse_args()
    return args


class ActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256):
        """
        PPO ActorCritic 네트워크 (훈련 코드와 동일한 구조)
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

    def get_action_and_value(self, valid_features, state_feature, deterministic=False):
        """
        행동 선택 및 가치 계산
        deterministic: True면 최고 확률 행동 선택, False면 확률적 샘플링
        """
        # 행동 선택
        logits = self.get_action_logits(valid_features)
        probs = Categorical(logits=logits)

        if deterministic:
            # 가장 높은 확률의 행동 선택 (결정론적)
            action = torch.argmax(logits)
        else:
            # 확률에 따른 샘플링 (확률적)
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        # 상태 가치
        value = self.get_value(state_feature)

        return action, log_prob, entropy, value, probs.probs


def test_single_game(model, env, deterministic=True, verbose=True):
    """단일 게임 실행"""
    obs, info = env.reset()

    total_reward = 0
    step_count = 0
    done = False

    if verbose:
        print(f"게임 시작 - 결정론적 모드: {deterministic}")

    while not done:
        if env.render_mode == "animate":
            # 애니메이션 모드에서는 자동으로 렌더링됨
            pass
        else:
            env.render()

        # 현재 상태의 특징 벡터들
        features = torch.from_numpy(obs["features"]).float()
        action_mask = torch.from_numpy(obs["action_mask"])

        # 유효한 행동들 필터링
        valid_indices = torch.where(action_mask == 1)[0].numpy()
        valid_features = features[valid_indices]

        # 상태를 대표하는 특징 (첫 번째 유효한 행동의 특징 사용)
        state_feature = valid_features[0:1]

        # 행동 선택
        with torch.no_grad():
            action_idx, log_prob, entropy, value, action_probs = model.get_action_and_value(
                valid_features, state_feature, deterministic=deterministic
            )

        # 실제 행동 매핑
        actual_action = valid_indices[action_idx.item()]

        if verbose and step_count % 100 == 0:
            print(f"Step {step_count}: 유효한 행동 수: {len(valid_indices)}, "
                  f"선택된 행동: {actual_action}, 상태 가치: {value.item():.3f}")
            if not deterministic:
                print(f"  행동 확률 분포: {action_probs.numpy()}")

        # 환경과 상호작용
        obs, reward, done, _, info = env.step(actual_action)

        total_reward += reward
        step_count += 1

        # 게임 종료 조건 체크 (무한 루프 방지)
        if step_count > 10000:
            print("최대 스텝 수 도달, 게임 종료")
            break

    return {
        'score': info['score'],
        'cleared_lines': info['cleared_lines'],
        'total_reward': total_reward,
        'steps': step_count
    }


def test_multiple_games(model, env, num_games=5, deterministic=True):
    """여러 게임 실행 및 통계 계산"""
    results = []

    print(f"\n=== {num_games}게임 테스트 시작 ===")
    print(f"모드: {'결정론적' if deterministic else '확률적'}")

    for game_idx in range(num_games):
        print(f"\n--- 게임 {game_idx + 1}/{num_games} ---")
        result = test_single_game(model, env, deterministic=deterministic, verbose=(game_idx == 0))
        results.append(result)

        print(f"게임 {game_idx + 1} 결과:")
        print(f"  점수: {result['score']}")
        print(f"  지운 줄: {result['cleared_lines']}")
        print(f"  총 보상: {result['total_reward']:.2f}")
        print(f"  스텝 수: {result['steps']}")

    # 통계 계산
    scores = [r['score'] for r in results]
    lines = [r['cleared_lines'] for r in results]
    rewards = [r['total_reward'] for r in results]
    steps = [r['steps'] for r in results]

    print(f"\n=== 전체 통계 ({num_games}게임) ===")
    print(f"평균 점수: {sum(scores) / len(scores):.1f} (최고: {max(scores)}, 최저: {min(scores)})")
    print(f"평균 지운 줄: {sum(lines) / len(lines):.1f} (최고: {max(lines)}, 최저: {min(lines)})")
    print(f"평균 보상: {sum(rewards) / len(rewards):.2f}")
    print(f"Average 스텝: {sum(steps) / len(steps):.1f}")

    return results


def test(opt):
    """메인 테스트 함수"""
    model_path = f"{opt.model_dir}/{opt.model_name}"

    print(f"PPO 모델 로딩: {model_path}")

    # 시드 설정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # 모델 불러오기
    model = ActorCritic()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("모델 로딩 성공!")
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return

    # 환경 설정
    env = gym.make("RL-Tetris-v0",
                   randomizer=BagRandomizer(),
                   render_mode="animate" if opt.num_games == 1 else None)
    env = GroupedWrapper(
        env, observation_wrapper=GroupedFeaturesObservation(env))

    try:
        if opt.num_games == 1:
            # 단일 게임 (시각화 포함)
            print("\n단일 게임 테스트 (시각화 포함)")
            result = test_single_game(model, env, deterministic=opt.deterministic)
            print(f"\n최종 결과:")
            print(f"점수: {result['score']}")
            print(f"지운 줄: {result['cleared_lines']}")
            print(f"총 보상: {result['total_reward']:.2f}")
            print(f"스텝 수: {result['steps']}")
        else:
            # 다중 게임 (통계)
            results = test_multiple_games(model, env, opt.num_games, opt.deterministic)

            # 결정론적 vs 확률적 비교를 위해 두 모드 모두 실행
            if opt.deterministic:
                print(f"\n--- 확률적 모드로도 테스트 ---")
                results_stochastic = test_multiple_games(model, env, opt.num_games, deterministic=False)

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    opt = get_args()
    print("PPO 모델 테스트 시작...")
    print(f"설정: {opt.__dict__}")
    test(opt)