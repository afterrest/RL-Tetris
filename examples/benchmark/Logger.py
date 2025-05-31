from pathlib import Path
import time
from typing import Sequence, Optional
from tensorboardX import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
class TetrisLogger:
    """
    ├─ agent_prefix : dqn/*
    ├─ agent_prefix : ppo/*
    └─ 공통 : episode/* , perf/* , schedule/*
    """

    def __init__(
        self, 
        log_dir: str | Path, 
        agent_prefix: str
    ) -> None:
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.t0 = time.time()
        
        self.agent_prefix = agent_prefix
        
        
    # 에피소드 끝날 때 호출
    def log_episode(
        self,
        score: int,
        cleared_lines: int,
        episode_len: int,
        total_env_steps: int,
        epoch: int,
    ) -> None:
        w = self.writer
        w.add_scalar(f"{self.agent_prefix}/score", score, epoch)
        w.add_scalar(f"{self.agent_prefix}/cleared_lines", cleared_lines, epoch)
        w.add_scalar("episode/episode_length", episode_len, epoch)
        w.add_scalar("episode/score", score, epoch)
        w.add_scalar("step/episode_length", episode_len, total_env_steps)

    # DQN : epoch 마지막 부분에 호출
    def log_dqn_step(
        self,
        td_loss: float,
        q_values: Sequence[float],
        target_q_values: Sequence[float],
        epsilon: float,
        epoch: int            # epoch
    ) -> None:
        w = self.writer
        w.add_scalar("dqn/td_loss", td_loss, epoch)
        w.add_scalar("dqn/q_value", q_values.mean().item(), epoch)
        w.add_scalar("dqn/target_q_value", target_q_values.mean().item(), epoch)
        w.add_scalar("dqn/epsilon", epsilon, epoch)
        w.add_histogram("dqn/q_hist", q_values, epoch)
        
    # PPO : epoch 마지막 부분에 호출
    def log_ppo_step(
        self,
        entropy: float,
        approx_kl: float,
        clip_frac: float,
        old_approx_kl: float,
        policy_loss: float,
        value_loss: float,
        explained_variance: float,      
        lr: float,                      # leraning rate
        epoch: int                # epoch
    ) -> None:
        w = self.writer
        tmp = {
                "ppo/entropy":              entropy,
                "ppo/approx_kl":            approx_kl,
                "ppo/clip_fraction":        clip_frac,
                "ppo/old_approx_kl":        old_approx_kl,
                "ppo/policy_loss":          policy_loss,
                "ppo/value_loss":           value_loss,
                "ppo/explained_variance":   explained_variance,
        }
        for k, v in tmp.items():
            w.add_scalar(k, v, epoch)
        w.add_scalar("schedule/lr", lr, epoch)

    def log_perf(self, epoch: int) -> None:
        elapsed = time.time() - self.t0
        sps = epoch / elapsed if elapsed > 0 else 0.0
        self.writer.add_scalar("perf/sps", sps, epoch)

    def flush_every(self, epoch: int, interval: int = 1_000) -> None:
        if epoch % interval == 0:
            self.writer.flush()

    def close(self) -> None:
        self.writer.close()
 