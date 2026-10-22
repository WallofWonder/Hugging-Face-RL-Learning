"""
@Created: 2026/10/8 11:18
@Author: Zhang Yifan
@File: unit3_with_optuna.py
@Description: 
"""
import sys
from typing import Any

import optuna
from rl_zoo3 import hyperparams_opt
from rl_zoo3.train import train


def sample_dqn_atari(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    # CNN 结构、buffer_size、gradient_steps、learning_starts 固定走 dqn.yml / -params
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": 1 - trial.suggest_float("one_minus_gamma", 1e-3, 3e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "train_freq": trial.suggest_categorical("train_freq", [4, 8]),
        "target_update_interval": trial.suggest_int("target_update_interval", 500, 10_000, log=True),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
    }


hyperparams_opt.HYPERPARAMS_SAMPLER["dqn"] = sample_dqn_atari
sys.argv = ["train",
            "--algo", "dqn",
            "--env", "SpaceInvadersNoFrameskip-v4",
            "-conf", "../unit3/dqn.yml",
            "-params", "learning_starts:20000",  # 覆盖 dqn.yml 里的超参 learning_starts
            "-optimize",  # 进入超参搜索模式
            "-n", "500000",  # 每个 trial 训练 5e5 步，覆盖 dqn.yml 里的 1e7，以实现在小型训练上调参
            "--n-trials", "40", # 当前这个进程最多跑 40 个 trial（包括被剪枝的trial）
            "--sampler", "tpe",  # TPESampler
            "--pruner", "median",  # MedianPruner
            "--n-startup-trials", "8", # 同时作用于采样器和剪枝器：前 8 个 trial 用随机采样（TPE 还没有足够的历史可以建模），而且在 8 个 trial 完成前不剪枝
            "--n-evaluations", "5", # 每个 trial 训练过程中评估 5 次
            "--eval-episodes", "5",  # 每次评估跑 5 局，取平均分
            "--n-eval-envs", "5",  # 评估环境开 5 个副本，5 局分摊到 5 个环境上
            "--device", "mps",
            "--storage", "sqlite:///optuna_dqn_si.db",  # 把 study 存到 sqlite 文件。/// 表示相对当前目录的路径，//// 表示绝对路径。
            "--study-name", "dqn_si",  # study 在数据库里的名字。续跑或多进程共享时必须一致，换个名字就是一个新的 study
            "-f", "logs/optuna"]
train()