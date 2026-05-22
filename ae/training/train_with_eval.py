import os
import time
import warnings
from typing import Callable, Optional

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
import gymnasium as gym
from supersuit import frame_stack_v2
import numpy as np


from my_wrapper import BombermanEnv, SelfPlayWrapper
from eval import BombermanEvalCallback





check_env(BombermanEnv(), warn=True)
# check_env(SelfPlayWrapper(), warn=True)

    # --- Import your wrapper (defined in your training file) ---
    # from my_wrapper import BombermanEnv
 
def make_env():
    env = BombermanEnv()
    check_env(env, warn=True)
    # your single-agent wrapper
    # frame stacking wrapper
    from gymnasium.wrappers import FrameStackObservation, FlattenObservation
    env = FrameStackObservation(env, stack_size=4)
    env = FlattenObservation(env)
    env = SelfPlayWrapper(env)
    env = ActionMasker(env, lambda e: e.get_action_mask())
    check_env(env, warn=True)
    return env
    

if __name__ == "__main__":
# Training envs (vectorised, reward-normalised)
    vec_env = make_vec_env(make_env, n_envs=8, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_reward=True, clip_reward=10.0, norm_obs=False)
    
    # # Learning rate schedule: linear decay from 3e-4 → 0
    # def lr_schedule(progress_remaining):
    #     return 3e-4 * progress_remaining
    
    model = MaskablePPO.load("../models/best_reward/best_reward_model.zip", env = vec_env, learning_rate=1e-5, force_reset=True,) 
    # # model = MaskablePPO(
    # #     policy="MlpPolicy",
    # #     env=vec_env,
    # #     n_steps=8192,
    # #     batch_size=1024,
    # #     n_epochs=10,
    # #     gamma=0.99,
    # #     gae_lambda=0.95,
    # #     clip_range=0.2,
    # #     ent_coef=0.003,
    # #     device="cpu",
    # #     learning_rate=3e-5,
    # #     policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
    # #     verbose=1,
    # #     tensorboard_log="./ae_logs/",
    # # )
    
    # Eval callback — uses a fresh (non-normalised) env for evaluation
    # so reward values are interpretable in the original scale
    eval_callback = BombermanEvalCallback(
        eval_env_fn=make_env,            # fresh env, no VecNormalize
        eval_freq=10_000,
        n_eval_episodes=10,
        best_model_save_path="../models/best_reward",
        combat_model_save_path="../models/best_combat",
        log_path="./ae_logs/eval_results.csv",
        plateau_patience=10,             # warn after 10 evals with no improvement
        anneal_entropy_on_plateau=True,  # automatically halve ent_coef on plateau
        verbose=1,
    )
    
    model.learn(total_timesteps = 5_000_000, callback=eval_callback, use_masking=True)
    # model.learn(total_timesteps = 1_500_000, use_masking=True,)
    # save_path = "../models/best_reward/best_reward_model"
    model.save("../models/best_reward/best_reward_model_final") # only on if eval is off
    print(f"saved to {save_path}")

