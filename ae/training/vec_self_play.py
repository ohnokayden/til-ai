import os
import time
import warnings
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from til_environment.actions import Action
from sb3_contrib import MaskablePPO
from omegaconf import OmegaConf
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_checker import check_env
import copy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import os
import io
import tempfile
from new_wrapper import SelfPlayWrapper,BombermanEnv

cfg = default_config()
cfg = OmegaConf.merge(cfg, {
    "rewards": {
        "step_penalty": -0.01,         # small cost per step
        "stationary_penalty": -0.05,   # discourage staying still
        "agent_collide_wall": -0.1,    # penalise bumping walls
        "collect_mission": 4.0,        # emphasise mission tiles
    },

                      
    "env": {
    "render_mode": "human",
    "grid_size": 16,
    "num_teams": 6,
    "num_iters": 200,
    "novice" : True, # can try changing
    "tile_respawn_steps": 40,
    }
})
def get_selfplay_env(env) -> "SelfPlayWrapper":
    """
    Walk the wrapper stack to find SelfPlayWrapper.
    With the current stack SelfPlayWrapper is already outermost,
    so this is mostly a safety utility for custom stacks.
    """
    w = env
    while w is not None:
        if isinstance(w, SelfPlayWrapper):
            return w
        w = getattr(w, "env", None)
    raise RuntimeError("No SelfPlayWrapper found in wrapper stack")
 
 
def make_env(cfg):
    """
    Returns a SelfPlayWrapper(ActionMasker(BombermanEnv)) env.
    SelfPlayWrapper is outermost so SubprocVecEnv.env_method() can
    call set_opponent_from_file() on it directly.
    """
    def _init():
        return SelfPlayWrapper(BombermanEnv(cfg))
    return _init
 
 
def train_selfplay(
    total_cycles: int = 10,
    steps_per_cycle: int = 1_000_000,
    save_dir: str = "models",
    n_envs: int = 4,
):
    """
    Vectorised self-play training using SubprocVecEnv.
 
    Each subprocess runs its own copy of:
      BombermanEnv -> SelfPlayWrapper -> ActionMasker
 
    At the start of every cycle the current policy is saved to a temp
    file and broadcast to every subprocess via env_method(), which is
    the only way to pass data into forked processes.
 
    Parameters
    ----------
    total_cycles   : number of self-play update rounds
    steps_per_cycle: total env steps across all workers per round
    save_dir       : where checkpoints are saved
    n_envs         : number of parallel workers (tune to your CPU count)
    """
    os.makedirs(save_dir, exist_ok=True)
 
    # --- build vectorised env ---
    # Each worker: BombermanEnv -> ActionMasker -> SelfPlayWrapper
    venv = SubprocVecEnv([make_env(cfg) for _ in range(n_envs)])
    venv = VecMonitor(venv)   # adds episode reward/length logging
 
    # --- initialise model ---
    model = MaskablePPO.load("../models/best_reward/best_reward_model.zip", env =venv, n_steps=1024,batch_size=1024,learning_rate=1e-5, force_reset=True,use_masking=False) 
 
    for cycle in range(total_cycles):
        print(f"\n{'='*50}")
        print(f"  Self-play cycle {cycle + 1}/{total_cycles}  ({n_envs} workers)")
        print(f"{'='*50}")
 
        # Save frozen opponent to a temp file so subprocesses can load it.
        # NamedTemporaryFile with delete=False so the file outlives this
        # scope while subprocesses are loading it; we clean it up after.
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            opp_path = f.name
        model.save(opp_path)
 
        # Broadcast the frozen opponent path to every worker.
        # env_method calls a method by name on the *innermost* env in
        # each subprocess — we need SelfPlayWrapper, so unwrap first.
        venv.env_method("set_opponent_from_file", opp_path)
 
        # Clean up temp file once all workers have loaded it
        os.remove(opp_path)
 
        # Train for N steps spread across all workers
        model.learn(
            total_timesteps=steps_per_cycle,
            reset_num_timesteps=False,
            progress_bar=True,
        )
 
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"cycle_{cycle + 1:03d}")
        model.save(ckpt_path)
        print(f"  Saved checkpoint -> {ckpt_path}.zip")
 
    venv.close()
    return model
 
 
# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────
 
def evaluate(model_path: str, n_episodes: int = 5, render: bool = True):
    """Load a saved model and watch it play against itself."""
    eval_cfg = OmegaConf.merge(cfg, {"env": {"render_mode": "human" if render else None}})
    env = SelfPlayWrapper(BombermanEnv(eval_cfg))
 
    model = MaskablePPO.load(model_path, env=env)
    env.set_opponent(model)  # play against itself
 
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep + 1}: total reward = {total_reward:.2f}")
 
    env.close()
 
 
# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    import multiprocessing
    # Use (cpu_count - 1) workers, leaving one core free for the main process
    n_envs = max(1, multiprocessing.cpu_count() - 1)
 
    trained_model = train_selfplay(
        total_cycles=10,
        steps_per_cycle=50_000,
        save_dir="models/selfplay",
        n_envs=n_envs,
    )
 
    # Optionally watch the final agent
    # evaluate("models/selfplay/cycle_010", n_episodes=3, render=True)