import os
import time
import warnings
from typing import Callable, Optional

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
import gymnasium as gym
import numpy as np


from my_wrapper import BombermanEnv





check_env(BombermanEnv(), warn=True)
# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one episode and collect Bomberman-specific stats
# ─────────────────────────────────────────────────────────────────────────────
 
def _run_episode(env: gym.Env, model) -> dict:
    """
    Run a single episode deterministically and return a stats dict.
 
    The env is assumed to be your BombermanSingleAgentEnv (or ActionMasker
    wrapped version). Observations are already flattened.
    """
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
 
    # Bomberman-specific counters — increment these based on per-step reward
    # signals. We infer events from reward deltas since the raw env info dict
    # may not always be populated in the wrapped env.
    episode_stats = {
        "reward": 0.0,
        "steps": 0,
        "estimated_missions": 0,   # reward spike ~5.0  → mission collected
        "estimated_kills": 0,      # reward spike ~15.0 → kill
        "estimated_base_hits": 0,  # reward spike ~20.0 → base damage
        "estimated_base_destroy": 0,  # reward spike ~50.0 → base destroyed
        "bombs_placed": 0,
    }
 
    prev_reward_sum = 0.0
 
    while not done:
        # Use action masking if available (ActionMasker wrapper exposes this)
        action_masks = None
        if hasattr(env, "action_masks"):
            action_masks = env.action_masks()
 
        if action_masks is not None:
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
        else:
            action, _ = model.predict(obs, deterministic=True)
 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
 
        total_reward += reward
        steps += 1
 
        # Track bomb placement (action index 5 = PLACE_BOMB)
        if int(action) == 5:
            episode_stats["bombs_placed"] += 1
 
        # Infer events from reward magnitude — these are approximate
        # but useful for tracking behaviour trends without modifying the env
        if reward >= 48.0:
            episode_stats["estimated_base_destroy"] += 1
        elif reward >= 12.0:
            episode_stats["estimated_kills"] += 1
        elif reward >= 18.0:
            episode_stats["estimated_base_hits"] += 1
        elif reward >= 4.0:
            episode_stats["estimated_missions"] += 1
 
    episode_stats["reward"] = total_reward
    episode_stats["steps"] = steps
    return episode_stats
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Main callback
# ─────────────────────────────────────────────────────────────────────────────
 
class BombermanEvalCallback(BaseCallback):
    """
    Evaluation callback tailored for the TIL-26 Bomberman AE environment.
 
    Parameters
    ----------
    eval_env_fn : callable
        A zero-argument function that returns a fresh, wrapped single-agent
        BombermanEnv (with ActionMasker applied). Called once at init.
        Example: lambda: ActionMasker(BombermanEnv(), lambda e: e.get_action_mask())
 
    eval_freq : int
        Evaluate every `eval_freq` *training* steps (across all envs).
        With n_envs=4, one update = 4 × n_steps environment steps, so
        eval_freq=10_000 evaluates roughly every 5 updates.
 
    n_eval_episodes : int
        Number of full episodes to run per evaluation. 10 is a good balance
        between reliability and speed for 200-step episodes.
 
    best_model_save_path : str
        Directory to save the best model (by mean episode reward).
 
    combat_model_save_path : str, optional
        Directory to save the best model by combat score (kills + base damage).
        Set None to skip.
 
    log_path : str, optional
        Directory for a CSV log of all eval results. Set None to skip.
 
    plateau_patience : int
        Warn (and optionally anneal ent_coef) if reward hasn't improved
        for this many consecutive evaluations. Default 10.
 
    anneal_entropy_on_plateau : bool
        If True, halve ent_coef when a plateau is detected (down to a floor
        of 0.001). Helps the policy converge when it stops improving.
 
    verbose : int
        0 = silent, 1 = print eval summary per evaluation.
    """
 
    def __init__(
        self,
        eval_env_fn: Callable[[], gym.Env],
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        best_model_save_path: str = "./models/best_reward",
        combat_model_save_path: Optional[str] = "./models/best_combat",
        log_path: Optional[str] = "./ae_logs/eval_results.csv",
        plateau_patience: int = 10,
        anneal_entropy_on_plateau: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.combat_model_save_path = combat_model_save_path
        self.log_path = log_path
        self.plateau_patience = plateau_patience
        self.anneal_entropy_on_plateau = anneal_entropy_on_plateau
 
        # Internal state
        self._eval_env: Optional[gym.Env] = None
        self._best_mean_reward: float = -np.inf
        self._best_combat_score: float = -np.inf
        self._evals_without_improvement: int = 0
        self._eval_count: int = 0
        self._csv_header_written: bool = False
 
    # ── Lifecycle ─────────────────────────────────────────────────────────────
 
    def _init_callback(self) -> None:
        """Called once when training starts. Create the eval env and directories."""
        self._eval_env = self.eval_env_fn()
 
        for path in [self.best_model_save_path, self.combat_model_save_path]:
            if path is not None:
                os.makedirs(path, exist_ok=True)
 
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
 
        if self.verbose >= 1:
            print("[BombermanEvalCallback] Initialised. Eval env created.")
 
    def _on_step(self) -> bool:
        """Called after every environment step during training. Returns False to stop training."""
 
        # Only evaluate every eval_freq steps
        if self.n_calls % self.eval_freq != 0:
            return True
 
        self._eval_count += 1
        t_start = time.time()
 
        # ── Run N evaluation episodes ─────────────────────────────────────────
        all_stats = [
            _run_episode(self._eval_env, self.model)
            for _ in range(self.n_eval_episodes)
        ]
 
        elapsed = time.time() - t_start
 
        # ── Aggregate stats ───────────────────────────────────────────────────
        rewards          = [s["reward"]                   for s in all_stats]
        missions         = [s["estimated_missions"]       for s in all_stats]
        kills            = [s["estimated_kills"]          for s in all_stats]
        base_destroys    = [s["estimated_base_destroy"]   for s in all_stats]
        bombs_placed     = [s["estimated_base_hits"]      for s in all_stats]  # reuse field
 
        mean_reward      = float(np.mean(rewards))
        std_reward       = float(np.std(rewards))
        mean_missions    = float(np.mean(missions))
        mean_kills       = float(np.mean(kills))
        mean_destroys    = float(np.mean(base_destroys))
        mean_bombs       = float(np.mean(bombs_placed))
 
        # Combat score: weighted sum of aggressive behaviours
        # Tune these weights based on what you want to incentivise
        combat_score = mean_kills * 2.0 + mean_destroys * 10.0 + mean_missions * 0.5
 
        # ── Log to TensorBoard ────────────────────────────────────────────────
        self.logger.record("eval/mean_reward",          mean_reward)
        self.logger.record("eval/std_reward",           std_reward)
        self.logger.record("eval/mean_missions",        mean_missions)
        self.logger.record("eval/mean_kills",           mean_kills)
        self.logger.record("eval/mean_base_destroys",   mean_destroys)
        self.logger.record("eval/mean_bombs_placed",    mean_bombs)
        self.logger.record("eval/combat_score",         combat_score)
        self.logger.record("eval/evals_without_improvement", self._evals_without_improvement)
        self.logger.record("eval/elapsed_seconds",      elapsed)
 
        # Dump to TensorBoard immediately (don't wait for next training log)
        self.logger.dump(self.num_timesteps)
 
        # ── Save best models ──────────────────────────────────────────────────
        if mean_reward > self._best_mean_reward:
            self._best_mean_reward = mean_reward
            self._evals_without_improvement = 0
            if self.best_model_save_path is not None:
                path = os.path.join(self.best_model_save_path, "best_reward_model")
                self.model.save(path)
                if self.verbose >= 1:
                    print(f"  ✓ New best reward model saved: {mean_reward:.2f} → {path}")
        else:
            self._evals_without_improvement += 1
 
        if combat_score > self._best_combat_score:
            self._best_combat_score = combat_score
            if self.combat_model_save_path is not None:
                path = os.path.join(self.combat_model_save_path, "best_combat_model")
                self.model.save(path)
                if self.verbose >= 1:
                    print(f"  ✓ New best combat model saved: score={combat_score:.2f} → {path}")
 
        # ── Plateau detection and entropy annealing ───────────────────────────
        if self._evals_without_improvement >= self.plateau_patience:
            if self.verbose >= 1:
                print(
                    f"  ⚠ Plateau detected: no reward improvement for "
                    f"{self._evals_without_improvement} evals."
                )
            if self.anneal_entropy_on_plateau:
                self._anneal_entropy()
            # Reset counter so it doesn't fire every step after plateau
            self._evals_without_improvement = 0
 
        # ── Write CSV log ─────────────────────────────────────────────────────
        if self.log_path is not None:
            self._write_csv_row(mean_reward, std_reward, mean_missions,
                                mean_kills, mean_destroys, combat_score)
 
        # ── Print summary ─────────────────────────────────────────────────────
        if self.verbose >= 1:
            print(
                f"[Eval #{self._eval_count} | step {self.num_timesteps:,}] "
                f"reward={mean_reward:.1f}±{std_reward:.1f} | "
                f"missions={mean_missions:.1f} | kills={mean_kills:.1f} | "
                f"base_destroys={mean_destroys:.1f} | "
                f"combat_score={combat_score:.1f} | "
                f"best={self._best_mean_reward:.1f} | "
                f"elapsed={elapsed:.1f}s"
            )
 
        return True  # returning False would stop training
 
    def _on_training_end(self) -> None:
        """Clean up eval env when training finishes."""
        if self._eval_env is not None:
            self._eval_env.close()
        if self.verbose >= 1:
            print(
                f"[BombermanEvalCallback] Training ended. "
                f"Best reward: {self._best_mean_reward:.2f} | "
                f"Best combat score: {self._best_combat_score:.2f}"
            )
 
    # ── Helpers ───────────────────────────────────────────────────────────────
 
    def _anneal_entropy(self) -> None:
        """
        Halve the model's ent_coef, down to a minimum of 0.001.
        This lets the policy converge when it has stopped improving from entropy.
        """
        current = self.model.ent_coef
        # ent_coef can be a float or a schedule — only anneal if it's a float
        if not isinstance(current, float):
            return
        new_val = max(current * 0.5, 0.001)
        self.model.ent_coef = new_val
        if self.verbose >= 1:
            print(f"  ↓ ent_coef annealed: {current:.4f} → {new_val:.4f}")
 
    def _write_csv_row(self, mean_reward, std_reward, missions, kills, destroys, combat) -> None:
        """Append one row to the CSV log file."""
        mode = "a"
        if not self._csv_header_written:
            mode = "w"
            self._csv_header_written = True
        with open(self.log_path, mode) as f:
            if mode == "w":
                f.write("timestep,eval_num,mean_reward,std_reward,"
                        "mean_missions,mean_kills,mean_base_destroys,combat_score\n")
            f.write(
                f"{self.num_timesteps},{self._eval_count},{mean_reward:.4f},"
                f"{std_reward:.4f},{missions:.2f},{kills:.2f},{destroys:.2f},{combat:.2f}\n"
            )
 
 

 
    # --- Import your wrapper (defined in your training file) ---
    # from my_wrapper import BombermanEnv
 
def make_env():
    env = BombermanEnv()  # your single-agent wrapper
    env = ActionMasker(env, lambda e: e.get_action_mask())
    return env

# Training envs (vectorised, reward-normalised)
vec_env = make_vec_env(make_env, n_envs=4)
vec_env = VecNormalize(vec_env, norm_reward=True, clip_reward=10.0, norm_obs=False)

# # Learning rate schedule: linear decay from 3e-4 → 0
# def lr_schedule(progress_remaining):
#     return 3e-4 * progress_remaining

model = MaskablePPO(
    policy="MlpPolicy",
    env=vec_env,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[512, 512])),
    verbose=1,
    tensorboard_log="./ae_logs/",
)

# Eval callback — uses a fresh (non-normalised) env for evaluation
# so reward values are interpretable in the original scale
eval_callback = BombermanEvalCallback(
    eval_env_fn=make_env,            # fresh env, no VecNormalize
    eval_freq=10_000,
    n_eval_episodes=10,
    best_model_save_path="./models/best_reward",
    combat_model_save_path="./models/best_combat",
    log_path="./ae_logs/eval_results.csv",
    plateau_patience=10,             # warn after 10 evals with no improvement
    anneal_entropy_on_plateau=True,  # automatically halve ent_coef on plateau
    verbose=1,
)

model.learn(total_timesteps=2_000_000, callback=eval_callback)

model.save("./models/best_reward/best_reward_model")

