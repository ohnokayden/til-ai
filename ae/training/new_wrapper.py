import gymnasium as gym
import numpy as np
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from til_environment.actions import Action
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from omegaconf import OmegaConf
import os
import io
import tempfile
 
 
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
 
cfg = default_config()
cfg = OmegaConf.merge(cfg, {
    "rewards": {
        "step_penalty": -0.01,
        "stationary_penalty": -0.05,
        "agent_collide_wall": -0.1,
        "collect_mission": 4.0,
    },
    "env": {
        "render_mode": None,          # set "human" only for watching
        "grid_size": 16,
        "num_teams": 6,
        "num_iters": 200,
        "novice": True,
        "tile_respawn_steps": 40,
    }
})
 
 
# ─────────────────────────────────────────────
# Base single-agent wrapper around PettingZoo AEC env
# ─────────────────────────────────────────────
 
class BombermanEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
 
    def __init__(self, cfg=None):
        self.cfg = cfg
        self._env = Bomberman(cfg)
        self.controlled = "agent_0"
 
        raw_space = self._env.observation_space(self.controlled)
        flat_dim = gym.spaces.utils.flatdim(raw_space)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (flat_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(6)
 
    def reset(self, seed=None, options=None):
        self._env.reset(seed=seed)
        self._skip_to_agent()
        obs = self._env.observe(self.controlled)
        return self._flat(obs), {}
 
    def step(self, action):
        # Apply action mask — fall back to STAY (4) if action is illegal
        mask = self._env.observe(self.controlled)["action_mask"]
        if not mask[action]:
            action = 4
        self._env.step(int(action))
 
        # Step all agents that come before agent_0 in this round
        for agent in self._env.agent_iter():
            if agent == self.controlled:
                break
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)
 
        obs, rew, term, trunc, info = self._env.last()
        return self._flat(obs), rew, term, trunc, info
 
    def _skip_to_agent(self):
        """Advance AEC env until it's agent_0's turn."""
        for agent in self._env.agent_iter():
            if agent == self.controlled:
                return
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)
 
    def _flat(self, obs):
        sp = self._env.observation_space(self.controlled)
        return gym.spaces.utils.flatten(sp, obs).astype(np.float32)
 
    def get_action_mask(self):
        return self._env.observe(self.controlled)["action_mask"]
 
    def close(self):
        self._env.close()
 
 
 
 
# ─────────────────────────────────────────────
# Self-play wrapper
# ─────────────────────────────────────────────
 
class SelfPlayWrapper(gym.Wrapper):
    """
    Outermost wrapper. Internally wraps ActionMasker(BombermanEnv) so
    that SB3/SubprocVecEnv always talks to SelfPlayWrapper directly,
    meaning env_method("set_opponent_from_file", ...) works without
    any unwrapping.
 
    Stack (internal):
      BombermanEnv  ->  ActionMasker  ->  SelfPlayWrapper  <- SB3 sees this
 
    Usage
    -----
    env = SelfPlayWrapper(BombermanEnv(cfg))
    env.set_opponent(model)
    """
 
    def __init__(self, env: BombermanEnv):
        # Wrap BombermanEnv in ActionMasker first, then super().__init__
        masked = ActionMasker(env, lambda e: e.get_action_mask())
        super().__init__(masked)
        self.opp_model = None
        # Expose the Dict observation space from ActionMasker
        self.observation_space = masked.observation_space
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def set_opponent_from_file(self, path: str):
        """
        Load a frozen opponent from a file path.
        Used by SubprocVecEnv.env_method() which can only pass
        serialisable arguments (strings) into subprocesses.
        """
        self.opp_model = MaskablePPO.load(path)
 
    def set_opponent(self, model):
        """
        Freeze a copy of `model` to use as the opponent.
        Uses save/load via an in-memory buffer instead of deepcopy,
        which would fail trying to pickle pygame surfaces.
        Pass None to fall back to random play.
        """
        if model is None:
            self.opp_model = None
        else:
            buf = io.BytesIO()
            model.save(buf)
            buf.seek(0)
            self.opp_model = MaskablePPO.load(buf)
 
    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------
 
    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
 
    def _get_base(self) -> "BombermanEnv":
        """self.env is ActionMasker; self.env.env is BombermanEnv."""
        return self.env.env
 
    def step(self, action):
        base = self._get_base()
        aec = base._env                          # PettingZoo AEC env
 
        # 1. Apply action mask for controlled agent
        mask = aec.observe(base.controlled)["action_mask"]
        if not mask[action]:
            action = 4
        aec.step(int(action))
 
        # 2. Step every other agent in correct AEC order
        for agent in aec.agent_iter():
            if agent == base.controlled:
                break                            # back to our turn — stop
 
            obs_dict, _, term, trunc, _ = aec.last()
 
            if term or trunc:
                a = None
            elif self.opp_model is not None:
                # Opponent policy expects flat obs + mask (trained on BombermanEnv)
                flat_obs = base._flat(obs_dict)
                opp_mask = obs_dict.get("action_mask",
                                        np.ones(base.action_space.n, dtype=np.int8))
                a, _ = self.opp_model.predict(
                    flat_obs,
                    action_masks=opp_mask,
                    deterministic=True,
                )
            else:
                a = aec.action_space(agent).sample()
 
            aec.step(a)
 
        # 3. Return plain flat obs — ActionMaskWrapper on top will add the mask
        obs_dict, rew, term, trunc, info = aec.last()
        return base._flat(obs_dict), rew, term, trunc, info
 
    def get_action_mask(self):
        return self.env.env.get_action_mask()   # ActionMasker -> BombermanEnv
 
    def action_masks(self):
        """Called by MaskablePPO directly on the outermost env."""
        return self.get_action_mask()
 
    def close(self):
        self.env.close()