import gymnasium as gym
import numpy as np
from til_environment.bomberman_env import Bomberman
from til_environment.config import default_config
from til_environment.actions import Action
from omegaconf import OmegaConf


cfg = default_config()
cfg = OmegaConf.merge(cfg, {
    "rewards": {
        "step_penalty": -0.01,         # small cost per step
        "stationary_penalty": -0.05,   # discourage staying still
        "agent_collide_wall": -0.1,    # penalise bumping walls
        "collect_mission": 10.0,       # emphasise mission tiles
    }
})


class BombermanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._env = Bomberman(self.cfg)
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
        mask = self._env.observe(self.controlled)["action_mask"]
        if not mask[action]: action = 4
        self._env.step(int(action))
        self._step_others()
        obs, rew, term, trunc, info = self._env.last()
        return self._flat(obs), rew, term, trunc, info

    def get_action_mask(self):
        obs = self._env.observe(self.controlled)
        return obs["action_mask"].astype(bool)

    def _skip_to_agent(self):
        for agent in self._env.agent_iter():
            if agent == self.controlled: return
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _step_others(self):
        for agent in self._env.agents:
            if agent == self.controlled: continue
            obs, _, term, trunc, _ = self._env.last()
            a = None if (term or trunc) else self._env.action_space(agent).sample()
            self._env.step(a)

    def _flat(self, obs):
        sp = self._env.observation_space(self.controlled)
        return gym.spaces.utils.flatten(sp, obs).astype(np.float32)

    def close(self): self._env.close()
